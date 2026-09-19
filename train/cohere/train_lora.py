"""Cohere Transcribe supervised LoRA on dysarthric TORGO.

Audio-to-transcript cross-entropy. Not DPO, not an LLM text objective.

Everything load-bearing was verified by train/cohere/preflight.py against
revision b1eacc26: the module names, the 91.8% encoder parameter split, the
teacher-forcing shift, and the language prompt prefix.

    uv run python -m train.cohere.train_lora --max-steps 400
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from train.cohere.data import SpeechSeq2SeqCollator, TorgoDataset
from train.cohere.model import MODEL_ID, load
from train.cohere.preflight import encoder_block_index, target_modules

LORA_R = 8
TOP_ENCODER_BLOCKS = 6


def build(top_blocks: int, dtype):
    model = load(dtype)
    targets = target_modules(model, top_blocks)
    if not targets:
        raise SystemExit("no LoRA targets resolved; run preflight first")
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
        bias="none", target_modules=targets,
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA on {len(targets)} modules, trainable {trainable:,}/{total:,} "
          f"({100*trainable/total:.3f}%)")
    return model


@torch.no_grad()
def dev_loss(model, loader, device, limit: int = 40) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        if n >= limit:
            break
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        total += model(**batch).loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--top-blocks", type=int, default=TOP_ENCODER_BLOCKS)
    parser.add_argument("--probe-steps", type=int, default=0,
                        help="run N steps, report throughput, exit. Sets max_steps honestly")
    parser.add_argument("--out", default=os.environ.get("BT_CHECKPOINT_DIR", "checkpoints/cohere"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    collate = SpeechSeq2SeqCollator(processor)
    train_loader = DataLoader(TorgoDataset("torgo_dys_train.jsonl"), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    dev_loader = DataLoader(TorgoDataset("torgo_dys_dev.jsonl"), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=2)

    model = build(args.top_blocks, dtype).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup, args.max_steps)

    step = 0
    began = time.perf_counter()
    history = []
    stop = args.probe_steps or args.max_steps

    while step < stop:
        for micro, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (micro + 1) % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                seen = step * args.batch_size * args.grad_accum
                rate = seen / (time.perf_counter() - began)
                print(f"step {step:4}/{stop}  loss {loss.item()*args.grad_accum:.4f}  "
                      f"{rate:.1f} ex/s", flush=True)

            if args.probe_steps and step >= args.probe_steps:
                break
            if step % args.eval_every == 0:
                value = dev_loss(model, dev_loader, device)
                history.append({"step": step, "dev_loss": value})
                print(f"  dev loss at step {step}: {value:.4f}", flush=True)
                # The 200-step kill rule lives in the queue, not here: this
                # prints the number the human decides on.
                model.save_pretrained(out / f"step-{step}")
                (out / "history.json").write_text(json.dumps(history, indent=2))
            if step >= stop:
                break

    elapsed = time.perf_counter() - began
    seen = step * args.batch_size * args.grad_accum
    print(f"\n{step} steps, {seen} examples, {elapsed/60:.1f} min, {seen/elapsed:.2f} ex/s")
    if args.probe_steps:
        per_step = elapsed / step
        print(f"\nPROBE: {per_step:.2f} s/step on {device}.")
        for budget_h in (1, 2, 3):
            print(f"  {budget_h}h budget -> {int(budget_h*3600/per_step)} steps")
        print("Pick max_steps from this, not from the plan's 400.")
        return

    model.save_pretrained(out / "best")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved adapter to {out/'best'}")


if __name__ == "__main__":
    main()
