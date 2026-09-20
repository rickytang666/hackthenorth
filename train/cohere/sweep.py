"""Capacity sweep: does more LoRA rank or more encoder depth buy accuracy?

Scored on F04, the validation speaker, ONLY. M02 is sealed and is not touched
here: re-decoding it to pick a config would be a second look at test data.

Each config trains from scratch for a fixed step count, then decodes F04 and
reports WER, which is the metric that matters. Dev loss is reported too because
the two can disagree.

    uv run python -m train.cohere.sweep
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from contract import manifest, predictions
from contract.evaluate import score
from train.cohere.data import SpeechSeq2SeqCollator, TorgoDataset
from train.cohere.model import MODEL_ID, load, prompt_ids
from train.cohere.preflight import target_modules

SAMPLE_RATE = 16000

# (rank, top encoder blocks). The first row reproduces the shipped adapter so
# every other row is compared against a number measured in the same job, on the
# same box, rather than against a remembered one.
CONFIGS = [
    (8, 6),     # shipped: dev WER 0.0330
    (32, 6),    # win 1: more capacity, same depth
    (8, 16),    # win 2: same capacity, more depth
    (32, 16),   # both
]


def decode_f04(model, processor, prompt, device, rows, model_id):
    model.eval()
    out = []
    for row in rows:
        audio, _ = sf.read(row["audio_filepath"], dtype="float32")
        inputs = processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           language="en", punctuation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        began = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, decoder_input_ids=prompt, max_new_tokens=64)
        out.append({
            "audio_filepath": row["audio_filepath"], "speaker_id": row["speaker_id"],
            "text": row["text"], "duration": row["duration"],
            "prediction": processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip(),
            "model_id": model_id, "latency_ms": 1000 * (time.perf_counter() - began),
        })
    model.train()
    return out


@torch.no_grad()
def dev_loss(model, loader, device, limit=40):
    model.eval()
    total = n = 0
    for batch in loader:
        if n >= limit:
            break
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        total += model(**batch).loss.item(); n += 1
    model.train()
    return total / max(1, n)


def run(rank, blocks, args, processor, prompt, device, dtype, loaders, dev_rows, out_dir):
    began = time.perf_counter()
    model = load(dtype)
    targets = target_modules(model, blocks)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        target_modules=targets))
    model = model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n=== r={rank} blocks={blocks}: {len(targets)} modules, "
          f"{trainable:,}/{total:,} = {100*trainable/total:.3f}% ===", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, 20, args.steps)
    train_loader, dev_loader = loaders

    step = 0
    while step < args.steps:
        for micro, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            (model(**batch).loss / args.grad_accum).backward()
            if (micro + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{args.steps}", flush=True)
            if step >= args.steps:
                break

    loss = dev_loss(model, dev_loader, device)
    tag = f"r{rank}_b{blocks}"
    rows = decode_f04(model, processor, prompt, device, dev_rows, f"{MODEL_ID}+{tag}")
    predictions.write(out_dir / f"sweep_{tag}.jsonl", rows)
    s = score(rows)
    minutes = (time.perf_counter() - began) / 60
    print(f"  r={rank} blocks={blocks}: F04 WER {s['wer']:.4f}  "
          f"words {s['wer_isolated_word']:.4f}  sentences {s['wer_sentence']:.4f}  "
          f"dev loss {loss:.4f}  ({minutes:.1f} min)", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"rank": rank, "blocks": blocks, "trainable": trainable, "pct": 100*trainable/total,
            "wer": s["wer"], "wer_word": s["wer_isolated_word"],
            "wer_sentence": s["wer_sentence"], "dev_loss": loss, "minutes": minutes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="results/sweep")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    collate = SpeechSeq2SeqCollator(processor)
    prompt = torch.tensor([prompt_ids(processor.tokenizer, "en", punctuation=False)],
                          dtype=torch.long, device=device)
    train_loader = DataLoader(TorgoDataset("torgo_dys_train.jsonl"), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    dev_loader = DataLoader(TorgoDataset("torgo_dys_dev.jsonl"), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=2)
    dev_rows = manifest.load("torgo_dys_dev.jsonl", resolve=True)

    results = []
    for rank, blocks in CONFIGS:
        results.append(run(rank, blocks, args, processor, prompt, device, dtype,
                           (train_loader, dev_loader), dev_rows, out_dir))
        (out_dir / "sweep_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== capacity sweep, scored on F04 (validation speaker) ===")
    print("| Rank | Enc blocks | Trainable | % of model | F04 WER | Words | Sentences | Dev loss | Min |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['rank']} | {r['blocks']} | {r['trainable']:,} | {r['pct']:.3f}% | "
              f"**{r['wer']:.4f}** | {r['wer_word']:.4f} | {r['wer_sentence']:.4f} | "
              f"{r['dev_loss']:.4f} | {r['minutes']:.1f} |")
    best = min(results, key=lambda r: r["wer"])
    shipped = results[0]
    gain = 100 * (shipped["wer"] - best["wer"]) / shipped["wer"] if shipped["wer"] else 0
    print(f"\nbest: r={best['rank']} blocks={best['blocks']} at WER {best['wer']:.4f}")
    print(f"shipped config r=8 blocks=6 scored {shipped['wer']:.4f} here")
    print(f"headroom found: {gain:.1f}% relative on F04")
    print("\nM02 was NOT decoded. Choosing a config on F04 is legitimate; re-decoding")
    print("the sealed speaker to report a better headline would not be.")


if __name__ == "__main__":
    main()
