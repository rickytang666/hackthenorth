"""Leave-one-speaker-out across all eight dysarthric TORGO speakers.

Decoding the tuned adapter on its own training speakers would be meaningless,
so each fold trains a fresh adapter excluding one speaker and scores only that
speaker. Every score below is therefore a genuine unseen-speaker result.

Each fold trains a FIXED step count, chosen once from the original run (dev loss
bottomed at 900 of 1200). No fold makes any decision using its own held-out
speaker, which is what keeps the folds honest.

    uv run python -m train.cohere.loso --steps 900
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from contract import manifest, predictions
from contract.evaluate import score
from contract.normalize import normalize
from train.cohere.data import SpeechSeq2SeqCollator, TorgoDataset
from train.cohere.model import MODEL_ID, load, prompt_ids
from train.cohere.preflight import target_modules

SPEAKERS = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
LORA_R = 8
SAMPLE_RATE = 16000


class SpeakerSubset(torch.utils.data.Dataset):
    """TorgoDataset filtered to exclude (or keep) given speakers."""

    def __init__(self, base: TorgoDataset, speakers: set[str], exclude: bool):
        self.base = base
        self.idx = [i for i, r in enumerate(base.rows)
                    if (r["speaker_id"] not in speakers) == exclude]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def decode_rows(model, processor, prompt, device, rows, model_id):
    import soundfile as sf
    out = []
    model.eval()
    for row in rows:
        audio, _ = sf.read(row["audio_filepath"], dtype="float32")
        inputs = processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           language="en", punctuation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        began = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, decoder_input_ids=prompt, max_new_tokens=64)
        text = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        out.append({**{k: row[k] for k in ("audio_filepath", "speaker_id", "text", "duration")},
                    "prediction": text, "model_id": model_id,
                    "latency_ms": 1000 * (time.perf_counter() - began)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top-blocks", type=int, default=6)
    parser.add_argument("--speakers", nargs="*", default=SPEAKERS)
    parser.add_argument("--out", default="results/loso")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    collate = SpeechSeq2SeqCollator(processor)
    prompt = torch.tensor([prompt_ids(processor.tokenizer, "en", punctuation=False)],
                          dtype=torch.long, device=device)
    everything = TorgoDataset("torgo_dys_allspeakers.jsonl")
    all_rows = manifest.load("torgo_dys_allspeakers.jsonl", resolve=True)

    # The frozen baseline is decoded once for every speaker: it saw none of them,
    # so one pass serves as the before column for all eight folds.
    base_model = load(dtype).eval().to(device)
    baseline = decode_rows(base_model, processor, prompt, device, all_rows, MODEL_ID)
    predictions.write(out / "baseline_allspeakers.jsonl", baseline)
    del base_model
    torch.cuda.empty_cache()
    print(f"baseline decoded over {len(baseline)} clips\n", flush=True)

    results = {}
    for held in args.speakers:
        began = time.perf_counter()
        print(f"=== fold {held}: train on the other seven ===", flush=True)
        train_set = SpeakerSubset(everything, {held}, exclude=True)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate, num_workers=2, drop_last=True)

        model = load(dtype)
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none",
            target_modules=target_modules(model, args.top_blocks)))
        model = model.to(device).train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, 20, args.steps)

        step = 0
        while step < args.steps:
            for micro, batch in enumerate(loader):
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                (model(**batch).loss / args.grad_accum).backward()
                if (micro + 1) % args.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    print(f"  {held} step {step}/{args.steps}", flush=True)
                if step >= args.steps:
                    break

        rows = [r for r in all_rows if r["speaker_id"] == held]
        tuned = decode_rows(model, processor, prompt, device,
                            rows, f"{MODEL_ID}+loso_{held}")
        predictions.write(out / f"tuned_{held}.jsonl", tuned)

        b = score([r for r in baseline if r["speaker_id"] == held])
        t = score(tuned)
        results[held] = {"n": len(rows), "base": b, "tuned": t,
                         "minutes": (time.perf_counter() - began) / 60}
        print(f"  {held}: base WER {b['wer']:.4f} -> tuned {t['wer']:.4f} "
              f"({100*(b['wer']-t['wer'])/b['wer']:.1f}% relative) "
              f"in {results[held]['minutes']:.1f} min\n", flush=True)
        (out / "loso_results.json").write_text(json.dumps(results, indent=2, default=float))

        del model
        torch.cuda.empty_cache()

    print("\n=== leave-one-speaker-out, all folds ===")
    print("| Speaker | Clips | Base WER | Tuned WER | Relative | Critical base | Critical tuned |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s, r in results.items():
        rel = 100 * (r["base"]["wer"] - r["tuned"]["wer"]) / r["base"]["wer"] if r["base"]["wer"] else 0
        print(f"| {s} | {r['n']} | {r['base']['wer']:.4f} | {r['tuned']['wer']:.4f} | "
              f"{rel:.1f}% | {r['base']['critical_error_rate']:.4f} | "
              f"{r['tuned']['critical_error_rate']:.4f} |")
    wers = [r["tuned"]["wer"] for r in results.values()]
    bases = [r["base"]["wer"] for r in results.values()]
    print(f"\nmean tuned WER {sum(wers)/len(wers):.4f}, range {min(wers):.4f} to {max(wers):.4f}")
    print(f"mean base  WER {sum(bases)/len(bases):.4f}, range {min(bases):.4f} to {max(bases):.4f}")


if __name__ == "__main__":
    main()
