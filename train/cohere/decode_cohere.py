"""Decode Cohere Transcribe and emit prediction rows plus real confidence.

Used twice: the Phase 0 gate (a handful of clips, proving confidence.py works on
real token log-probabilities) and the dev/test decodes that fill the scorecard.

    uv run python -m train.cohere.decode_cohere --manifest torgo_dys_dev.jsonl --limit 4
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor

from contract import manifest, predictions
from contract.confidence import from_logprobs
from train.cohere.model import MODEL_ID, load, prompt_ids

SAMPLE_RATE = 16000


def token_logprobs(scores, sequences, prompt_len: int) -> list[float]:
    """Per-token log-probabilities of the tokens the model actually emitted.

    NVIDIA's confidence utility is broken on TDT and Cohere ships none, so both
    lanes derive this themselves and feed contract.confidence. Identical maths
    on both sides is what makes the clarification rates comparable.
    """
    out = []
    for step, step_scores in enumerate(scores):
        logprobs = torch.log_softmax(step_scores[0].float(), dim=-1)
        index = prompt_len + step
        if index >= sequences.shape[1]:
            break
        out.append(logprobs[sequences[0, index]].item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="torgo_dys_dev.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shortest", action="store_true", help="pick the shortest clips")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    rows = manifest.load(args.manifest, resolve=True)
    if args.shortest:
        rows = sorted(rows, key=lambda r: r["duration"])
    if args.limit:
        rows = rows[: args.limit]

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load(dtype, args.adapter).eval().to(device)
    # Without this the model does no language ID and drifts into other scripts.
    prompt = torch.tensor([prompt_ids(processor.tokenizer, "en", punctuation=False)],
                          dtype=torch.long, device=device)
    model_id = MODEL_ID + (f"+{Path(args.adapter).name}" if args.adapter else "")

    # Warm once. A cold first call is not steady-state latency.
    warm_audio, _ = sf.read(rows[0]["audio_filepath"], dtype="float32")
    warm = processor([warm_audio], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                     language="en", punctuation=False)
    with torch.inference_mode():
        model.generate(**{k: v.to(device) for k, v in warm.items()},
                       decoder_input_ids=prompt, max_new_tokens=4)

    out_rows = []
    for row in rows:
        audio, sr = sf.read(row["audio_filepath"], dtype="float32")
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"expected {SAMPLE_RATE} Hz, got {sr}")
        inputs = processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           language="en", punctuation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        began = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                decoder_input_ids=prompt,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = 1000.0 * (time.perf_counter() - began)

        sequences = gen.sequences
        prompt_len = sequences.shape[1] - len(gen.scores)
        text = processor.tokenizer.decode(sequences[0], skip_special_tokens=True).strip()
        confidence = from_logprobs(token_logprobs(gen.scores, sequences, prompt_len))

        print(f"  {row['utterance_id']:24} ref={row['text']!r:28} hyp={text!r:28} "
              f"conf={confidence.sequence:.3f} weakest={confidence.weakest_token:.3f} "
              f"-> {confidence.action()}  ({latency_ms:.0f} ms)")

        out_rows.append({
            "audio_filepath": row["audio_filepath"],
            "speaker_id": row["speaker_id"],
            "text": row["text"],
            "prediction": text,
            "model_id": model_id,
            "latency_ms": latency_ms,
            "duration": row["duration"],
            "utterance_id": row["utterance_id"],
            "confidence": confidence.sequence,
            "weakest_token": confidence.weakest_token,
            "n_tokens": confidence.n_tokens,
            "action": confidence.action(),
        })

    if args.out:
        predictions.write(args.out, out_rows)
        print(f"\nwrote {len(out_rows)} rows to {args.out}")
    else:
        print("\n" + json.dumps({"n": len(out_rows),
              "mean_confidence": sum(r["confidence"] for r in out_rows)/len(out_rows)}, indent=2))


if __name__ == "__main__":
    main()
