"""Shared decode loop. Both lanes emit byte-comparable prediction files from it
without sharing any model code.

    from contract.decode import decode_manifest
    decode_manifest("torgo_dys_dev.jsonl", my_transcribe, "nvidia/parakeet-tdt-0.6b-v2",
                    "results/baseline_parakeet.jsonl")
"""

import time
from collections.abc import Callable
from pathlib import Path

from contract import manifest, predictions

Transcribe = Callable[[list[str]], list[str]]


def decode_manifest(
    manifest_name: str,
    transcribe: Transcribe,
    model_id: str,
    out_path: str | Path,
    batch_size: int = 16,
) -> list[dict]:
    """Decode one manifest and write a prediction JSONL.

    `transcribe` takes a list of absolute audio paths and returns one string per
    path, in order. Latency is wall clock per batch divided evenly across it,
    steady state only: warm the model before calling this.
    """
    rows = manifest.load(manifest_name, resolve=True)
    out_rows: list[dict] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        began = time.perf_counter()
        hyps = transcribe([r["audio_filepath"] for r in batch])
        elapsed_ms = 1000.0 * (time.perf_counter() - began)
        if len(hyps) != len(batch):
            raise RuntimeError(f"transcribe returned {len(hyps)} for {len(batch)} inputs")
        per_item = elapsed_ms / len(batch)
        for row, hyp in zip(batch, hyps):
            out_rows.append({
                "audio_filepath": row["audio_filepath"],
                "speaker_id": row["speaker_id"],
                "text": row["text"],
                "prediction": hyp,
                "model_id": model_id,
                "latency_ms": per_item,
                "duration": row["duration"],
                "utterance_id": row["utterance_id"],
            })

    predictions.write(out_path, out_rows)
    return out_rows
