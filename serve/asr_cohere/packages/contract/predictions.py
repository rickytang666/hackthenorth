"""Prediction row schema and writer. Every decode from either lane writes this."""

import json
from pathlib import Path

FIELDS = ("audio_filepath", "speaker_id", "text", "prediction", "model_id", "latency_ms")

FILENAMES = (
    "baseline_parakeet.jsonl",
    "tuned_parakeet.jsonl",
    "comparison_parakeet_1p1b.jsonl",
    "baseline_cohere.jsonl",
    "tuned_cohere.jsonl",
)


def validate_row(row: dict) -> None:
    missing = [f for f in FIELDS if f not in row]
    if missing:
        raise ValueError(f"prediction row missing {missing}")
    if not isinstance(row["latency_ms"], (int, float)):
        raise ValueError("latency_ms must be numeric, steady state, excluding cold start")


def write(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            validate_row(row)
            f.write(json.dumps(row) + "\n")


def read(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).open()]
    for row in rows:
        validate_row(row)
    return rows
