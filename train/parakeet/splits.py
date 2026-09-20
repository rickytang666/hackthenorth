"""Person A's deterministic Phase 1 split derived from the frozen dev set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from contract import manifest
from contract.normalize import tokens
from train.parakeet import BASELINE_MANIFEST, TRAINING_MANIFEST, VALIDATION_MANIFEST

# F04 validation is intentionally mild. The tiny pre-training smoke baseline is
# instead withheld from the severe training speakers so obvious data-routing
# mistakes cannot hide behind near-typical speech. M02 remains sealed test.
SEVERE_BASELINE_QUOTAS = {
    "F01": {"word": 2, "sentence": 2},
    "M01": {"word": 2, "sentence": 2},
    "M04": {"word": 1, "sentence": 1},
}


@dataclass(frozen=True)
class Phase1Splits:
    training: list[dict]
    baseline: list[dict]
    validation: list[dict]


def _stable_rank(row: dict) -> str:
    return hashlib.sha256(row["utterance_id"].encode()).hexdigest()


def phase1_split() -> Phase1Splits:
    """Derive training, 10-row severe baseline, and F04 validation sets."""
    original_training = manifest.load("torgo_dys_train.jsonl", resolve=False)
    validation = manifest.load("torgo_dys_dev.jsonl", resolve=False)

    baseline = []
    for speaker, quota in SEVERE_BASELINE_QUOTAS.items():
        speaker_rows = [
            row
            for row in original_training
            if row["speaker_id"] == speaker and row["mic"].lower() == "headmic"
        ]
        words = sorted(
            (row for row in speaker_rows if len(tokens(row["text"])) == 1),
            key=_stable_rank,
        )
        sentences = sorted(
            (row for row in speaker_rows if len(tokens(row["text"])) > 1),
            key=_stable_rank,
        )
        if len(words) < quota["word"] or len(sentences) < quota["sentence"]:
            raise RuntimeError(f"speaker {speaker} lacks rows for baseline quota")
        baseline.extend(words[: quota["word"]])
        baseline.extend(sentences[: quota["sentence"]])

    baseline.sort(key=lambda row: row["utterance_id"])
    baseline_groups = {row["utterance_group"] for row in baseline}
    # Remove both microphone views of each withheld utterance from training.
    training = [
        row
        for row in original_training
        if row["utterance_group"] not in baseline_groups
    ]

    if len(baseline) != 10 or len(validation) != 244:
        raise RuntimeError("invalid Phase 1 split sizes")
    if baseline_groups & {row["utterance_group"] for row in training}:
        raise RuntimeError("baseline/training utterance-group overlap")
    if {row["speaker_id"] for row in validation} != {"F04"}:
        raise RuntimeError("validation speaker drifted from F04")
    return Phase1Splits(training=training, baseline=baseline, validation=validation)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def write_phase1_manifests(directory: Path) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    splits = phase1_split()
    training_path = directory / TRAINING_MANIFEST
    baseline_path = directory / BASELINE_MANIFEST
    validation_path = directory / VALIDATION_MANIFEST
    write_jsonl(training_path, splits.training)
    write_jsonl(baseline_path, splits.baseline)
    write_jsonl(validation_path, splits.validation)
    return training_path, baseline_path, validation_path
