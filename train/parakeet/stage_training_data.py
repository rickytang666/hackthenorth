"""Stage only Person A's dysarthric Phase 1 training/evaluation audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from contract import manifest
from train.parakeet import BASELINE_MANIFEST, TRAINING_MANIFEST, VALIDATION_MANIFEST
from train.parakeet.splits import phase1_split, write_phase1_manifests

DEFAULT_OUTPUT = Path("/private/tmp/voicebridge-parakeet-training/data")


def _safe_replace_directory(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path("/") or len(resolved.parts) < 4:
        raise ValueError(f"refusing to replace unsafe staging path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(output: Path = DEFAULT_OUTPUT) -> dict:
    """Build a bundle with dysarthric train, validation, and baseline data only."""
    manifest.check_hashes("contract/MANIFEST_HASHES")
    source_root = manifest.data_root().resolve()
    _safe_replace_directory(output)
    staged_root = output.resolve() / "voicebridge"
    staged_manifests = staged_root / "manifests"
    staged_manifests.mkdir(parents=True)

    # Preserve only the frozen manifests needed to prove the train/dev inputs.
    # Clean controls and the sealed M02 test manifest are not mounted.
    for name in ("torgo_dys_train.jsonl", "torgo_dys_dev.jsonl"):
        shutil.copy2(manifest.manifest_dir() / name, staged_manifests / name)
    training_path, baseline_path, validation_path = write_phase1_manifests(
        staged_manifests
    )

    splits = phase1_split()
    rows_by_purpose = {
        "training": splits.training,
        "validation": splits.validation,
        "baseline": splits.baseline,
    }
    unique_audio: dict[Path, Path] = {}
    for rows in rows_by_purpose.values():
        for row in rows:
            relative = Path(row["audio_filepath"])
            unique_audio[relative] = source_root / relative

    audio_bytes = 0
    for relative, source in unique_audio.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        _link_or_copy(source, staged_root / relative)
        audio_bytes += source.stat().st_size

    receipt = {
        "audio_bytes": audio_bytes,
        "audio_files": len(unique_audio),
        "baseline_manifest": BASELINE_MANIFEST,
        "baseline_rows": len(splits.baseline),
        "derived_manifest_hashes": {
            TRAINING_MANIFEST: _sha256(training_path),
            BASELINE_MANIFEST: _sha256(baseline_path),
            VALIDATION_MANIFEST: _sha256(validation_path),
        },
        "frozen_manifest_hashes": {
            name: manifest.hash_manifest(name)
            for name in ("torgo_dys_train.jsonl", "torgo_dys_dev.jsonl")
        },
        "max_training_duration_seconds": max(
            float(row["duration"]) for row in splits.training
        ),
        "over_30_second_training_rows": sum(
            float(row["duration"]) > 30 for row in splits.training
        ),
        "training_manifest": TRAINING_MANIFEST,
        "training_rows": len(splits.training),
        "validation_manifest": VALIDATION_MANIFEST,
        "validation_rows": len(splits.validation),
    }
    (staged_root / "TRAINING_BUNDLE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(stage(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
