"""Stage only the frozen dysarthric dev split for the Baseten baseline job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from contract import manifest
from train.parakeet import BASELINE_MANIFEST, TRAINING_MANIFEST, VALIDATION_MANIFEST
from train.parakeet.splits import phase1_split, write_phase1_manifests

DEFAULT_OUTPUT = Path("/private/tmp/voicebridge-parakeet-baseline/data")


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


def stage(output: Path) -> dict:
    manifest.check_hashes("contract/MANIFEST_HASHES")
    source_root = manifest.data_root().resolve()
    _safe_replace_directory(output)
    staged_root = output.resolve() / "voicebridge"
    staged_manifests = staged_root / "manifests"
    staged_manifests.mkdir(parents=True)

    # All five small manifests travel with the bundle so the standard frozen
    # hash check remains unchanged. Only dev audio is staged.
    for name in manifest.MANIFESTS:
        shutil.copy2(manifest.manifest_dir() / name, staged_manifests / name)
    write_phase1_manifests(staged_manifests)

    splits = phase1_split()
    rows = splits.baseline

    audio_bytes = 0
    for row in rows:
        relative = Path(row["audio_filepath"])
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = staged_root / relative
        _link_or_copy(source, destination)
        audio_bytes += source.stat().st_size

    receipt = {
        "baseline_manifest": BASELINE_MANIFEST,
        "baseline_rows": len(rows),
        "baseline_speakers": sorted({row["speaker_id"] for row in rows}),
        "training_manifest": TRAINING_MANIFEST,
        "training_rows": len(splits.training),
        "validation_manifest": VALIDATION_MANIFEST,
        "validation_rows": len(splits.validation),
        "duration_hours": sum(float(row["duration"]) for row in rows) / 3600.0,
        "audio_bytes": audio_bytes,
        "manifest_hashes": manifest.hash_all(),
    }
    (staged_root / "BASELINE_BUNDLE.json").write_text(
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
