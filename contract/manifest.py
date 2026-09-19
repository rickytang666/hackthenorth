"""Manifest row schema, validation, and path resolution.

Paths in a manifest are RELATIVE to $VOICEBRIDGE_DATA. Phase 0 runs on one
machine, training runs on Baseten, and the second builder is on a third machine,
so an absolute path is wrong on two of the three and the shared hash cannot
match. Resolution happens here, at load time, and nowhere else.
"""

import hashlib
import json
import os
from pathlib import Path

FIELDS = (
    "audio_filepath",
    "text",
    "duration",
    "speaker_id",
    "speech_status",
    "mic",
    "utterance_id",
    "utterance_group",
)

MANIFESTS = (
    "torgo_dys_train.jsonl",
    "torgo_dys_dev.jsonl",
    "torgo_dys_test.jsonl",
    "torgo_clean_replay.jsonl",
    "torgo_clean_eval.jsonl",
)

DEV_SPEAKERS = {"F04"}
TEST_SPEAKERS = {"M02"}


def data_root() -> Path:
    root = os.environ.get("VOICEBRIDGE_DATA")
    if not root:
        raise RuntimeError("VOICEBRIDGE_DATA is not set. Export it to the data root.")
    return Path(root)


def manifest_dir() -> Path:
    return data_root() / "manifests"


def validate_row(row: dict) -> None:
    missing = [f for f in FIELDS if f not in row]
    if missing:
        raise ValueError(f"row missing fields {missing}: {row.get('utterance_id')}")
    if Path(row["audio_filepath"]).is_absolute():
        raise ValueError(
            f"absolute path in manifest: {row['audio_filepath']}. "
            "Paths must be relative to $VOICEBRIDGE_DATA."
        )
    if row["speech_status"] not in ("dysarthria", "healthy"):
        raise ValueError(f"bad speech_status: {row['speech_status']}")
    if not isinstance(row["duration"], (int, float)) or row["duration"] <= 0:
        raise ValueError(f"bad duration for {row['utterance_id']}")


def load(name: str, resolve: bool = True) -> list[dict]:
    """Load one manifest. With resolve, audio_filepath becomes an absolute path."""
    path = manifest_dir() / name
    rows = []
    root = data_root()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            validate_row(row)
            if resolve:
                row = {**row, "audio_filepath": str(root / row["audio_filepath"])}
            rows.append(row)
    return rows


def load_all(resolve: bool = True) -> dict[str, list[dict]]:
    return {name: load(name, resolve) for name in MANIFESTS}


def hash_manifest(name: str) -> str:
    """SHA-256 of the raw manifest bytes. Identical on every machine."""
    return hashlib.sha256((manifest_dir() / name).read_bytes()).hexdigest()


def hash_all() -> dict[str, str]:
    return {name: hash_manifest(name) for name in MANIFESTS}


def check_hashes(expected_path: str | Path) -> None:
    """Assert local manifests match contract/MANIFEST_HASHES. Raises on drift."""
    expected = {}
    for line in Path(expected_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()
        expected[name] = digest
    actual = hash_all()
    bad = {n: (expected.get(n), actual[n]) for n in MANIFESTS if expected.get(n) != actual[n]}
    if bad:
        raise RuntimeError(f"manifest hash mismatch, do not train: {bad}")


if __name__ == "__main__":
    for name, digest in hash_all().items():
        print(f"{digest}  {name}")
