"""Build the five frozen manifests from the packaged TORGO release. Runs once.

Deterministic by construction: rows are sorted by utterance_id before writing, so
every machine that runs this produces byte-identical manifests and therefore
identical SHA-256 hashes. That equality is the Phase 0 exit gate which lets the
second builder trust data they did not generate.

    export VOICEBRIDGE_DATA=$PWD/data/voicebridge
    python -m data.prepare_torgo --source data/torgo-dataset

Audio is written under $VOICEBRIDGE_DATA/torgo_wav and manifests reference it by
a RELATIVE path, because Phase 0 runs on one machine, training runs on Baseten,
and the second builder is on a third machine.
"""

import argparse
import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SAMPLE_RATE = 16000

# Frozen before any baseline is scored. Eight dysarthric speakers: F01 F03 F04
# M01 M02 M03 M04 M05. Six train, one dev, one test, never rebalanced on results.
DEV_SPEAKERS = {"F04"}
TEST_SPEAKERS = {"M02"}

# Held out of the replay pool so the clean-speech regression gate is speaker-disjoint.
CONTROL_EVAL_SPEAKERS = {"MC04", "FC03"}

SKIP_TRANSCRIPTS = {"", "xxx", "x x x", "[no speech]"}


def parse_ids(wav_name: str) -> dict:
    """FC01_1_arrayMic_0066.wav -> speaker FC01, session 1, mic arrayMic, index 0066."""
    stem = Path(wav_name).stem
    parts = stem.split("_")
    if len(parts) >= 4:
        return {
            "speaker_id": parts[0],
            "mic": parts[2],
            "utterance_id": stem,
            # Paired mic views of one utterance share this. It never crosses a split.
            "utterance_group": f"{parts[0]}_{parts[1]}_{parts[3]}",
        }
    return {"speaker_id": parts[0], "mic": "unknown", "utterance_id": stem, "utterance_group": stem}


def split_of(speaker_id: str) -> str:
    if speaker_id in DEV_SPEAKERS:
        return "dev"
    if speaker_id in TEST_SPEAKERS:
        return "test"
    return "train"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/torgo-dataset")
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N rows")
    args = parser.parse_args()

    root = Path(os.environ.get("VOICEBRIDGE_DATA", "")).expanduser()
    if not str(root):
        raise SystemExit("export VOICEBRIDGE_DATA first")
    wav_dir = root / "torgo_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{args.source}/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquet shards under {args.source}")

    dys: list[dict] = []
    healthy: list[dict] = []
    seen = 0

    for shard in files:
        table = pq.read_table(shard)
        for row in table.to_pylist():
            if args.limit and seen >= args.limit:
                break
            seen += 1
            text = (row["transcription"] or "").strip()
            if text.lower() in SKIP_TRANSCRIPTS:
                continue
            ids = parse_ids(row["audio"]["path"])
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                raise RuntimeError(f"unexpected sample rate {sr} in {ids['utterance_id']}")
            if not np.isfinite(audio).all() or audio.size == 0:
                continue
            rel = Path("torgo_wav") / f"{ids['utterance_id']}.wav"
            sf.write(root / rel, audio, SAMPLE_RATE, subtype="PCM_16")
            record = {
                "audio_filepath": str(rel),
                "text": text,
                "duration": float(audio.size) / SAMPLE_RATE,
                "speaker_id": ids["speaker_id"],
                "speech_status": row["speech_status"],
                "mic": ids["mic"],
                "utterance_id": ids["utterance_id"],
                "utterance_group": ids["utterance_group"],
            }
            (dys if row["speech_status"] == "dysarthria" else healthy).append(record)

    key = lambda r: r["utterance_id"]
    dys.sort(key=key)
    healthy.sort(key=key)

    head = lambda rows: [r for r in rows if r["mic"].lower() == "headmic"]
    train = [r for r in dys if split_of(r["speaker_id"]) == "train"]
    dev = [r for r in head(dys) if split_of(r["speaker_id"]) == "dev"]
    test = [r for r in head(dys) if split_of(r["speaker_id"]) == "test"]

    train_spk = {r["speaker_id"] for r in train}
    assert train_spk.isdisjoint({r["speaker_id"] for r in dev + test}), "speaker leak"
    assert {r["speaker_id"] for r in dev}.isdisjoint({r["speaker_id"] for r in test})
    train_groups = {r["utterance_group"] for r in train}
    assert train_groups.isdisjoint({r["utterance_group"] for r in dev + test}), "group leak"

    # Controls are never positive training examples. At most 10% replay, and only
    # if the tuned model fails the clean-speech regression gate.
    controls = head(healthy)
    clean_eval = [r for r in controls if r["speaker_id"] in CONTROL_EVAL_SPEAKERS]
    replay_pool = [r for r in controls if r["speaker_id"] not in CONTROL_EVAL_SPEAKERS]
    assert clean_eval and replay_pool, "control split is empty"
    clean_replay = replay_pool[: max(1, len(train) // 10)]
    assert {r["speaker_id"] for r in clean_replay}.isdisjoint(
        {r["speaker_id"] for r in clean_eval}
    ), "control speaker leak"

    out = root / "manifests"
    for name, rows in {
        "torgo_dys_train.jsonl": train,
        "torgo_dys_dev.jsonl": dev,
        "torgo_dys_test.jsonl": test,
        "torgo_clean_replay.jsonl": clean_replay,
        "torgo_clean_eval.jsonl": clean_eval,
    }.items():
        write_jsonl(out / name, rows)
        hours = sum(r["duration"] for r in rows) / 3600
        speakers = sorted({r["speaker_id"] for r in rows})
        print(f"{name:26} {len(rows):6} rows  {hours:5.2f} h  {speakers}")


if __name__ == "__main__":
    main()
