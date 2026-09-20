#!/usr/bin/env bash
set -euo pipefail

# This is an evaluation-only Baseten job. Do not add a training command here.
python -m pip install \
  "jiwer>=3,<5"

OUTPUT_ROOT="${BT_CHECKPOINT_DIR:?Baseten must set BT_CHECKPOINT_DIR}"

python - <<'PY'
from pathlib import Path

from contract.manifest import check_hashes, load
from train.parakeet import BASELINE_MANIFEST

check_hashes("contract/MANIFEST_HASHES")
rows = load(BASELINE_MANIFEST, resolve=True)
missing = [
    row["audio_filepath"]
    for row in rows
    if not Path(row["audio_filepath"]).is_file()
]
if missing:
    raise RuntimeError(f"{len(missing)} manifest audio paths are missing")
print("manifest hashes: MATCH")
print(f"baseline rows: {len(rows)}")
print("baseline audio paths: PRESENT")
PY

mkdir -p "${OUTPUT_ROOT}/results"
python -m train.parakeet.decode \
  --checkpoint "${PARAKEET_MODEL_PATH:?missing mounted Parakeet checkpoint}" \
  --model-id nvidia/parakeet-tdt-0.6b-v2 \
  --manifest torgo_dys_baseline_10.jsonl \
  --output "${OUTPUT_ROOT}/results/baseline_parakeet.jsonl" \
  --batch-size "${BASELINE_BATCH_SIZE:-16}"

python -m contract.evaluate \
  "${OUTPUT_ROOT}/results/baseline_parakeet.jsonl" \
  --json | tee "${OUTPUT_ROOT}/results/baseline_parakeet_scorecard.txt"
