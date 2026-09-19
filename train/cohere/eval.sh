#!/usr/bin/env bash
# Decode frozen and tuned Cohere on the same manifest, on the same hardware.
# Both numbers must come from one box or the scorecard compares machines.
set -euo pipefail

pip install -q uv
uv sync --frozen

mkdir -p "$VOICEBRIDGE_DATA"
uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m data.prepare_torgo --source "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -c "
from contract import manifest
manifest.check_hashes('contract/MANIFEST_HASHES')
print('manifest hashes match Phase 0')
"

OUT="${BT_CHECKPOINT_DIR:-results}"
mkdir -p "$OUT"

echo "=== frozen baseline ==="
uv run python -m train.cohere.decode_cohere \
  --manifest "${EVAL_MANIFEST:-torgo_dys_dev.jsonl}" \
  --out "$OUT/baseline_cohere.jsonl"

echo "=== TORGO-tuned adapter ==="
uv run python -m train.cohere.decode_cohere \
  --manifest "${EVAL_MANIFEST:-torgo_dys_dev.jsonl}" \
  --adapter "$ADAPTER_PATH" \
  --out "$OUT/tuned_cohere.jsonl"

echo "=== scorecard ==="
uv run python -m contract.evaluate "$OUT/baseline_cohere.jsonl" "$OUT/tuned_cohere.jsonl"
