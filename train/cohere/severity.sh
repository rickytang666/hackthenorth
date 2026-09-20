#!/usr/bin/env bash
# Complete per-speaker eval: every dysarthric speaker, headMic, base vs tuned.
# Runs on the H100 because holding two 2B models locally exhausts a 24 GB Mac.
# No training here. Inference only.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq git
pip install -q uv
uv sync --frozen

mkdir -p "$VOICEBRIDGE_DATA"
uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m data.prepare_torgo --source "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -c "
from contract import manifest; manifest.check_hashes('contract/MANIFEST_HASHES'); print('hashes match')
"
uv run python -m data.make_allspeaker_manifest

OUT="${BT_CHECKPOINT_DIR:-results}"; mkdir -p "$OUT"
echo "=== frozen baseline, all speakers ==="
uv run python -m train.cohere.decode_cohere \
  --manifest torgo_dys_allspeakers.jsonl --out "$OUT/allspeakers_baseline.jsonl"
echo "=== tuned adapter, all speakers ==="
uv run python -m train.cohere.decode_cohere \
  --manifest torgo_dys_allspeakers.jsonl --adapter "$ADAPTER_PATH" \
  --out "$OUT/allspeakers_tuned.jsonl"

uv run python -m contract.per_speaker \
  "$OUT/allspeakers_baseline.jsonl" "$OUT/allspeakers_tuned.jsonl"
