#!/usr/bin/env bash
# Rank the eight dysarthric speakers by difficulty using the FROZEN baseline.
# Fair ranking: the base model has seen none of them. Not an evaluation of the
# tuned model, which trained on six of these eight.
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
uv run python -m train.cohere.decode_cohere \
  --manifest torgo_dys_allspeakers.jsonl --out "$OUT/allspeakers_baseline.jsonl"
uv run python -m train.cohere.decode_cohere \
  --manifest torgo_dys_allspeakers.jsonl --adapter "$ADAPTER_PATH" \
  --out "$OUT/allspeakers_tuned.jsonl"
