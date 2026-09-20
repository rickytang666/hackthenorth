#!/usr/bin/env bash
# Leave-one-speaker-out across all eight dysarthric TORGO speakers.
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

uv run python -m train.cohere.loso \
  --steps "${LOSO_STEPS:-900}" \
  --out "${BT_CHECKPOINT_DIR:-results}/loso"
