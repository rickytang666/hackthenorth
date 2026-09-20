#!/usr/bin/env bash
# Capacity sweep: LoRA rank and encoder depth, scored on the validation speaker.
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
uv run python -m train.cohere.sweep --steps "${SWEEP_STEPS:-900}" --out "${BT_CHECKPOINT_DIR:-results}/sweep"
