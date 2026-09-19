#!/usr/bin/env bash
# Entry point for the Baseten training job.
set -euo pipefail

pip install -q uv
uv sync --frozen

# Rebuild the manifests on the box rather than shipping WAVs: prepare_torgo.py is
# deterministic, so the hashes must match what Phase 0 committed. If they do not,
# the training data is not the data we evaluated against and the run is void.
uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir ./data/torgo-dataset
uv run python -m data.prepare_torgo --source ./data/torgo-dataset
uv run python -c "
from contract import manifest
manifest.check_hashes('contract/MANIFEST_HASHES')
print('manifest hashes match Phase 0')
"

uv run python -m train.cohere.train_lora \
  --max-steps "${MAX_STEPS:-400}" \
  --probe-steps "${PROBE_STEPS:-0}" \
  --out "${BT_CHECKPOINT_DIR:-checkpoints/cohere}"
