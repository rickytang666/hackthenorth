#!/usr/bin/env bash
set -euo pipefail

# Person A only: NeMo Parakeet adapter training and post-train evaluation.
# Healthy controls and the sealed test speaker are intentionally absent.
OUTPUT_ROOT="${BT_CHECKPOINT_DIR:?Baseten must set BT_CHECKPOINT_DIR}"
DATA_ROOT="${VOICEBRIDGE_DATA:?missing staged VoiceBridge data}"
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_INTERVAL="${VAL_INTERVAL:-400}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
ADAPTER_DIM="${ADAPTER_DIM:-32}"
ADAPTER_LR="${ADAPTER_LR:-3e-6}"
TRAIN_BATCH_DURATION="${TRAIN_BATCH_DURATION:-60.0}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"

python - <<'PY'
import hashlib
import json
from pathlib import Path

from contract.manifest import load
from train.parakeet import BASELINE_MANIFEST, TRAINING_MANIFEST, VALIDATION_MANIFEST

data_root = Path("data/voicebridge")
check_hashes_path = Path("contract/MANIFEST_HASHES")
expected = {}
for line in check_hashes_path.read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        digest, name = line.split()
        expected[name] = digest
for name in ("torgo_dys_train.jsonl", "torgo_dys_dev.jsonl"):
    actual = hashlib.sha256((data_root / "manifests" / name).read_bytes()).hexdigest()
    if actual != expected[name]:
        raise RuntimeError(f"frozen manifest hash mismatch: {name}")

rows = {
    "training": load(TRAINING_MANIFEST, resolve=True),
    "validation": load(VALIDATION_MANIFEST, resolve=True),
    "baseline": load(BASELINE_MANIFEST, resolve=True),
}
for purpose, purpose_rows in rows.items():
    missing = [row["audio_filepath"] for row in purpose_rows if not Path(row["audio_filepath"]).is_file()]
    if missing:
        raise RuntimeError(f"{purpose}: {len(missing)} audio paths are missing")
print(json.dumps({purpose: len(purpose_rows) for purpose, purpose_rows in rows.items()}, sort_keys=True))
print("frozen train/dev hashes: MATCH")
print("training, validation, baseline audio paths: PRESENT")
PY

mkdir -p "${OUTPUT_ROOT}/results"

# Relative audio paths in the immutable manifests resolve from this directory.
pushd "${DATA_ROOT}" >/dev/null
python /opt/NeMo/examples/asr/asr_adapters/train_asr_adapter.py \
  --config-path=/opt/NeMo/examples/asr/conf/asr_adapters \
  --config-name=asr_adaptation \
  model.pretrained_model=null \
  "model.nemo_model=${PARAKEET_MODEL_NEMO:?missing mounted Parakeet .nemo}" \
  model.adapter.adapter_name=torgo_dysarthria \
  model.adapter.adapter_type=linear \
  model.adapter.adapter_module_name=encoder \
  model.adapter.adapter_state_dict_name=torgo_dysarthria_adapter.pt \
  model.adapter.linear.in_features=1024 \
  "model.adapter.linear.dim=${ADAPTER_DIM}" \
  model.adapter.linear.dropout=0.1 \
  "model.train_ds.manifest_filepath=manifests/${TRAIN_MANIFEST}" \
  "model.validation_ds.manifest_filepath=manifests/${VALIDATION_MANIFEST}" \
  "+model.train_ds.batch_duration=${TRAIN_BATCH_DURATION}" \
  +model.train_ds.use_bucketing=false \
  +model.train_ds.text_field=text \
  +model.validation_ds.text_field=text \
  model.validation_ds.batch_size=16 \
  model.train_ds.num_workers=8 \
  model.validation_ds.num_workers=8 \
  +model.train_ds.max_duration=30.0 \
  model.optim.name=adamw \
  "model.optim.lr=${ADAPTER_LR}" \
  model.optim.weight_decay=0.0 \
  model.optim.sched.warmup_steps=50 \
  model.optim.sched.warmup_ratio=null \
  model.optim.sched.min_lr=3e-7 \
  "trainer.accumulate_grad_batches=${GRAD_ACCUMULATION}" \
  "trainer.max_steps=${MAX_STEPS}" \
  "trainer.val_check_interval=${VAL_INTERVAL}" \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.accelerator=gpu \
  trainer.strategy=auto \
  trainer.precision=bf16-mixed \
  trainer.log_every_n_steps=10 \
  trainer.num_sanity_val_steps=0 \
  exp_manager.exp_dir="${OUTPUT_ROOT}" \
  exp_manager.name=parakeet_torgo_adapter \
  exp_manager.create_checkpoint_callback=true \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=1 \
  exp_manager.checkpoint_callback_params.always_save_nemo=true \
  ++exp_manager.checkpoint_callback_params.save_last=false \
  ++exp_manager.checkpoint_callback_params.save_best_model=false \
  ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=false \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.mode=min \
  ++exp_manager.early_stopping_callback_params.min_delta=0.001 \
  "++exp_manager.early_stopping_callback_params.patience=${EARLY_STOPPING_PATIENCE}" \
  ++exp_manager.early_stopping_callback_params.strict=true \
  ++exp_manager.early_stopping_callback_params.check_on_train_epoch_end=false
popd >/dev/null

BEST_NEMO="$(python - "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

checkpoints = sorted(
    Path(sys.argv[1]).rglob("*.nemo"), key=lambda path: path.stat().st_mtime
)
print(checkpoints[-1] if checkpoints else "")
PY
)"
if [[ -z "${BEST_NEMO}" || ! -f "${BEST_NEMO}" ]]; then
  echo "No .nemo checkpoint was produced" >&2
  exit 1
fi
printf '%s\n' "${BEST_NEMO}" > "${OUTPUT_ROOT}/results/selected_checkpoint.txt"

python -m train.parakeet.decode \
  --checkpoint "${BEST_NEMO}" \
  --model-id nvidia/parakeet-tdt-0.6b-v2-torgo-adapter \
  --manifest "${BASELINE_MANIFEST}" \
  --output "${OUTPUT_ROOT}/results/adapter_baseline_10.jsonl" \
  --batch-size 10 \
  --manifest-hash-scope train-dev

python -m contract.evaluate \
  "${OUTPUT_ROOT}/results/adapter_baseline_10.jsonl" \
  --json | tee "${OUTPUT_ROOT}/results/adapter_scorecard.txt"
