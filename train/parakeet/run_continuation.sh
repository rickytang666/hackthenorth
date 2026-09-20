#!/usr/bin/env bash
set -euo pipefail

# Person A only: warm-continue the learned encoder adapter. The optimizer and
# scheduler start fresh because the prior sparse Lightning checkpoint is not
# restorable and Baseten did not register a resumable checkpoint for that job.
OUTPUT_ROOT="${BT_CHECKPOINT_DIR:?Baseten must set BT_CHECKPOINT_DIR}"
DATA_ROOT="${VOICEBRIDGE_DATA:?missing staged VoiceBridge data}"
WORKSPACE_ROOT="$(pwd)"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:?missing adapter checkpoint}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-1000}"
START_TOTAL_STEPS="${START_TOTAL_STEPS:-2000}"
VAL_INTERVAL="${VAL_INTERVAL:-400}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
CONTINUATION_LR="${CONTINUATION_LR:-3e-7}"
CONTINUATION_MIN_LR="${CONTINUATION_MIN_LR:-1e-7}"
TRAIN_BATCH_DURATION="${TRAIN_BATCH_DURATION:-60.0}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"

if ! [[ "${ADDITIONAL_STEPS}" =~ ^[0-9]+$ && "${START_TOTAL_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "step counts must be non-negative integers" >&2
  exit 2
fi
TOTAL_STEPS=$((START_TOTAL_STEPS + ADDITIONAL_STEPS))
# Training runs from the data directory so manifests can keep relative audio
# paths. Resolve the separately mounted adapter before changing directories.
export ADAPTER_CHECKPOINT="$(python -c 'import os; print(os.path.abspath(os.environ["ADAPTER_CHECKPOINT"]))')"

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from contract.manifest import load
from train.parakeet import BASELINE_MANIFEST, TRAINING_MANIFEST, VALIDATION_MANIFEST

data_root = Path(os.environ["VOICEBRIDGE_DATA"])
expected = {}
for line in Path("contract/MANIFEST_HASHES").read_text().splitlines():
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
adapter = Path(os.environ["ADAPTER_CHECKPOINT"])
if not adapter.is_file():
    raise RuntimeError(f"adapter checkpoint is missing: {adapter}")
print(json.dumps({purpose: len(items) for purpose, items in rows.items()}, sort_keys=True))
print(f"adapter bytes={adapter.stat().st_size} sha256={hashlib.sha256(adapter.read_bytes()).hexdigest()}")
PY

mkdir -p "${OUTPUT_ROOT}/results"

pushd "${DATA_ROOT}" >/dev/null
python "${WORKSPACE_ROOT}/train/parakeet/continue_asr_adapter.py" \
  --config-path=/opt/NeMo/examples/asr/conf/asr_adapters \
  --config-name=asr_adaptation \
  model.pretrained_model=null \
  "model.nemo_model=${PARAKEET_MODEL_NEMO:?missing mounted Parakeet .nemo}" \
  model.adapter.adapter_name=torgo_dysarthria \
  model.adapter.adapter_module_name=encoder \
  "+model.adapter.restore_state_path=${ADAPTER_CHECKPOINT}" \
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
  "model.optim.lr=${CONTINUATION_LR}" \
  model.optim.weight_decay=0.0 \
  model.optim.sched.warmup_steps=20 \
  model.optim.sched.warmup_ratio=null \
  "model.optim.sched.min_lr=${CONTINUATION_MIN_LR}" \
  "trainer.accumulate_grad_batches=${GRAD_ACCUMULATION}" \
  "trainer.max_steps=${ADDITIONAL_STEPS}" \
  "trainer.val_check_interval=${VAL_INTERVAL}" \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.accelerator=gpu \
  trainer.strategy=auto \
  trainer.precision=bf16-mixed \
  trainer.log_every_n_steps=10 \
  trainer.num_sanity_val_steps=0 \
  exp_manager.exp_dir="${OUTPUT_ROOT}" \
  exp_manager.name="parakeet_torgo_adapter_total_${TOTAL_STEPS}" \
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

checkpoints = sorted(Path(sys.argv[1]).rglob("*.nemo"), key=lambda path: path.stat().st_mtime)
print(checkpoints[-1] if checkpoints else "")
PY
)"
if [[ -z "${BEST_NEMO}" || ! -f "${BEST_NEMO}" ]]; then
  echo "No best-validation .nemo checkpoint was produced" >&2
  exit 1
fi
printf '%s\n' "${BEST_NEMO}" > "${OUTPUT_ROOT}/results/selected_checkpoint.txt"

# Export the best validation state, not merely the terminal state, as the seed
# for the next continuation stage.
python - "${BEST_NEMO}" "${OUTPUT_ROOT}/results/torgo_dysarthria_adapter_total_${TOTAL_STEPS}.pt" <<'PY'
import sys
from nemo.collections.asr.models import ASRModel

model = ASRModel.restore_from(sys.argv[1], map_location="cpu")
model.save_adapters(sys.argv[2])
PY

python -m train.parakeet.decode \
  --checkpoint "${BEST_NEMO}" \
  --model-id "nvidia/parakeet-tdt-0.6b-v2-torgo-adapter-${TOTAL_STEPS}" \
  --manifest "${BASELINE_MANIFEST}" \
  --output "${OUTPUT_ROOT}/results/adapter_total_${TOTAL_STEPS}_baseline_10.jsonl" \
  --batch-size 10 \
  --manifest-hash-scope train-dev

python -m contract.evaluate \
  "${OUTPUT_ROOT}/results/adapter_total_${TOTAL_STEPS}_baseline_10.jsonl" \
  --json | tee "${OUTPUT_ROOT}/results/adapter_total_${TOTAL_STEPS}_scorecard.txt"

python - "${OUTPUT_ROOT}/results/continuation_receipt.json" <<'PY'
import json
import os
import sys
from pathlib import Path

receipt = {
    "continuation_kind": "adapter_warm_start",
    "optimizer_state_restored": False,
    "scheduler_state_restored": False,
    "adapter_state_restored": True,
    "start_total_steps": int(os.environ["START_TOTAL_STEPS"]),
    "additional_steps_ceiling": int(os.environ["ADDITIONAL_STEPS"]),
    "total_steps_ceiling": int(os.environ["START_TOTAL_STEPS"]) + int(os.environ["ADDITIONAL_STEPS"]),
    "learning_rate": float(os.environ["CONTINUATION_LR"]),
}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
