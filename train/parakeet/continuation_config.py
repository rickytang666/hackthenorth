"""Baseten H100 warm continuation for Person A's trained Parakeet adapter."""

import os

from truss.base import truss_config
from truss_train import WeightsSource, definitions

BASE_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
BASE_IMAGE = os.environ.get("PINNED_NEMO_IMAGE")
if not BASE_IMAGE:
    raise RuntimeError("PINNED_NEMO_IMAGE must name the baseline-tested NeMo image")

PROJECT_NAME = os.environ.get(
    "BASETEN_TRAINING_PROJECT_NAME", "voicebridge-parakeet-adapter-continuation-t34"
)
TRAINING_DATA = os.environ.get(
    "VOICEBRIDGE_TRAINING_DATA", "/private/tmp/voicebridge-parakeet-training/data"
)
ADAPTER_SEED_DIR = os.environ.get(
    "VOICEBRIDGE_ADAPTER_SEED_DIR",
    "/private/tmp/voicebridge-parakeet-continuation/adapter_seed",
)
MODEL_MOUNT = f"/app/models/{BASE_MODEL_ID}"

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./train/parakeet/run_continuation.sh"],
    environment_variables={
        "ADAPTER_CHECKPOINT": "adapter_seed/torgo_dysarthria_adapter.pt",
        "ADDITIONAL_STEPS": os.environ.get("ADDITIONAL_STEPS", "1000"),
        "BASELINE_MANIFEST": "torgo_dys_baseline_10.jsonl",
        "CONTINUATION_LR": os.environ.get("CONTINUATION_LR", "3e-7"),
        "CONTINUATION_MIN_LR": os.environ.get("CONTINUATION_MIN_LR", "1e-7"),
        "EARLY_STOPPING_PATIENCE": os.environ.get("EARLY_STOPPING_PATIENCE", "5"),
        "GRAD_ACCUMULATION": os.environ.get("GRAD_ACCUMULATION", "4"),
        "PARAKEET_MODEL_NEMO": f"{MODEL_MOUNT}/parakeet-tdt-0.6b-v2.nemo",
        "PYTHONPATH": ".",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "START_TOTAL_STEPS": os.environ.get("START_TOTAL_STEPS", "2000"),
        "TRAIN_BATCH_DURATION": os.environ.get("TRAIN_BATCH_DURATION", "60.0"),
        "TRAIN_MANIFEST": "torgo_dys_train_phase1.jsonl",
        "VAL_INTERVAL": os.environ.get("VAL_INTERVAL", "400"),
        "VALIDATION_MANIFEST": "torgo_dys_validation.jsonl",
        "VOICEBRIDGE_DATA": "data/voicebridge",
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=definitions.Compute(
        node_count=1,
        cpu_count=16,
        memory="96Gi",
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H100,
            count=1,
        ),
    ),
    runtime=runtime,
    weights=[WeightsSource(source=f"hf://{BASE_MODEL_ID}", mount_location=MODEL_MOUNT)],
    workspace=definitions.Workspace(
        workspace_root="../..",
        external_dirs=[TRAINING_DATA, ADAPTER_SEED_DIR],
        exclude_dirs=[
            "../../.git",
            "../../.venv",
            "../../.workspace",
            "../../data",
            "../../results",
        ],
    ),
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=job)
