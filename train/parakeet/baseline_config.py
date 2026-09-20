"""Baseten H100 job for the required frozen Parakeet baseline only.

Run from the repository root:
    truss train push train/parakeet/baseline_config.py

This job never calls a training entry point. It writes predictions and the
scorecard beneath BT_CHECKPOINT_DIR so they can be downloaded after completion.
"""

import os

from truss.base import truss_config
from truss_train import WeightsSource, definitions

BASE_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
PROJECT_NAME = os.environ.get(
    "BASETEN_TRAINING_PROJECT_NAME", "voicebridge-parakeet-frozen-baseline"
)
BASE_IMAGE = os.environ.get("PINNED_NEMO_IMAGE")
if not BASE_IMAGE:
    raise RuntimeError(
        "PINNED_NEMO_IMAGE must name the NeMo image already smoke-tested in Phase 0"
    )

MODEL_MOUNT = f"/app/models/{BASE_MODEL_ID}"
BASELINE_DATA = os.environ.get(
    "VOICEBRIDGE_BASELINE_DATA",
    "/private/tmp/voicebridge-parakeet-baseline/data",
)

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./train/parakeet/run_baseline.sh"],
    environment_variables={
        # Baseten starts commands from the gathered workspace root. External
        # directories retain their basename, so the staged bundle is at
        # ./data/voicebridge inside the job.
        "VOICEBRIDGE_DATA": "data/voicebridge",
        "PARAKEET_MODEL_PATH": MODEL_MOUNT,
        "PYTHONPATH": ".",
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
    # Upload code plus a separately staged 10-file severe baseline bundle. The
    # complete local corpus, healthy controls, and sealed test audio stay local.
    workspace=definitions.Workspace(
        workspace_root="../..",
        external_dirs=[BASELINE_DATA],
        exclude_dirs=[
            "../../.git",
            "../../.venv",
            "../../.workspace",
            "../../data",
            "../../results",
        ],
    ),
)

training_project = definitions.TrainingProject(
    name=PROJECT_NAME,
    job=job,
)
