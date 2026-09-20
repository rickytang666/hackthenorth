"""Baseten job that decodes frozen and tuned Cohere on one box.

The experimental contract requires the same hardware and the same evaluator for
every number in the promotion table, so the baseline and the adapter are decoded
in a single job rather than wherever is convenient.

    ADAPTER_JOB_ID=qkeeyeq uv run truss train push train/cohere/config_eval.py --team 34
"""

import os
from pathlib import Path

from truss.base import truss_config
from truss_train import definitions

REPO_ROOT = Path(__file__).resolve().parents[2]

workspace = definitions.Workspace(
    workspace_root=str(REPO_ROOT),
    exclude_dirs=[str(REPO_ROOT / d) for d in
                  (".git", ".venv", "results", "serve", "app", "plans", "bench")],
)

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./train/cohere/sweep.sh"],
    environment_variables={
        "VOICEBRIDGE_DATA": "/tmp/voicebridge-data",
        "HF_TOKEN": definitions.SecretReference(name="hf_token"),
        "SWEEP_STEPS": os.environ.get("SWEEP_STEPS", "900"),
        # The adapter travels in the workspace: 13.7 MB, cheaper than wiring
        # cross-job checkpoint mounts for a file this small.
        "ADAPTER_PATH": "./checkpoints/cohere/best",
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(enabled=True),
)

job = definitions.TrainingJob(
    image=definitions.Image(base_image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"),
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H100, count=1
        ),
        node_count=1,
    ),
    runtime=runtime,
    workspace=workspace,
)

training_project = definitions.TrainingProject(name="voicebridge-sweep-t34", job=job)
