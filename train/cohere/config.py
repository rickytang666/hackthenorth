"""Baseten Training Job for the Cohere LoRA lane.

    uv run truss train push train/cohere/config.py
    uv run truss train logs --job-id "$JOB_ID" --tail
"""

import os

from truss.base import truss_config
from truss_train import definitions

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./train/cohere/run.sh"],
    environment_variables={
        "VOICEBRIDGE_DATA": "/data/voicebridge",
        "HF_TOKEN": definitions.SecretReference(name="hf_token"),
        "MAX_STEPS": os.environ.get("MAX_STEPS", "400"),
        "PROBE_STEPS": os.environ.get("PROBE_STEPS", "0"),
    },
    cache_config=definitions.CacheConfig(enabled=True),
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
)

training_project = definitions.TrainingProject(name="voicebridge-cohere-torgo", job=job)
