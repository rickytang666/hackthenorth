"""Baseten Training Job for the Cohere LoRA lane.

    uv run truss train push train/cohere/config.py
    uv run truss train logs --job-id "$JOB_ID" --tail
"""

import os
from pathlib import Path

from truss.base import truss_config
from truss_train import definitions

REPO_ROOT = Path(__file__).resolve().parents[2]

# Upload the whole repo, not just this directory: run.sh needs contract/,
# data/ and train/, and the manifest hashes must be the committed ones.
workspace = definitions.Workspace(
    workspace_root=str(REPO_ROOT),
    # Only top-level children may be excluded, and they resolve against this
    # file's directory, so pass absolute paths. The dataset lives outside the
    # repo entirely (see env.sh); `serve` and `app` are not needed to train.
    exclude_dirs=[str(REPO_ROOT / d) for d in
                  (".git", ".venv", "results", "serve", "app", "plans", "bench")],
)

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./train/cohere/run.sh"],
    environment_variables={
        "VOICEBRIDGE_DATA": "/tmp/voicebridge-data",
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
    workspace=workspace,
)

training_project = definitions.TrainingProject(name="voicebridge-cohere-lora-torgo-t34", job=job)
