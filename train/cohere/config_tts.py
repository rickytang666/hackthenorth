"""Baseten job that times OpenVoice synthesis on the serving GPU.

Whether chunked streaming TTS gets built is decided by this number, so it has
to be measured on the hardware the renderer is served on, not on a laptop.

    uv run truss train push train/cohere/config_tts.py --team 34
"""

import os
from pathlib import Path

from truss.base import truss_config
from truss_train import definitions

REPO_ROOT = Path(__file__).resolve().parents[2]

workspace = definitions.Workspace(
    workspace_root=str(REPO_ROOT),
    # serve/ is NEEDED here: the renderer lives in serve/voice.
    exclude_dirs=[str(REPO_ROOT / d) for d in
                  (".git", ".venv", "results", "app", "plans", "bench", "checkpoints")],
)

runtime = definitions.Runtime(
    start_commands=["/bin/bash ./train/cohere/tts_bench.sh"],
    environment_variables={
        "VOICEBRIDGE_DATA": "/tmp/voicebridge-data",
        "HF_TOKEN": definitions.SecretReference(name="hf_token"),
        "EVAL_MANIFEST": os.environ.get("EVAL_MANIFEST", "torgo_dys_dev.jsonl"),
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

training_project = definitions.TrainingProject(name="voicebridge-tts-bench-t34", job=job)
