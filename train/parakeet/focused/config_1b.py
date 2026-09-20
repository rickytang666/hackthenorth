"""Parakeet 1.1B comparison: fixed greedy, max 1500 steps or plateau per arm."""
from truss.base import truss_config
from truss_train import WeightsSource, definitions
runtime=definitions.Runtime(
 start_commands=['python -m pip install "jiwer>=3,<5"','timeout 7200 bash -c "set -e; python -u -m train.parakeet.focused.run baseline; python -u -m train.parakeet.focused.run adapter; python -u -m train.parakeet.focused.run partial; python -u -m train.parakeet.focused.run final_frozen"'],
 environment_variables={'VOICEBRIDGE_DATA':'data/voicebridge','PARAKEET_MODEL_PATH':'/app/models/parakeet','PYTHONPATH':'.','FOCUSED_MODEL_ID':'nvidia/parakeet-tdt-1.1b','FOCUSED_MAX_STEPS':'1500','FOCUSED_NORMALIZE_TARGETS':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'},
 cache_config=definitions.CacheConfig(enabled=True),checkpointing_config=definitions.CheckpointingConfig(enabled=True))
job=definitions.TrainingJob(image=definitions.Image(base_image='nvcr.io/nvidia/nemo:25.04.03'),compute=definitions.Compute(node_count=1,cpu_count=16,memory='96Gi',accelerator=truss_config.AcceleratorSpec(accelerator=truss_config.Accelerator.H100,count=1)),runtime=runtime,weights=[WeightsSource(source='hf://nvidia/parakeet-tdt-1.1b',mount_location='/app/models/parakeet')],workspace=definitions.Workspace(workspace_root='../../..',external_dirs=['/private/tmp/voicebridge-focused/data'],exclude_dirs=['../../../.git','../../../.venv','../../../.workspace','../../../data','../../../results']))
training_project=definitions.TrainingProject(name='voicebridge-focused-1b-adaptation-t34',job=job)
