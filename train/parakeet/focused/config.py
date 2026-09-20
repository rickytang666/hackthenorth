"""One H100; frozen baseline then matched adapter and selective-unfreezing arms."""
from truss.base import truss_config
from truss_train import WeightsSource, definitions
runtime=definitions.Runtime(
 start_commands=['python -m pip install "jiwer>=3,<5"','timeout 5400 bash -c "set -e; python -u -m train.parakeet.focused.run baseline; python -u -m train.parakeet.focused.run adapter; python -u -m train.parakeet.focused.run partial; python -u -m train.parakeet.focused.run final_frozen"'],
 environment_variables={'VOICEBRIDGE_DATA':'data/voicebridge','PARAKEET_MODEL_PATH':'/app/models/parakeet','PYTHONPATH':'.','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'},
 cache_config=definitions.CacheConfig(enabled=True),checkpointing_config=definitions.CheckpointingConfig(enabled=True))
job=definitions.TrainingJob(image=definitions.Image(base_image='nvcr.io/nvidia/nemo:25.04.03'),compute=definitions.Compute(node_count=1,cpu_count=16,memory='96Gi',accelerator=truss_config.AcceleratorSpec(accelerator=truss_config.Accelerator.H100,count=1)),runtime=runtime,weights=[WeightsSource(source='hf://nvidia/parakeet-tdt-0.6b-v2',mount_location='/app/models/parakeet')],workspace=definitions.Workspace(workspace_root='../../..',external_dirs=['/private/tmp/voicebridge-focused/data'],exclude_dirs=['../../../.git','../../../.venv','../../../.workspace','../../../data','../../../results']))
training_project=definitions.TrainingProject(name='voicebridge-focused-adaptation-t34',job=job)
