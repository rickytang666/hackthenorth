import json
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent if modal.is_local() else Path('/research/check')
ROOT=HERE.parents[1] if modal.is_local() else Path('/research')
app=modal.App('dryft-bf16-stat-fusion')
cache=modal.Volume.from_name('dryft-qwen-research-cache')
image=(modal.Image.from_registry('pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel')
       .pip_install('transformers==4.51.3','safetensors==0.5.3','tokenizers==0.21.1','numpy==2.2.6')
       .env({'HF_HOME':'/cache/huggingface','TRITON_CACHE_DIR':'/cache/codex-fp8-selective-triton-v1'})
       .add_local_dir(ROOT/'engine','/research/engine',ignore=['**/__pycache__/**'])
       .add_local_dir(ROOT/'agent/prefill_research','/research/prompts',ignore=['**/__pycache__/**'])
       .add_local_dir(HERE,'/research/check',ignore=['**/__pycache__/**']))
@app.function(image=image,gpu='H100!',cpu=8,memory=32768,timeout=180,max_containers=1,scaledown_window=2,volumes={'/cache':cache})
def experiment(mode:str):
    import subprocess,sys
    process=subprocess.Popen([sys.executable,'-u','/research/check/'+mode+'.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    lines=[]
    for line in process.stdout:
        print(line,end='',flush=True);lines.append(line)
    result=dict(exit_code=process.wait(),output=''.join(lines));cache.commit();return result
@app.local_entrypoint()
def main(mode:str='micro'):
    result=experiment.remote(mode)
    (ROOT/f'experiments/35-bf16-stat-fusion/{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
