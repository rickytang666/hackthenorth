"""Isolated H100 experiment for prefill; frozen baseline and separate app."""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/research/prefill")
ROOT = HERE.parents[1] if modal.is_local() else Path("/research/production")
app = modal.App("dryft-prefill-graph-research")
cache = modal.Volume.from_name("dryft-qwen-research-cache")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "numpy==2.2.6")
    .env({"HF_HOME": "/cache/huggingface", "TRITON_CACHE_DIR": "/tmp/prefill-triton"})
    .add_local_dir("/tmp/dryft-prefill-baseline/engine", "/research/engine")
    .add_local_dir(HERE, "/research/prefill", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "engine", "/research/production/engine", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "tests", "/research/production/tests", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "agent/client.py", "/research/production/agent/client.py")
)


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=1200,
              max_containers=1, scaledown_window=2, volumes={"/cache": cache})
def experiment(mode: str):
    import subprocess
    import sys
    if mode == "production":
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                               cwd="/research/production", capture_output=True, text=True)
        print(tests.stdout + tests.stderr, flush=True)
        if tests.returncode:
            return {"exit_code": tests.returncode, "output": tests.stdout + tests.stderr}
    process = subprocess.Popen([sys.executable, "-u", "/research/prefill/bench.py", mode],
                               cwd="/research", stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    result = {"exit_code": process.wait(), "output": "".join(lines)}
    if mode == "production":
        result["gpu_tests"] = tests.stdout + tests.stderr
    return result


@app.local_entrypoint()
def main(mode: str = "quick"):
    result = experiment.remote(mode)
    target = HERE.parents[1] / "experiments" / "15-prefill-graph" / f"{mode}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", target)
