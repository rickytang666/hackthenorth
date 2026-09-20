"""Before/after decode overlap, with token delivery through an actual pipe."""
import gc
import json
import statistics
import subprocess
import sys
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/research/prefill")
sys.path.insert(0, "/research/engine")
from engine import Engine
from prompts import corpora, prompts, emit


def run(engine, inputs, length):
    torch.cuda.synchronize()
    # A real pipe carries a JSON line per token to a separate reader process.
    # Include serialization/flush and receiver timestamps, the work that the
    # adopted scheduler overlaps with the following GPU decode step.
    receiver_code = """
import json, sys, time
print('ready', flush=True)
arrivals, delivered = [], []
for line in sys.stdin:
    arrivals.append(time.perf_counter())
    delivered.append(json.loads(line))
print(json.dumps([arrivals, delivered]), flush=True)
"""
    receiver = subprocess.Popen([sys.executable, "-u", "-c", receiver_code],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, bufsize=1)
    assert receiver.stdout.readline().strip() == "ready"
    start = time.perf_counter()
    outputs = []
    try:
        for tokens in engine.generate(inputs, length):
            outputs.append(tokens)
            receiver.stdin.write(json.dumps(tokens, separators=(",", ":")) + "\n")
            receiver.stdin.flush()
    finally:
        receiver.stdin.close()
    arrivals, delivered = json.loads(receiver.stdout.read())
    assert receiver.wait() == 0
    assert delivered == outputs
    first, end = arrivals[0], arrivals[-1]
    assert len(outputs) == length
    assert all(len(row) == len(inputs) for row in outputs)
    return outputs, [end-start, first-start, (end-first)/max(1, length-1)]


@torch.inference_mode()
def main(mode):
    assert mode == "production"
    if mode == "production":
        # Keep the baseline class bound to its frozen modules while importing
        # the exact source that would be shipped as the candidate.
        for name in list(sys.modules):
            if name in {"engine", "attention", "decode", "fused_layer", "speculate",
                        "projections", "prefill", "kernels"} or name.startswith("kernels."):
                del sys.modules[name]
        sys.path.remove("/research/engine")
        sys.path.insert(0, "/research/production/engine")
        from engine import Engine as ProductionEngine
        mode = "full"
    emit("environment", delivery="json_lines_pipe_reader_process", gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
    path = snapshot_download("Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    streams = corpora(tokenizer)
    baseline = Engine(path)
    candidate = ProductionEngine(path)
    cases = [(1,512,32), (4,2048,32), (16,512,128)]
    if mode == "full":
        cases += [(2,256,32), (8,256,32), (24,256,32), (32,256,32), (1,128,1)]
    records = []
    for shape in cases:
        inputs = prompts(tokenizer, streams, shape, -1)[0]
        # Match fresh-workload calibration: a previous shape must not choose
        # projection kernels for this shape's decode or verification graphs.
        import projections
        projections._choices.clear()
        for name, engine in (("baseline", baseline), ("candidate", candidate)):
            start = time.perf_counter()
            run(engine, inputs, shape[2])
            emit("warmup", name=name, shape=shape, seconds=time.perf_counter()-start,
                 allocated_gib=torch.cuda.memory_allocated()/2**30,
                 reserved_gib=torch.cuda.memory_reserved()/2**30)
        timing = {"baseline": [], "candidate": []}
        identical = []
        for trial in range(5):
            inputs = prompts(tokenizer, streams, shape, trial)[0]
            order = [("baseline", baseline), ("candidate", candidate)]
            if trial % 2:
                order.reverse()
            outputs = {}
            for name, engine in order:
                output, times = run(engine, inputs, shape[2])
                outputs[name] = output
                timing[name].append(times)
                records.append((name, inputs, output))
            identical.append(outputs["baseline"] == outputs["candidate"])
        medians = {name: [statistics.median(t[i] for t in samples) for i in range(3)]
                   for name, samples in timing.items()}
        emit("engine_ab", shape=shape, baseline_seconds=medians["baseline"],
             candidate_seconds=medians["candidate"], samples_seconds=timing,
             identical_tokens=identical, speedup=medians["baseline"][0]/medians["candidate"][0])
        if mode == "quick" and shape[0] == 1:
            # Profile warmed eager prefill only, outside the timed A/B samples.
            ids = torch.tensor(inputs, device="cuda")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                baseline.state.prefill(baseline.model, ids)
                torch.cuda.synchronize()
            emit("prefill_profile", table=prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=15))
    if mode != "full":
        return
    del engine, order, baseline, candidate
    gc.collect()
    torch.cuda.empty_cache()
    reference = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    failures = {"baseline": 0, "candidate": 0}
    worst = {"baseline": 0., "candidate": 0.}
    checked = {"baseline": 0, "candidate": 0}
    for name, inputs, outputs in records:
        for row, prompt in enumerate(inputs):
            generated = [step[row] for step in outputs]
            ids = torch.tensor([prompt+generated], device="cuda")
            logits = reference(ids, use_cache=False, logits_to_keep=len(generated)+1).logits[:,:-1].float()
            choice = torch.tensor(generated, device="cuda").view(1,-1,1)
            deficit = logits.amax(-1)-logits.gather(-1,choice).squeeze(-1)
            failures[name] += int((deficit > 2).sum().item())
            checked[name] += deficit.numel()
            worst[name] = max(worst[name], deficit.max().item())
    emit("teacher_forced", failed_positions=failures, checked_tokens=checked, worst_logit_deficit=worst)
    assert not any(failures.values()), failures


if __name__ == "__main__":
    main(sys.argv[1])
