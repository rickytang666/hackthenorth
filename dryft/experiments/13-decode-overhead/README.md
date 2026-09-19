# Cached positions and decode overhead

Baseline: `faece38` (direct KV-cache writes). Candidate is the working tree;
no GPU speedup or official score has been measured for this experiment.

Changes enabled by default:

- Precompute default RoPE tables with the loaded reference rotary module during
  shape initialization. Reuse a slice for prefill and device-indexed rows for
  decode/verification. Non-default rotary variants retain native calculation.
- Omit dense decode and verifier masks only when every layer explicitly declares
  that its attention kernel enforces causality using absolute positions. Native
  and mixed layer paths retain masks, including static-cache capacity masking.

The tables contain only mathematical position data, never prompt contents.
The existing cache reset and speculative rollback semantics are unchanged.

An additional residual-add/RMSNorm Triton kernel is opt-in only. It stores the
rounded BF16 residual sum and consumes that sum directly for normalization.
The experimental layer path applies it to decode/verification rows that do not
already use fused norm/MLP projection. It remains disabled by default because
there is no GPU timing or numerical result yet.

## Local validation

Python 3.11, torch 2.5.1, transformers 4.51.3, safetensors 0.5.3 and tokenizers
0.21.1 installed in the ignored `.venv`. CPU tests verify exact FP32/BF16 RoPE
table equivalence at changed and boundary positions; tiny-model prefill/decode
logits match exactly with cached versus native RoPE. Existing fixed-cache and
speculative rollback checks pass. GPU tests require CUDA and Triton and were
not executed on this Mac. Source parsing and local archive packaging pass.
The Dryft binary is absent, so the official archive validator was not run.

## GPU measurements still required

Use the pinned CUDA 12.4 / torch 2.5.1 / Triton 3.1 environment and the pinned
local Qwen checkpoint. Run every comparison in a fresh process. The runner
does not download weights or submit official runs.

```sh
python agent/benchmark_gpu.py --engine-dir engine --model-path /path/to/checkpoint \
  --batch 4 --prompt-length 2048 --output-length 32 \
  --prompts /path/to/b4-prompts.json --profile \
  --output experiments/13-decode-overhead/b4-candidate.json
```

The prompt JSON is a list of six batches: one warmup plus five measured
samples, each `[batch, prompt_length]`. Use diverse natural-language and code
prompts, with the same file for every candidate. Without a prompt file, the
runner uses reproducible random token IDs as a diagnostic only; these do not
measure realistic prompt-lookup acceptance.

Run these separate configurations to isolate each change:

1. `--native-rope --keep-decode-mask`: previous position/mask behavior.
2. `--keep-decode-mask`: cached RoPE alone.
3. No flags: cached RoPE plus mask removal.
4. `--fuse-residual-norm`: additional experimental residual/normalization fusion.

Also compare the unmodified `faece38` engine from a separate checkout, using
`--engine-dir` and no experiment flags. Alternate baseline/candidate process
order. Test public shapes `(1,512,32)`, `(4,2048,32)`, `(16,512,128)` and batch
boundaries 2, 8, 24, 32, 33 and 64 with suitable prompt/output lengths.

The runner reports wall-clock TTFT/TPOT/throughput, allocator peak memory,
sample range/median (not the official stability statistic), and an independent
native teacher-forced replay on every emitted prefix. Profiling is a separate
untimed generation that writes a Chrome trace and kernel table beside the JSON.
No stdout is written during timed generation. External Dryft pipe timing and
official memory/correctness gates remain authoritative.

Inspect the new profile before changing projection dispatch. In particular,
measure vocabulary projection and batch-one output/down projections. Do not
enable new projection configurations based solely on microbenchmarks.

Remote execution is currently blocked: this machine has no CUDA device,
Modal installation/configuration, Modal token, or Dryft token in the session.
