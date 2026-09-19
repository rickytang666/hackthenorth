# Optimization experiments

The best completed official candidate passed and ranked at **748.07 tok/s**,
up from **354.37 tok/s**. Subsequent candidates must pass the same gates and
improve this measured result.

| Snapshot | Change | Status |
| --- | --- | --- |
| `00-baseline` | Unchanged upstream engine | Archived locally; API upload rejected (HTTP 405) |
| `01-fused-norm` | Replace all hidden and Q/K RMSNorms with the bundled Triton kernel | Archived locally; GPU evaluation pending |
| `02-cuda-graph` | Fused norms, fixed KV buffers, CUDA graph decode, direct decoder-layer dispatch | Passed; 354.37 tok/s; commit `11d5c55` |
| `03-fused-gqa` | Direct grouped KV attention, fused Q/K norm + RoPE, fused SwiGLU | Passed; 706.61 tok/s; commit `3996403` |
| `04-exact-speculation` | Prompt-lookup drafts, captured multi-token verification, exact acceptance and cache rollback | Passed; 693.60 tok/s; commit `cc98be4` |
| `05-packed-projections` | One decode QKV projection and one MLP gate/up projection; stride-aware fused kernels | Passed; 748.07 tok/s; commit `03f4123` |
| `06-packed-no-speculation` | Same packed projections, with prompt lookup disabled | Failed a hidden output check; public cases passed; rejected |
| `07-confirm-packed` | Exact engine source from the passing 748.07 tok/s candidate | Fresh-prompt confirmation pending |

First passing public results (five samples each):

| Workload | tok/s | TTFT | TPOT | Speedup over paired native |
| --- | ---: | ---: | ---: | ---: |
| B1, 512 → 32 | 125.87 | 20.39 ms | 7.53 ms | 2.43× |
| B4, 2048 → 32 | 201.81 | 163.87 ms | 15.18 ms | 1.38× |
| B16, 512 → 128 | 953.04 | 153.56 ms | 15.71 ms | 1.42× |

These public workloads do not set the 354.37 tok/s ranking; the hidden six do.
The user reports the current whole-run limit is 15 minutes after pickup,
excluding queue time. Per-engine load/warmup and sample budgets remain 300 s.

Candidate 03 public throughput is 177.08 / 371.85 / 2240.75 tok/s, with decode
steps at 5.19 / 5.92 / 6.02 ms. All gates passed; peak memory across the complete
run was 13.94 GiB. The official score improved by 1.994× over candidate 02.

Candidate 05 passed at 748.07 tok/s, with public throughput 179.64 / 389.10 /
2393.73 tok/s and decode steps at 4.92 / 5.51 / 5.57 ms. Peak memory across the
complete run was 14.94 GiB. Candidate 06 failed correctness on a hidden case,
so its public speed improvements cannot be used. Candidate 07 restores exactly
the candidate 05 engine source for confirmation on fresh prompts.

Each local snapshot holds `engine.tar.gz`; archives are ignored by git. The
upstream baseline is also available from the original git history.

The graph candidate captures one decode step during the initial warmup. Replay
updates the input token and absolute position on the GPU. Prefill still uses
causal SDPA over just the new prompt. During decode an explicit mask excludes
unused cache capacity. Each generation overwrites prompt slots and resets the
decode position. The model weights and arithmetic precision remain BF16.

## Validation

Run `.venv/bin/python -m unittest discover -s tests -v`. Tests use a small random
Qwen3 model to compare fixed-cache tokens with full-prefix native forwards,
including stale cache contents, repeated prompts, batch changes, a one-token
prompt and a one-token output. A separate CUDA test checks captured replay and
state reset; it is skipped on this Mac. These are not substitutes for the
platform's full-model teacher-forced validation.

`./bin/dryft validate engine` passes. CPU checks use PyTorch 2.5.1 and
Transformers 4.51.3, matching the benchmark's model APIs.

## Current submission workflow

As of 2026-09-19, the live [participant guide](https://htn.dryft.ai/docs) differs
from the starter docs: new submissions require a connected GitHub repository,
public runs are retired, and official ranking uses six hidden workloads.
`dryft submit` and the starter's `Dryft.submit` return HTTP 405. Token
authentication was verified with `dryft doctor`.

Create/connect the approved private repository at https://htn.dryft.ai/repos
with engine folder `engine`. The connection requires GitHub App access. Push
the unchanged baseline first, followed by each candidate, and collect official
results before retaining or extending an optimization. Avoid queuing duplicate
runs while waiting for GPU capacity.

Use the tracking helper for each new iteration:

```sh
python3 agent/experiment.py snapshot 03-next-change
# Push the candidate to the connected default branch, then copy its run ID.
python3 agent/experiment.py track 03-next-change RUN_ID
python3 agent/experiment.py status 03-next-change
python3 agent/experiment.py logs 03-next-change
```

The helper reads the ignored, owner-readable `.env`, stores reports outside the
submitted engine, and never uploads through the retired endpoint. Compare
throughput, TTFT, TPOT, peak memory, correctness and sample spread for every
workload. Candidate 03 eliminates the pinned adapter's grouped-head expansion
in decode and uses a split-cache attention reduction. Prefill stays native.

Candidate 04 proposes three tokens from earlier matching four-token sequences
in the current prompt and verified output. It runs only at batch 1; other
batches use regular captured decode. Qwen verifies current + draft in one
causally masked forward, accepts only the matching greedy prefix, and emits one
target correction or bonus token. Predictions after the first rejection are
discarded. Cache position advances by the number of emitted tokens, masking
the speculative tail. No draft model, approximate output or extra weights are
used. Prompt-dependent acceptance may affect the 25% timing-spread gate, so
the complete official evaluation matters. Candidate 04 passed all gates but
scored about 1.8% below candidate 03. Candidate 06 tested disabling speculation
but failed a hidden correctness check. Different runs use different prompts;
this does not establish why 06 failed or prove speculation fixes a numerical
issue. Further optimization is paused pending confirmation and GPU profiling.

Candidate 05 retains candidate 04 and packs projection weights during loading.
Q/K/V prefill modules share slices of the packed decode weight, avoiding weight
duplication. Fused norm/RoPE and SwiGLU consume the resulting strided views
directly. It removes two QKV matrix launches per decode layer and one gate/up
launch per layer for both prefill and decode; weights remain BF16.
