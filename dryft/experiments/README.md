# Optimization experiments

The first official candidate passed and ranked at **354.37 tok/s**. Subsequent
candidates must pass the same gates and improve this measured result.

| Snapshot | Change | Status |
| --- | --- | --- |
| `00-baseline` | Unchanged upstream engine | Archived locally; API upload rejected (HTTP 405) |
| `01-fused-norm` | Replace all hidden and Q/K RMSNorms with the bundled Triton kernel | Archived locally; GPU evaluation pending |
| `02-cuda-graph` | Fused norms, fixed KV buffers, CUDA graph decode, direct decoder-layer dispatch | Passed; 354.37 tok/s; commit `11d5c55` |
| `03-fused-gqa` | Direct grouped KV attention, fused Q/K norm + RoPE, fused SwiGLU | Official run queued; commit `3996403` |
| `04-exact-speculation` | Prompt-lookup drafts, captured multi-token verification, exact acceptance and cache rollback | Local checks passed; awaiting comparison with 03 |

First passing public results (five samples each):

| Workload | tok/s | TTFT | TPOT | Speedup over paired native |
| --- | ---: | ---: | ---: | ---: |
| B1, 512 → 32 | 125.87 | 20.39 ms | 7.53 ms | 2.43× |
| B4, 2048 → 32 | 201.81 | 163.87 ms | 15.18 ms | 1.38× |
| B16, 512 → 128 | 953.04 | 153.56 ms | 15.71 ms | 1.42× |

These public workloads do not set the 354.37 tok/s ranking; the hidden six do.
The user reports the current whole-run limit is 15 minutes after pickup,
excluding queue time. Per-engine load/warmup and sample budgets remain 300 s.

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
this candidate requires a complete official evaluation before promotion.
