# Optimization experiments

No H100 measurements have been collected yet. The current engine is a candidate,
not a demonstrated speedup.

| Snapshot | Change | Status |
| --- | --- | --- |
| `00-baseline` | Unchanged upstream engine | Archived locally; API upload rejected (HTTP 405) |
| `01-fused-norm` | Replace all hidden and Q/K RMSNorms with the bundled Triton kernel | Archived locally; GPU evaluation pending |
| `02-cuda-graph` | Fused norms, fixed KV buffers, CUDA graph decode, direct decoder-layer dispatch | Current candidate; GPU evaluation pending |

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
workload. In particular, fixed-capacity attention expands grouped K/V heads in
the pinned Transformers adapter; eliminating that expansion is a possible next
optimization after measuring this candidate.
