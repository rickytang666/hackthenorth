# Isolated prefill research

This experiment owns only prefill execution and graph capture. It does not
change decode attention, projection kernels, speculative decoding, or the
separate `.claude/worktrees/fp8-spec` worktree. Its Modal app is
`dryft-prefill-graph-research`, distinct from that session's app.

The baseline is an immutable copy of commit
`9c40eb70a77104f934972549f2fc5c902c1c93e5` under
`/tmp/dryft-prefill-baseline/engine`. Recreate it with:

```sh
mkdir -p /tmp/dryft-prefill-baseline
git archive 9c40eb70a77104f934972549f2fc5c902c1c93e5 engine | tar -x -C /tmp/dryft-prefill-baseline
modal run agent/prefill_research/run.py --mode production
```

Modes: `quick` tests the three public shapes and profiles eager prefill;
`full` tests the research adapter across eight shapes and teacher-forces
generated tokens; `production` runs all GPU tests and repeats the full A/B
using the actual production source. Each A/B uses five alternating paired
trials, distinct corpus windows, and identical inputs on both engines.

The full engine test includes prefill, decode, token transfer, and iteration
over the generator. It is not a kernel-only throughput estimate. The one-token
case tests prefill-only generation; its third timing column is generator
cleanup time, not meaningful TPOT.

CUDA graph input addresses are fixed, but input IDs are overwritten before
each replay. Every replay computes fresh KV entries and resets decode state.
The design follows PyTorch's [CUDA graph guidance](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

Results are saved in `experiments/15-prefill-graph/`. The initial production
launcher failed before any tests because its local path calculation ran on
the remote worker; that app was stopped and the path handling corrected.
