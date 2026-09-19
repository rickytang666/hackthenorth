# Adopt the Claude worktree's measured improvements

Source: `39e0a0a43b1d6df129a36a5069690abc6097e429`, including its calibration
parent `98df5861d106afc9b68a41f83760989d0fb87bd5`, from the
`.claude/worktrees/fp8-spec` branch. Baseline: `92070fb` on our main branch.
The source worktree was only read; its files, branch, and GPU jobs were not
modified.

Adopted production changes:

- Enable the existing BF16 residual-add/RMSNorm fusion for decode paths that
  do not already fuse the MLP input normalization into a projection.
- Before production graph capture, calibrate projection choices by timing
  complete decode/verification CUDA graphs, with a 2% preference for the
  existing configuration. Candidate outputs are checked against native linear.
- Add the source's pipelined BF16 dot kernel and expanded tile/layout search.

The source measurements found about 0.7–0.9% improvement from residual-norm
fusion and approximately parity from calibration. Calibration is intended
to adapt to GPU-dependent kernel rankings, not promise an additional gain.
The FP8 draft, layer-skipping draft, persistent fusion experiments, and
isolated microbenchmark winners rejected in the source notes are not enabled.

Source evidence is saved in `source-autotune_ab.json`,
`source-fusenorm_ab.json`, and `../fp8-draft-research.md`. The isolated Modal
launcher and benchmark are in `agent/worktree_research/`.

Integration changes keep the calibration import inside the GPU capture
methods, preserving the existing CPU-only decode tests. Documentation was
corrected to describe graph replay timing and avoid promising that timing
hysteresis guarantees no regressions. The source's production algorithm is
otherwise preserved, except for the layout correction below.

The first local integration test passed all 21,605 tokens per engine but
regressed B2 by 1.31%. Inspection found that the source's `_legacy` choice
retained tile configurations but dropped the column layout when falling back
to native linear. Our existing B2/B24/B32 MLP down projection uses that
column layout. The port now preserves the native-linear layout in the
fallback and offers both native layouts during calibration. Initial-run
evidence is retained in `initial-production.json` and `initial-production.log`.

New GPU coverage checks the projection candidate's row padding, masked
output channels, both weight layouts, both tile choices, and CUDA graph
replays with changed input values. The full A/B resets calibration choices
between workloads, as each official workload starts a fresh engine.
Each pair gets identical diverse prompts; token delivery is measured through
a pipe to a separate process. Native BF16 teacher-forced replay checks both
engines' own generated prefixes. This is a local transport benchmark, not
the official Dryft harness.


## Final integration validation

[H100 run](https://modal.com/apps/afreedhassan/main/ap-usy61fVoL9ebNyuv4asx2x). Five paired samples per shape.

| Batch | Prompt/output | Before TPOT | After TPOT | Total throughput gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512/32 | 3.962 ms | 3.940 ms | +0.49% |
| 4 | 2048/32 | 4.484 ms | 4.448 ms | +0.77% |
| 16 | 512/128 | 4.552 ms | 4.504 ms | +0.96% |
| 2 | 256/32 | 3.944 ms | 3.923 ms | +0.49% |
| 8 | 256/32 | 4.116 ms | 4.068 ms | +1.13% |
| 24 | 256/32 | 4.689 ms | 4.640 ms | +0.10% |
| 32 | 256/32 | 4.785 ms | 4.722 ms | +0.77% |

All 27 GPU tests passed; local tests passed 11 with 16 CUDA skips. Native
BF16 teacher-forced replay checked 21,605 tokens per engine: zero failed
positions, worst logit deficit 1.375 for each engine. Some near-tie greedy
choices differ, so token-for-token equivalence is not claimed.

Maximum measured candidate warmup after model loading was 26.03 seconds.
The single-output-token edge case also passed, with a -1.25% (0.071 ms)
local timing difference; it does not execute the changed decode path.
The multi-token gains above are small local improvements, not an official
score forecast. Final raw timings, calibration choices and correctness totals
are recorded in `measurements.json`.
