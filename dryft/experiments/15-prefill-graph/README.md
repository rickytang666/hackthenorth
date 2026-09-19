# Fixed-shape prefill CUDA graph

Baseline: `9c40eb70a77104f934972549f2fc5c902c1c93e5`.
Production validation: [Modal H100 run](https://modal.com/apps/afreedhassan/main/ap-8PLlu6DzH9ya099Y5HFfjn).

The existing decode CUDA graphs leave prefill eager. For at most 2,048 total
prompt tokens, capture the existing prefill operations during warmup and
replay with fresh input IDs. Every replay recomputes the prompt KV entries,
sets the current token, and resets decode position. Larger prefills retain
the existing eager path. No decode kernels or model arithmetic change.

The initial B1 profile measured 9.30 ms of GPU work while eager TTFT was
14.88 ms; graph replay reduced TTFT to 9.47 ms. Large public prefills showed
no meaningful graph benefit, motivating the small-prefill dispatch.
This follows [PyTorch CUDA graph guidance](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
with the speedup established by this experiment rather than inferred from it.

## Final production A/B

Five alternating paired trials per shape on distinct corpus windows. Timing
includes prefill, decode, host token transfer, and generator iteration.

| Batch | Prompt/output | Baseline TTFT | Candidate TTFT | Total throughput gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512/32 | 15.22 ms | 9.60 ms | +4.85% |
| 4 | 2048/32 | 123.37 ms | 123.84 ms | -0.26% |
| 16 | 512/128 | 111.84 ms | 111.95 ms | -0.04% |
| 2 | 256/32 | 15.08 ms | 9.42 ms | +4.43% |
| 8 | 256/32 | 28.96 ms | 27.89 ms | +0.52% |
| 24 | 256/32 | 79.85 ms | 82.01 ms | -0.74% |
| 32 | 256/32 | 111.74 ms | 110.42 ms | +0.35% |
| 1 | 128/1 | 13.17 ms | 5.40 ms | +143.42% |

Large prefills do not enter the new path; their sub-1% variations reflect
run-to-run timing variation. The one-output-token case measures prefill-only
generation, not sustained decode. These are local measurements, not an
estimated official leaderboard score.

## Verification

- All 25 tests passed on H100; local checks passed 10 tests with 15 CUDA skips.
- 21,605 generated tokens per engine passed native BF16 teacher-forced replay.
- Both engines' worst logit deficit was 1.375, below the 2.0 gate.
- Every paired trial produced identical token IDs.
- New tests compare all cache entries and decode tokens after changing prompts,
  overwriting cache contents, and corrupting token/position state before replay.
- Output-length-one and eager-fallback cases pass.
- The research adapter was separately measured and replay-verified before the
  exact production source was validated. Final-source evidence is in
  `measurements.json`; `snapshot.json` identifies the archive.

This work did not modify the separate `.claude/worktrees/fp8-spec` worktree,
its experiments, or its GPU apps. Prefill research uses a separate Modal app
and a frozen baseline. The first production launcher failed before tests due
to a remote path initialization error; it was stopped and corrected.
