# Adopt decode/token-delivery overlap

Ported the complete scheduling change and CPU scheduling test from
[StephenShao90/dryftChallenge commit bb46600905ef53a39aaefefdbd607645ab9cff88](https://github.com/StephenShao90/dryftChallenge/commit/bb46600905ef53a39aaefefdbd607645ab9cff88).
Local parent: `eb43949`.

For ordinary generation, launch the next token's CUDA graph before yielding
the current token's already-materialized Python list. The next GPU decode can
therefore overlap the caller's serialization and delivery of the current
token. There are exactly output_length - 1 decode replays, no replay for a
single-token response, and no pending replay at the final yield. The
speculative path retains its existing verification schedule.

Our newer fused greedy head, batch-one down/residual projection, and prefill
graphs remain in place. This port does not change model arithmetic, cache
layout, or kernel implementations. The separate Claude worktree is untouched.

The user reported 938 tok/s for the source commit. That is not a measured
score for this combined engine. The requested before/after comparison uses
an immutable snapshot of `eb43949` and the adopted production source on the
same H100, with identical prompts and alternating run order. A separate
reader process timestamps JSON token lines sent through a pipe, including
the serialization and delivery that this scheduling change can overlap.
This is a local transport benchmark, not the official Dryft harness.


## Verified before/after result

[Modal H100 run](https://modal.com/apps/afreedhassan/main/ap-aDcaDZ3pDhVqUZl9d1knML).
Five alternating paired trials per shape, distinct corpus windows and
identical input IDs per pair. Both engines include our fused greedy head,
B1 down/residual kernel, and prefill graphs; overlap is the only production
code difference.

| Batch | Prompt/output | Before TPOT | After TPOT | Total throughput gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512/32 | 3.971 ms | 3.928 ms | +1.00% |
| 4 | 2048/32 | 4.703 ms | 4.498 ms | +2.00% |
| 16 | 512/128 | 4.757 ms | 4.555 ms | +3.80% |
| 2 | 256/32 | 4.174 ms | 3.964 ms | +4.94% |
| 8 | 256/32 | 4.278 ms | 4.119 ms | +3.42% |
| 24 | 256/32 | 4.906 ms | 4.694 ms | +3.09% |
| 32 | 256/32 | 4.997 ms | 4.817 ms | +1.98% |

B1's speculative path does not use the new scheduler; its small change is
run-to-run variation, not evidence of an overlap benefit. The one-token case
also has no next decode step to overlap and was checked as an edge case.

All 26 tests passed on the H100. Locally, 11 tests passed and 15 CUDA tests
were skipped. Native BF16 teacher-forced replay passed 21,605 tokens per
engine with zero failing positions and worst logit deficit 1.375 for both.
All paired generations produced identical token IDs. No full-model speedup
is inferred from an isolated kernel test: the reported gains include prefill,
all decode steps, and streamed delivery, but remain local measurements.
