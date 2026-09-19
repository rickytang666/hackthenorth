# Latest worktree decode fusion adoption

Source: worktree commit 58be5a7ca84d2e98469d1bb3e1c739c3741ed2f5 plus its
88c8951 calibration-interface/gate-up dependency. Baseline: main 5a8839b,
whose engine is byte-identical to e134e74. Worktree files and jobs are untouched.

The previously reverted prefill changes are deliberately excluded: prefill
still uses FlashAttention, with a 2048-total-token CUDA graph cutoff.

The new candidate folds Q/K normalization, RoPE, and the newest cache entry
into split decode attention. Old cache reads exclude the current position;
only the final split persists and contributes the newest KV, so concurrent
blocks do not read a slot while another block writes it. The merge kernel
remains. Multiple-query speculative verification retains the old attention
path. Batch-one normalized projection paths remain as before.

Two further candidates fuse RMSNorm into QKV and the residual into down
projection. Whole-graph calibration chooses between fused and legacy paths
on the actual run GPU. Gate/up tile calibration is included as a dependency.
The source changes hysteresis to 0.5% and graph timing to 20 replays.

New GPU tests exercise current-position zero, split boundaries, final cache
slots, poisoned unused tails, changing packed inputs, both projection weight
layouts and tail columns. Full engine verification uses diverse prompts,
five alternating paired samples on eight shapes, JSON token delivery, and
native BF16 teacher-forced replay on each engine's own output prefixes.

## H100 results

All 30 GPU tests passed. Five alternating paired trials per shape.

| Batch | Prompt/output | Throughput change | Baseline TPOT ms | Candidate TPOT ms |
|---:|---:|---:|---:|---:|
| 1 | 512/32 | +0.18% | 3.786 | 3.778 |
| 4 | 2048/32 | +1.18% | 4.443 | 4.372 |
| 16 | 512/128 | +1.51% | 4.479 | 4.398 |
| 2 | 256/32 | +2.12% | 3.904 | 3.817 |
| 8 | 256/32 | +0.60% | 4.050 | 4.007 |
| 24 | 256/32 | -0.11% | 4.632 | 4.572 |
| 32 | 256/32 | +0.86% | 4.741 | 4.686 |
| 1 | 128/1 | +0.34% | 0.000 | 0.000 |

Full teacher-forced replay checked 21,605 tokens per engine, zero failed positions, worst logit deficit 0.75 for both. The local B24 result was 0.11% slower; other tested shapes improved. Public timing gains are not official leaderboard scores.
