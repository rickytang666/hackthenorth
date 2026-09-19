# Direct KV-cache writes during QK normalization/RoPE

Baseline: FlashAttention prefill commit `388469b`. Its official run was
still queued when this diagnostic experiment completed.

Write rotated K and unchanged V directly to their static cache slots in the
QK normalization/RoPE kernel. This removes two separate index-copy kernels
per transformer layer (72 launches per decode step) and the intermediate K
buffer. The existing math and BF16 cast boundaries are preserved. The path
also handles the four-query exact speculative verifier. Prefill retains the
FlashAttention path.

Final-source H100 comparison, five alternating A/B samples per shape,
using the same diverse corpus prompt generator as candidate 11:

| Batch | Prompt/output | Baseline TPOT ms | Candidate TPOT ms | TPOT reduction | Total throughput gain |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 512/32 | 3.990 | 3.798 | 4.8% | 4.6% |
| 4 | 2048/32 | 4.737 | 4.533 | 4.3% | 2.6% |
| 16 | 512/128 | 4.819 | 4.597 | 4.6% | 4.3% |
| 2 | 256/32 | 4.221 | 4.038 | 4.3% | 4.1% |
| 8 | 256/32 | 4.389 | 4.185 | 4.6% | 3.9% |
| 24 | 256/32 | 4.956 | 4.736 | 4.4% | 2.4% |

Seven GPU tests passed. The new test requires bit-for-bit equality to the
separate normalization/RoPE and cache update operations, including packed
QKV strides, nonconsecutive cache positions, CUDA graph replay with changed
inputs/positions, preservation of untouched slots, and rollback/overwrite.
It covers B1/B4/B16 single-token decode and B1/B3 four-token verification.

Full-model teacher-forced checks passed 16,480 tokens per engine with zero
positions beyond the 2.0-logit margin; worst deficit was 1.375 for both.
Eight local CPU tests passed; nine CUDA-only tests were skipped locally.

This is a measured decode improvement, not an official leaderboard score.
Raw timing samples and corpus hashes are in `measurements.json`.
