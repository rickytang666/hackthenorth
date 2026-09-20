# Gate/up projection with fused SwiGLU

Baseline: official commit `faece38`, which passed at 895.3238 tokens/s.

For 2–32 rows, pack BF16 gate/up weights as interleaved column pairs during
warmup. Compute both projections with one tensor-core GEMM and apply SwiGLU
in its epilogue. Preserve BF16 casts after both projections, after SiLU,
and after the final multiplication. This removes the packed intermediate
activation tensor and one separate SwiGLU launch per layer. B1 uses the
existing norm/projection fusion; its four-query speculative verifier can
use the new epilogue. Prefill above 32 rows retains native cuBLAS.

The kernel follows the tiled GEMM approach in the pinned
[Triton 3.1 matmul](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/ops/matmul.py).
This is a new epilogue implementation, not an upstream performance claim.

A bounded H100 sweep rotated real weights from layers 0, 11, 23, and 35
beyond L2 capacity. All 36 tested variants matched native BF16 outputs
bit-for-bit on the tested activations, at scales 0.25, 1, and 4. The selected
16x128x64 tile (four warps, three stages) improved gate/up plus SwiGLU time
by 3.8–5.9% across batches 2, 4, 8, 16, 24, and 32.

Full-generation measurements use five alternating matched A/B samples per
shape and diverse corpus prompts. Throughput includes prefill.

| Batch | Prompt/output | Baseline TPOT ms | Candidate TPOT ms | Total throughput gain |
| --- | --- | ---: | ---: | ---: |
| 1 | 512/32 | 3.841 | 3.819 | 0.69% |
| 4 | 2048/32 | 4.575 | 4.506 | 1.04% |
| 16 | 512/128 | 4.656 | 4.565 | 1.56% |
| 2 | 256/32 | 4.069 | 4.006 | 1.37% |
| 8 | 256/32 | 4.227 | 4.154 | 1.37% |
| 24 | 256/32 | 4.796 | 4.707 | 1.13% |
| 32 | 256/32 | 4.896 | 4.806 | 1.79% |

Eight GPU tests passed, including the new epilogue check with CUDA graph
replay, changed inputs, partial row/column tiles, and native BF16 reference.
Full-model teacher-forced verification checked 21,600 tokens per engine:
zero failures and worst logit deficit 1.375 for both. Local CPU tests:
eight passed, ten CUDA-only tests skipped.

These are local H100 improvements, not a new official leaderboard score.
Raw samples and corpus hashes are in `measurements.json`.
