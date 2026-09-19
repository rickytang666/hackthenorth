# Batch-aware BF16 projections

Baseline: `ea4f5ab`, the passing 788.8 tok/s official submission.

Reuse the GEMM shipped in Triton 3.1, with measured tiles and column-major
copies of selected BF16 weights. Dispatch by projection type and total input
rows (batch times query count), covering single-token decode and speculative
verification. Native cuBLAS handles larger inputs and prefill. No runtime
autotuning or weight precision changes.

The sweep used four distinct matrices per projection, exceeding H100 L2,
CUDA graph replay, and median-of-five timing. Intermediate row counts and
boundaries were tested before selecting ranges; individual matrix throughput
is not an official score.

Full-generation H100 comparison of the actual submitted source, alternating
engine order across five matched samples per case:

| Batch | Prompt/output | Baseline ms | Candidate ms | Throughput gain |
| --- | --- | ---: | ---: | ---: |
| 1 | 512/32 | 99.088 | 93.737 | 5.7% |
| 4 | 2048/32 | 324.430 | 306.274 | 5.9% |
| 16 | 512/128 | 834.804 | 763.823 | 9.3% |
| 2 | 256/32 | 170.592 | 155.395 | 9.8% |
| 8 | 256/32 | 194.144 | 176.842 | 9.8% |
| 24 | 256/32 | 272.578 | 263.870 | 3.3% |

Five GPU kernel tests passed, including projection row boundaries 1 through 64.
Teacher-forced replay checked 16,480 output tokens per engine against the
pinned native model: zero failures for both, worst logit deficit 0.375 for
baseline and 0.25 for candidate (limit 2.0). CPU suite: eight passed, seven
CUDA-only tests skipped locally.

Prompts are synthetic diagnostic text, not Dryft's hidden corpus. These
measurements cannot predict the official score or establish universal
correctness. Column-major copies increase memory use; they are created only
for selected decode shapes during warmup.

Raw measurements: `../projection-measurements.json`. The upstream kernel is
https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/ops/matmul.py.
