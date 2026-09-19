# Fused greedy vocabulary projection and batch-one down/residual

Baseline: SwiGLU fusion commit `eea1b37`; its official run was still measuring
when this experiment finished.

Two changes:

- Evaluate every vocabulary row in a tiled BF16 GEMM, round its FP32 sum to
  BF16, and keep only each tile's maximum and first matching vocabulary id.
  A second reduction selects the global greedy token. This preserves
  lowest-id tie breaking and avoids materializing the full logit tensor.
  Rows 1–16 use a 16x64x128 tile; rows 17–32 use 16x256x64. Larger batches
  retain native linear plus argmax. The original tied weight layout is kept.
- For B1 MLP down projection, use a vector reduction over 1,024-element K
  chunks and four output channels per program. Sum FP32 partials, cast the
  projection to BF16, then add the residual before storing BF16. Larger
  batches retain the existing down projection. This avoids tensor-core
  padding for the single-row operation and fuses the residual addition.

The fused head design follows the local-winner/global-winner approach
presented in [FMMS/FlashSampling](https://github.com/FlashSampling/FlashSampling),
with greedy selection and explicit native BF16 rounding instead of sampling.
The implementation and numbers here are specific to our pinned runtime.

Real-weight H100 microbenchmarks used layers 0, 11, 23, and 35 for down
projection, rotating beyond L2 capacity. Its selected variant improved
projection-plus-residual time by 7.6%. The selected head configurations
improved linear-plus-argmax time by about 7–10% at batches 1, 4, 16, and 32.
All measured head outputs matched native greedy ids, including all-equal
logits. Down results passed atol=0.04/rtol=0.02 against native BF16.

Full-generation timings, five alternating matched A/B samples per shape:

| Batch | Prompt/output | Baseline TPOT ms | Candidate TPOT ms | Total throughput gain |
| --- | --- | ---: | ---: | ---: |
| 1 | 512/32 | 3.836 | 3.739 | 3.01% |
| 4 | 2048/32 | 4.524 | 4.492 | 0.34% |
| 16 | 512/128 | 4.571 | 4.536 | 0.77% |
| 2 | 256/32 | 3.999 | 3.984 | 0.86% |
| 8 | 256/32 | 4.148 | 4.125 | 0.36% |
| 24 | 256/32 | 4.717 | 4.682 | 0.07% |
| 32 | 256/32 | 4.811 | 4.775 | 1.31% |

Teacher-forced verification checked 21,600 tokens per engine with zero
failures; worst logit deficit was 1.375 for both. Ten GPU kernel tests passed
in that run. A final integration adjustment preserves the original forward
path for CPU/FP32/toy-model reference tests; it leaves the measured Qwen BF16
CUDA path unchanged. All 20 tests passed on the final source on H100, including cache reset,
speculative rollback, stream lengths, graph replay, native rounding, and
greedy ties. Local tests: eight passed, twelve CUDA-only skips.

This is a modest measured improvement, not a demonstrated 10% overall gain
or a new official leaderboard score. Raw samples are in `measurements.json`.
