# BF16 fusion after removing FP8

Baseline: `0c03705`, which restores the engine from `ec476e9`. That BF16
engine scored 968.2197 on Dryft. Mixed FP8 submission `49e6569` failed
`incorrect_output` on B16/P512/N128 despite passing local probes, so none
of the FP8 fusion work is included here.

## Producer RMS statistics: rejected

The down/residual projection already holds the rounded BF16 residual in
registers. The candidate reduces its squared values per output tile into
80 FP32 partial sums per row. The next QKV tile consumes those sums and
normalizes its inputs while multiplying, eliminating the normalized
activation tensor and its separate RMSNorm launch. BF16 rounding boundaries
are retained.

Eight real consecutive layer boundaries rotate 650,117,120 weight bytes
through CUDA graphs. Five alternating timing samples per configuration;
three changed input scales per numerical comparison. H100, PyTorch 2.5.1,
Triton 3.1. Remote experiment took 21.66 seconds.

| Batch | Baseline pair | Best candidate pair | Change in latency |
| --- | ---: | ---: | ---: |
| 4 | 38.39 us | 48.07 us | +25.2% |
| 16 | 39.93 us | 49.34 us | +23.5% |
| 32 | 44.53 us | 48.12 us | +8.1% |

B4/B16 outputs were bit-identical. B32's fastest configuration had relative
L2 error 0.000205 and maximum absolute difference 0.0078125. These are
kernel comparisons, not a model-level accuracy verdict. No candidate spilled
registers. Fusion still repeats normalization arithmetic within every QKV
output tile and changes the GEMM pipeline; fewer intermediate bytes alone
do not establish a speed win. We have not separately attributed all of the
regression to producer versus consumer overhead.

The best B16 variant replaces an 81,920-byte normalized tensor with 5,120
bytes of partial sums, but rereads statistics in each QKV tile. These are
logical tensor sizes, not measured HBM traffic. Model weight traffic remains
unchanged. This candidate stays outside `engine/`.

## Single-partition attention direct stores: validated

Existing attention writes normalized FP32 partial results and FP32 logsumexp,
then launches a merge that reads both and stores BF16 output. With one
partition the merge weight is exactly one. The retained change writes BF16
output directly from the attention accumulator, omits logsumexp storage,
and removes the merge kernel. Both regular grouped attention and the fused
Q/K norm + RoPE + cache write attention path use this shortcut.

Partition sizes and the multi-partition path remain unchanged. At B16 with
one query this eliminates 528,384 logical bytes per layer (FP32 partial
write/read plus logsumexp write/read), or 18.14 MiB over 36 layers, and 36
kernel launches per decode step. It does not eliminate weight reads or
change KV storage precision.

Short-context attention microbenchmarks were bit-identical and improved
from 19.58 to 17.73 us at B16/capacity319 and 26.73 to 24.47 us at
B32/capacity319. The test took 13.36 seconds. Forcing capacity639 into one
larger partition lost 17.5%; capacity1151 also regressed. Those partition
changes were rejected.

This shortcut applies only when the existing policy already produces one
partition. The three public workload capacities all use multiple partitions,
so this is not an expected improvement on those shapes. Hidden workload
shapes are unknown; no leaderboard gain is claimed.

## Reproduction

Run `modal run agent/stat_fusion/run.py --mode micro` for projection statistics,
`--mode attention_micro` for attention partition screening, or `--mode ab`
for full generation and correctness. `attention_baseline.py` is the frozen
pre-change BF16 attention implementation used for paired comparisons.
Raw results and focused measurements accompany this note.

## Full-generation validation

The follow-up H100 run took 47.30 seconds. Five alternating full-generation
wall-clock timings per mode/shape, with existing projection choices fixed
and identical in both modes. This is a short screening experiment, not a
fresh-process official benchmark or a prediction of the hidden score.

| Batch / prompt / output | Throughput change | Tokens checked | Worst logit deficit |
| --- | ---: | ---: | ---: |
| 16 / 256 / 64 | +1.47% | 1,024 | 0.25 |
| 32 / 256 / 64 | +0.99% | 2,048 | 0.50 |
| 16 / 512 / 128 | +0.23% (unchanged kernel path) | 2,048 | 0.375 |
| 4 / 2048 / 32 | -0.06% (unchanged kernel path) | 128 | 0.0 |

All 5,248 generated tokens were identical between modes; no teacher-forced
position exceeded the 2.0-logit limit. The small changes on the unchanged
multi-partition paths are timing variation, not credited gains.

Six grouped-attention cases also passed bit-exact CUDA graph replay checks
with changed Q/K/V inputs and positions (including position zero), query
counts 1 and 4, batch sizes 1/4/16/32, and capacities 31/63/319/511/639.
The fused prologue path was separately bit-exact in the attention microtests
and all four generation comparisons.
