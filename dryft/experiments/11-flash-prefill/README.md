# FlashAttention prefill with native grouped-query attention

Baseline: `c9db6b9`, the passing 807.7 tok/s official engine.

Use PyTorch 2.5.1's built-in FlashAttention backend with `enable_gqa=True`
for causal prefill. Preserve eight KV heads instead of materializing 32,
combine Q/K/V projections, and reuse the BF16-preserving QK norm/RoPE kernel.
An explicit attention mask retains the reference path. Decode is unchanged.
No new dependency, lower precision, or standalone FlashAttention build.

The preliminary H100 profiler confirmed
`aten::_scaled_dot_product_flash_attention`. The final source was compared
against a frozen copy of the baseline on an H100 using Torch 2.5.1/CUDA 12.4
and Transformers 4.51.3. Five matched samples per shape alternate engine order.
Prompts use fixed random windows of WikiText-2, CPython heapq source, and
Shakespeare, with plain continuation and chat instructions. Corpus URLs and
hashes are recorded in `measurements.json`.

| Batch | Prompt/output | Baseline ms | Flash ms | Throughput gain | TTFT ms | Candidate range/median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 512/32 | 146.081 | 137.120 | 6.5% | 22.42 → 13.95 | 8.8% |
| 4 | 2048/32 | 304.039 | 269.514 | 12.8% | 157.63 → 122.33 | 1.0% |
| 16 | 512/128 | 757.624 | 722.150 | 4.9% | 146.65 → 111.19 | 0.8% |
| 2 | 256/32 | 152.619 | 144.776 | 5.4% | 21.82 → 13.53 | 0.5% |
| 8 | 256/32 | 174.920 | 164.817 | 6.1% | 39.20 → 28.87 | 0.3% |
| 24 | 256/32 | 260.468 | 233.493 | 11.6% | 108.26 → 80.31 | 0.9% |

All six GPU tests passed, including reference attention comparison, unexpanded
KV cache contents, future-token isolation, and explicit-mask fallback.
Teacher-forced replay checked 16,480 tokens per engine: zero positions outside
the 2.0-logit margin, worst deficit 1.375 for both engines. Local CPU suite:
eight passed, eight CUDA-only tests skipped.

These are diagnostic workloads, not an official score or a universal
correctness guarantee. The change targets prefill; it does not reduce the
projection bottleneck during decode. The sample range above spans varied
prompts and is not Dryft's official stability calculation.

Upstream API and implementation:
https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/functional.py
