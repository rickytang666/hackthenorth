# Prefill backend calibration and larger CUDA graphs

Adopt the production changes from worktree commit
`a1c2125cb4c613f44aa9cfcbce5a0f3bf25897d2` onto main `e134e74`.
The source worktree is read-only and its independent work is untouched.

During untimed warmup, time causal BF16 grouped SDPA with FlashAttention and
cuDNN on the real tensors; cache the fastest supported backend. Keep Flash
as the fallback. The adoption extends the selection key with device, dtype,
and Q/K/V strides, so different layouts cannot reuse an incompatible choice.

Raise the prefill graph cutoff from 2,048 to 16,384 total prompt tokens,
covering both 8,192-token public shapes. Every replay copies the new prompt,
overwrites prompt KV entries, and resets decode position; no prefix is reused.

Validation covers changed packed attention inputs under graph replay against
native math attention, changed prompts and cache poisoning for an 8,192-token
prefill, and eager fallback above the new cutoff. Full H100 A/B uses identical
diverse prompts, alternating five paired trials per shape, token delivery to
a separate reader process, and full teacher-forced replay of generated tokens.
Projection calibration stays independent per engine and workload. Timings are
local end-to-end measurements rather than official Dryft scores.

## H100 results

All 29 GPU tests passed. Five alternating paired trials per shape.

| Batch | Prompt/output | Total throughput change | Baseline TTFT ms | Candidate TTFT ms |
|---:|---:|---:|---:|---:|
| 1 | 512/32 | +0.88% | 9.62 | 9.60 |
| 4 | 2048/32 | +1.93% | 117.06 | 112.17 |
| 16 | 512/128 | +0.22% | 107.80 | 106.84 |
| 2 | 256/32 | +1.34% | 9.92 | 9.82 |
| 8 | 256/32 | -0.42% | 27.04 | 27.30 |
| 24 | 256/32 | +0.49% | 77.19 | 76.72 |
| 32 | 256/32 | -0.12% | 104.10 | 103.79 |
| 1 | 128/1 | -0.67% | 5.60 | 5.64 |

Teacher-forced replay checked 21,605 tokens per engine: zero failing positions. Worst logit deficit was 0.75 for baseline and 1.375 for candidate, below the 2.0 margin. cuDNN was selected for B4/2048; see measurements for all backend choices. The observed small negative changes are retained above; this is not an across-the-board speedup and public gains do not predict official score.
