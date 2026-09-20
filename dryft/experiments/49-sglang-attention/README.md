# Reuse SGLang BF16 decode attention

The engine can select SGLang v0.4.6's grouped Triton decode attention during
warmup at batch sizes >= 4. The adapter replaces paged-cache indexing with our
contiguous cache addressing and a device-side position. Attribution, the
upstream source hash, and Apache-2.0 license are included in the kernel source.
It uses only the existing Torch/Triton runtime and BF16 weights.

Source: https://github.com/sgl-project/sglang/blob/v0.4.6/python/sglang/srt/layers/attention/triton_ops/decode_attention.py

The existing attention implementation remains the fallback. Batch-one decode,
prefill, and speculative verification retain their existing paths.

## H100 measurements

Same-container whole-generation A/B, three alternating trials per prompt,
with other projection choices frozen identically. The independent split-down
experiment was disabled for these measurements and is not part of this change.
Public shapes are diagnostic, not predictors of the hidden official score.

| Batch / prompt / output | Throughput change across tested prompts |
| --- | --- |
| 1 / 512 / 32 | -1.0% to -3.6%; candidate not enabled |
| 4 / 2048 / 32 | +0.7% to +1.1% |
| 8 / 1024 / 64 | +1.6% to +1.8% |
| 16 / 512 / 128 | +1.9% to +2.3% |
| 32 / 128 / 32 | -1.8%; retain fallback |

These are forced-choice measurements. Production calibration selected SGLang
with eight splits at B8 and four splits at B16; it selected the existing fused
prologue at B4 and B32. This verifies real attention selection with other
projection choices frozen, not a complete cold retuning or an official score.

The calibration correctness probe also needed a fix: graph timings mutate
live KV contents, making later candidates appear inconsistent with the saved
reference. The probe now snapshots one layer's KV and position for numerical
comparisons. Timed decode graphs continue to use the live cache.

## Validation

- 96 kernel graph checks passed against FP32 SDPA, including changed inputs,
  positions, partial tiles, and NaNs in unused cache slots.
- 9,152 candidate output positions and the same number of baseline positions
  passed native own-prefix verification. Maximum candidate deficit was 0.75
  (0.5 for batched paths), below the allowed 2.0.
- 3,712 more positions passed after actual production attention calibration,
  using fresh prompts. Maximum deficit: 0.25.
- The focused GPU regression test verifies calibration remains stable when
  live KV/position changes and that normal decode consumes current cache data.
- Local tests and `dryft validate engine` passed.

`measurements.json` preserves the completed structured kernel, generation,
integrated-dispatch, and calibration records. Full research scripts and logs
remain in the sibling `gpu-interpreter` workspace under experiment 49.
