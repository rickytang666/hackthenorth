# Down projection / next-layer RMSNorm fusion

The engine now offers `down_next_norm` as a warmup-calibrated candidate. It merges the split-down FP32 partials, rounds to BF16, adds and rounds the residual, and normalizes for the next layer in the same merge kernel. Both the residual and normalized values are preserved. The next layer consumes the normalized tensor once, so its separate input RMSNorm launch disappears. The last layer uses the existing path. This also works for multi-token speculative verification. Prefill keeps its existing implementation.

The fusion reuses the local split-down experiment, which was uncommitted before this change. All weights and intermediate storage remain BF16 except the existing FP32 accumulations/statistics. The normalization divisor/epsilon and compiler contraction settings match the original RMSNorm kernel; the earlier prototype did not match those settings and is retained only as an initial experiment.

## Validation

- Integrated three-way H100 comparison: current baseline, split-down without norm fusion, split-down with norm fusion. Four shapes, two fresh prompt seeds per shape, three alternating timed samples per mode. Each mode has its own captured state. Projection choices are frozen and attention is calibrated identically for the paired tests.
- All 7,424 integrated candidate token positions passed native own-prefix validation. All candidate tokens were identical to the split-down comparison outputs. Kernel-level tests compare both outputs bit-for-bit against separate split-down and RMSNorm on replay with changed inputs, residuals, and gains, including verification row counts.
- Cold production calibration, with no frozen choices, selected fusion at batch 16 and verification row count 8. Fresh generations at batch 16 and batch 1 passed another 1,056 native positions. Shape warmups took 64.4 and 31.2 seconds. The batch-one shape reused the loaded model after the first shape, but its row-specific projection choices were not preselected.
- Complete CUDA traces: 370 GPU events per decode step for baseline and split-down, 335 for fusion. These include one copy event; the difference is 35 eliminated RMSNorm kernel launches. Assertions verify that all 36 layer operations are present in the trace. The initial incomplete trace's 328 count is invalid.
- Focused GPU regression test passed. Local unittest discovery passed (37 tests, 28 skipped without CUDA/dependencies). Dryft package validation and git diff whitespace checks passed.

## Performance and scope

Integrated full-generation speedups versus the current baseline range from 1.020x to 1.036x. Against split-down alone, fusion contributes 1.002x to 1.018x. These are local paired results, not evidence of an official score increase. Actual startup calibration can retain the old path when it wins.

The public Dryft upload/run was blocked by automatic approval review because it requires explicit authorization to upload engine source and launch the public benchmark. Nothing was uploaded, no public run was started, and no changes were committed or pushed.
