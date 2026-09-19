# Tiled prefill Q/K normalization and rotary embeddings

Port of `engine/kernels/fused.py` from worktree commit
`65b2ef9ceeaa0bc616f2007e4a645d17d4f849aa`. The initial read-only snapshot of
the uncommitted file was verified byte-for-byte against that final commit.
The Claude worktree and its validation jobs were not modified.

Baseline: our main commit `1963840`, which already includes residual/RMSNorm
fusion, graph-based projection calibration, and the corrected native-linear
weight-layout fallback. This experiment isolates the new prefill kernel's
additional effect; the worktree's reported combined gains are not reused as
measurements of this isolated change.

The new kernel launches one program per batch/token and processes its 32 Q
heads in two groups of 16 and its eight K heads in one group. Cosine/sine
tables and norm gains are reused across heads. BF16 rounding boundaries are
preserved; the FP32 reduction order may differ slightly. The old row kernel
remains available through `tile=False`, and decode-cache writes retain their
existing row kernel. A host-side shape guard falls back to the row kernel
when head counts cannot be covered safely by the tile's unmasked loads.
Qwen's fixed 32/8-head shapes always use the tiled path.

The initial GPU suite found an existing test compared the unchanged decode
cache kernel bit-exactly to `norm_rope`'s default, which is now tiled prefill.
Three values out of 671,744 differed, with maximum absolute difference
0.00390625. The cache test now explicitly selects the original row reference
and retains zero tolerance. Separate checks compare tiled prefill to the row
kernel and native normalization/RoPE, including packed strides, changed CUDA
graph inputs, zero inputs, and unsupported-head-count fallback. The initial
failure is retained in `initial-tests.json` and `initial-tests.log`.

The full A/B uses identical diverse prompts and alternating paired runs with
JSON token delivery through a pipe to a separate process. Both engines run
their own calibration implementation and reset choices for each workload;
module routing prevents one engine's lazy calibration import from selecting
the other engine's module. Teacher-forced replay checks each engine's own
generated prefixes against the native BF16 reference. Local timings are not
the official Dryft score.

## Final H100 validation

All 28 GPU tests passed. Five alternating paired trials per shape:

| Batch | Prompt/output | Total throughput change | Baseline TTFT (ms) | Candidate TTFT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 512/32 | +0.46% | 9.78 | 9.56 |
| 4 | 2048/32 | +1.28% | 123.06 | 118.27 |
| 16 | 512/128 | +0.50% | 111.40 | 106.95 |
| 2 | 256/32 | +0.18% | 9.41 | 9.20 |
| 8 | 256/32 | +0.26% | 27.95 | 27.30 |
| 24 | 256/32 | +1.27% | 79.33 | 76.96 |
| 32 | 256/32 | +1.82% | 108.96 | 104.21 |
| 1 | 128/1 | -0.37% | 5.48 | 5.50 |

Teacher-forced verification checked 21,605 tokens per engine with zero failed positions. Worst logit deficit: baseline 1.375, candidate 0.75, against the allowed 2.0 margin. Outputs are not universally bit-identical. The single-output edge was 0.37% slower (about 0.02 ms). These are local measurements, not an official score.
