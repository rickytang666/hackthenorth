# Batch-one attention output projection and residual fusion

Baseline: main `c9e6a04`. BF16 weights and all native BF16 rounding boundaries
are retained. This is the winner from the GPU-interpreter research loop;
none of the losing interpreter implementations is imported by the engine.

## Change

The B1 decoder's attention output projection previously wrote a BF16 tensor,
then a separate add read it and the residual to produce another BF16 tensor.
The new coalesced scalar projection evaluates two output rows per CTA and adds
the residual in its store epilogue. Its FP32 projection sum rounds to BF16
**before** residual addition. It reads the same projection weights, removes
one launch and a 2560-element BF16 write/read round trip per layer, and avoids
materializing the separate projection result. It does not quantize or change
KV-cache layout. Prefill, batched decode, and the four-query verifier retain
their existing branches.

The B1 chain-A calibration adapter uses the same new operation in its legacy
path, so its timing matches actual execution. Existing chain calibration and
fallback remain available. There is no new model-sized weight packing.

The four-real-layer H100 screen measured 11.76 us versus 16.66 us for the old
projection-plus-add pair (1.417x). The primary improvement is a better B1
projection implementation and one fewer launch; the removed activation traffic
is only 10 KiB per layer, not a reduction in model-weight traffic.

## Full-generation evidence

An initial process tested three B1 prompts with seven alternating matched
samples each: +3.73%, +5.24%, +5.44% (geometric mean +4.80%). Native
teacher-forced verification checked 192 candidate positions with no failures;
worst chosen-token logit deficit was 0.25.

A second process tested the integrated source against the exact baseline layer
source, with baseline projection calibration frozen identically for each mode.
Separate DecodeStates/CUDA graphs were warmed before timing. Seven samples per
mode alternate ordering. All output lengths and timing-repeat determinism passed.

| Prompt / output | Seed | Main tok/s | Fused tok/s | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 128 / 64 | 2101 | 262.67 | 277.21 | +5.54% |
| 512 / 64 | 2102 | 260.32 | 286.78 | +10.16% |
| 2048 / 64 | 2103 | 225.02 | 235.62 | +4.71% |

The second-process geometric mean gain is **6.78% on these B1 cases**. This is
not a whole-leaderboard estimate. Outputs can differ at near ties; speculative
acceptance can consequently change too. The first repeat case had identical
tokens, while the larger gain on another case includes a changed continuation.
Per-sample TTFT, TPOT and speculation counters are retained in the measurements.

Eight additional fresh B1 prompts, plus an integrated-source recalibration
smoke run, raised second-process candidate verification to **736 positions**.
There were **zero failed positions**, worst logit deficit **0.375** against
2.0. The smoke selected the legacy branch for both B1 chain choices, exercising
the new output projection. Two successive new prompts each at B4 and B16
produced **640 exactly equal output positions** across baseline/candidate,
checking the unchanged branches and cache reset.

The engine-comparison harness peaked at 21,422,246,400 allocated bytes while
retaining multiple mode/shape states; adding the independent native verifier
peaked at 29,686,486,016 bytes on an 85,017,624,576-byte GPU. These are harness
allocations, not official per-workload peak-memory measurements.

## Checks and reproduction

- New H100 test passes the explicit halfway-rounding/cancellation case and
  changed-input graph replays with poisoned output storage (1.94 seconds).
- Local unittest discovery: 32 tests, 6 passed and 26 dependency/GPU skips.
- Independent read-only integration review found no blocking issues.
- Integrated full-generation run completed in 105.57 seconds, excluding queue
  and container startup. Official-run correctness remains separately enforced.

Research harness: `agent/gpu_interpreter/integrated_ab.py` on the research
branch; run `modal run agent/gpu_interpreter/run.py --mode integrated_ab`.
`baseline_fused_layer.py` is the exact `c9e6a04` layer source. Detailed sample
arrays, calibration selections, correctness counts and memory figures are in
`measurements.json` and `initial-measurements.json`.
