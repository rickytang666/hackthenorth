# Speculative draft research, 2026-09-19

Goal: +20% official throughput over commit `eea1b37` (895.3 tok/s lineage) by
amortizing the ~8.0 GB BF16 weight read per decode step across multiple
tokens. Prompt lookup only fires when output copies the prompt and only at
batch 1; a model-based draft fires every cycle at every batch. The contract
allows any draft because emitted tokens always come from the exact BF16
verifier ("a speculative decoder with exact verification passes by
construction"). All runs on Modal H100, PyTorch 2.5.1+cu124, Triton 3.1,
via `agent/baseten/modal_profile.py`; raw JSON in `experiments/modal-*.json`.

## Run 1 - `fp8_spec`: acceptance is excellent, torch FP8 GEMM is not

`profile_fp8_spec.py`. Teacher-forced agreement of an FP8
quantize-dequantize proxy (rowwise weight scales, per-row activation
quantization modeled) against native BF16 greedy on 2,048 generated
positions from diverse corpus prompts (batch 1/4/8, prompts 512-2048):

| Draft | Agreement | Tokens/cycle k=3 | k=4 |
| --- | ---: | ---: | ---: |
| FP8 all projections | 94.9% | 3.65 | 4.40 |
| FP8, BF16 lm_head | 96.0% | 3.70 | 4.52 |

Speed: `torch._scaled_mm` rowwise with per-call activation quantization
projected 12.5 ms/step of projections vs 3.7 ms BF16 - 3.3x slower at
decode row counts. The cutlass rowwise path is built for large M.

## Run 2 - `w8a16`: Triton weight-only FP8 GEMV falls short; cuBLAS FP8 shines on big shapes

`bench_w8a16_gemv.py` v2. Full-K vector kernel (production `_norm_projection`
shape) for rows 1-4, pipelined `tl.dot` kernel for rows 8-32, plus pure
`_scaled_mm` tensorwise with pre-quantized activations:

- Vector kernel rows=1: gate_up 1.13x, lm_head 1.11x, down 1.01x, qkv 0.87x.
  Multi-row loop re-reads the weight tile: rows=2 doubles time. 0.8-1.6 TB/s.
- Dot kernel rows 16: 0.7-1.4 TB/s (BF16 cuBLAS reaches 2.8 TB/s on lm_head).
- `_scaled_mm` (GEMM only, quantization excluded): lm_head 1.96x, gate_up
  1.67x at rows 8-32 - near the FP8 bandwidth bound - but *slower* than BF16
  on qkv/o/down, which sit at a latency floor.

Best-of-per-shape composite draft step: ~1.2-1.3x faster than BF16.
The economics need ~1.7x.

## Run 3 - `fp8_dot`: native FP8 tensor cores in Triton 3.1 are worse

`bench_fp8_dot.py`. Host-transposed (K, N) FP8 weights, `tl.dot(x_fp8,
w_fp8)` with no in-kernel transpose or conversion, stages 3-4, split-K.
Uniformly 0.3-0.67x vs BF16 cuBLAS, <=1.0 TB/s at every shape and row
count. Triton 3.1's FP8 dot path does not approach bandwidth on skinny
GEMMs; Marlin-class shuffled-layout kernels would be needed, which is out
of scope for source-only Triton within hackathon time.

Conclusion so far: FP8 draft *accuracy* is proven (94.9-96.0%), FP8 draft
*speed* is not achievable on this stack beyond ~1.3x composite.

## Run 4 - `layerskip`: agreement of a truncated-depth draft

`profile_layerskip.py`. Draft = first L layers + final norm + lm_head,
running on the existing production kernels at exactly (L/36) of projection
cost plus lm_head - no new kernels, no new numerics. Also FP8 combos at
L=24/18. Reference generations cached on the volume
(`fp8-spec-generation-records-v1.json`).

Teacher-forced agreement on the same 2,048 positions:

| Draft | Agreement | Tokens/cycle k=3 |
| --- | ---: | ---: |
| layers 30/36 | 38.1% | 1.57 |
| layers 24/36 | 14.3% | 1.15 |
| layers 18/36 | 1.5% | 1.02 |
| layers 12/36 | 0.2% | 1.00 |
| FP8 + 24 layers | 14.1% | 1.15 |
| FP8 + 18 layers | 1.6% | 1.02 |

Unadapted early exit is useless on this model: even at 83% depth the draft
disagrees on five of eight positions.

## Verdict: no viable draft on this stack

The cycle model, batch 1, k=3: `ms_per_token = (3*draft_step + verify_step)
/ tokens_per_cycle` against 3.78 ms baseline. A draft must cost <=~0.6x of
a BF16 step at high acceptance to clear +20%.

- FP8 draft: 94.9% agreement, but the fastest measured composite
  (BF16 qkv/o + cuBLAS FP8 gate_up/down/lm_head) is ~0.85x - cycle math
  gives at most +3%.
- Layer-skip draft: guaranteed 0.5-0.67x cost, but 1.5-14% agreement.
- Trained drafts (EAGLE/Medusa-class) would fix agreement at low cost but
  shipping weights is prohibited.

Speculation beyond the existing prompt lookup is uneconomical in this
runtime. Reallocate to direct decode-step work, in expected-value order:

## Run 5 - `small_gemv`: per-shape GEMV is near its ceiling; the enemy is per-kernel latency

`bench_small_gemv.py`, after rebasing onto `9c40eb7` (924.2 tok/s official,
fused greedy head + b1 down/residual landed). Six vector variants (including
the production pattern and the new down pattern, split-K 2/4, pipelined
chunked-K) at rows 1-4 and three pipelined dot tiles at rows 8-32, on
qkv/o/gate_up/down vs cuBLAS:

- Adoptable but small: qkv dot-tile 1.05-1.09x at rows 8-16, o vector
  1.13x at rows 2, gate_up dot-tile 1.03-1.05x at rows 8-16, down vector
  1.07x at rows 1 (production already has this). Projected step saving
  only 70-125 us (~2-3%).
- The telling pattern: every best time sits ~5-9 us above its
  bytes/3.0-TB/s ideal (qkv 15.5 vs 10.5, o 14.1 vs 7, gate_up 39.8 vs 33,
  down 25.7 vs 16.6). That fixed per-kernel cost times ~145 projection
  kernels per step accounts for ~0.9-1.3 ms of the 3.71 ms batch-1 step -
  the same order as the whole remaining gap to roofline.

Conclusion: bandwidth tuning per kernel is exhausted; the remaining large
lever looked like reducing the NUMBER of kernels per step. Runs 6-9 tested
that hypothesis directly.

## Runs 6-9 - `chain`/`splitk`: the persistent-kernel hypothesis, tested and mostly rejected

`bench_persistent_chain.py` fuses o+residual+RMSNorm+gate_up+SwiGLU into ONE
kernel, crossing the full-row reduction with a release/acquire atomic
barrier (volatile-load spin, double-buffered self-cleaning counters,
single-wave grid sized from the compiled kernel's register usage so the
spin cannot deadlock). `bench_splitk_gemv.py` isolates split-K with
pre-zeroed atomic outputs.

- The barrier mechanism is EXACT: max_diff 0.0 vs the production chain.
- Fusion alone: 1.019x by interleaved same-process A/B medians
  (55.95 vs 54.90 us) - ~38 us/step, about +1%. The ~5 us per-kernel cost
  is a DRAM ramp/dependency stall, not launch overhead, and fusing the
  kernels does not remove the stall.
- Cross-barrier L2 weight prefetch (warming phase-B tiles during the wait)
  made it WORSE (56.9-59.7 us): the prefetch traffic costs more than the
  warm start saves.
- Split-K with all overheads excluded: o stays at exactly cuBLAS speed
  (13.8 us, 1.5 TB/s) even at 5,120 CTAs - parallelism is not the limiter.
- Methodology warning: baseline drifted 55.4 -> 78.2 us between Modal
  containers. Only interleaved same-process A/B numbers are comparable;
  one run's cross-container "1.36x" was clock drift, not a speedup.
- Triton 3.1 lacks BF16 atomic_add (compile error), so split-K outputs
  must stay FP32 with the cast fused into the consumer.

## Run 10 - `proj_ab`: kernel rankings flip between H100 containers

`bench_projection_ab.py`, production Projection vs cuBLAS vs pipelined dot
tiles, same process. In this container the production upstream-matmul tiles
ran 1.6-1.8x SLOWER than plain cuBLAS for qkv and o at every row count
2-32 (31-34 us vs 14.6-16.9 us), while an earlier container showed the
dot tile BEATING cuBLAS on the same shapes it now loses. gate_up was a
wash (production 0.995-1.009x vs the best candidate). Conclusion: these
GEMV rankings are hardware-instance-dependent; any fixed selection tuned
off-platform can regress on the judge's GPU, in either direction.

Response (engine change, this worktree): `projections.py` now autotunes at
load time. The first decode-shaped call per (kind, rows) measures every
candidate - cuBLAS, the frozen upstream tile, dot tiles in both layouts -
rotating real sibling-layer weights so the measurement streams from HBM,
verifies each against native linear, and caches the winner. It runs during
the untimed warmup before CUDA graph capture and costs milliseconds.
`kernels/dotgemv.py` adds the pipelined tensor-core candidate with masked
rows (no padding copies). Validated by `profile_autotune_ab.py` via
`agent/benchmark_gpu.py` fresh-process A/B with teacher-forced checks.

## Runs 11-13 - `autotune_ab`: only in-graph timing ranks kernels truthfully

Three calibration designs for load-time projection selection, each
validated by fresh-process full-generation A/B (`profile_autotune_ab.py`
over `agent/benchmark_gpu.py`, teacher-forced checks all clean):

1. Isolated microbenchmark autotune (DRAM-rotating, sibling weights):
   picked kernels that win standalone and LOST 0.3-5.6% end to end.
2. Eager whole-step timing: launch-bound, misranked again, lost 2.6-5.3%.
3. Throwaway CUDA-graph replay timing per candidate (coordinate descent,
   shared memory pool, 2% hysteresis toward the frozen config): calibration
   now selects the frozen tiles at every observed (kind, rows) - they
   genuinely are in-graph optimal on these containers (e.g. qkv rows-4 tile
   3.967 ms/step vs best dot 3.981) - and the A/B lands at parity
   (+0.5%, -0.3%, +0.0%).

Conclusions: the team's frozen projection selection is confirmed optimal
in-graph on Modal H100s; the isolated "adoptable +2-3% GEMV wins" from run
5 do not survive in-graph measurement and are withdrawn. The graph-replay
calibration harness (engine/projections.py `calibrate`, engine/decode.py
hooks, engine/kernels/dotgemv.py) is worth keeping only as insurance
against judge-hardware ranking flips (run 10 proved they happen): expected
gain ~0, worst case parity, ~5-8 s of the untimed load budget.

## Where this leaves +20%

Every large single lever measured on this stack is now quantified:
speculation ~+3% max (draft too slow), per-shape GEMV tuning ~+2-3%,
fusion/persistent ~+1%, prefetch negative. The 1100 tok/s leaderboard
scores are most plausibly an accumulation of small wins plus strength on
the batched hidden workloads. Adoptable now from these runs: qkv and
gate_up pipelined dot tiles at rows 8-16 (+1-1.5% batched), o rows-2
vector config, the +1% barrier fusion if its complexity is acceptable.
Decode attention at batch>=4 already uses split-KV partial/merge kernels
at ~2.2 TB/s of KV reads; pushing toward 3.0 TB/s is worth ~200-250 us on
the batch-16 step (up to ~+5%) and is the largest single remaining item.

1. **Small-shape GEMV bandwidth.** Isolated BF16 effective bandwidth:
   qkv 1.79, o 1.49, down 1.9, gate_up 2.5, lm_head 3.0 TB/s. Production
   projections average 2.41 TB/s; closing qkv/o/down toward 2.8-2.9 TB/s
   is worth ~500-700 us of the 3,776 us batch-1 step (+12-18% decode).
2. **Decode attention at batch >= 4.** 815-847 us at batch 4/16 is
   ~1.6 TB/s of KV reads; a split-KV flash-decode kernel toward 2.5 TB/s
   saves ~300 us (+5-6% on batched workloads).
3. **Fused lm_head argmax** (FlashSampling-style, greedy only) and
   residual-add fusion into o/down epilogues: +2-4% combined.
