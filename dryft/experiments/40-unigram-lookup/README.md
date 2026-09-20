# One-token history fallback

Baseline: `15fb7d3`, official score 990.7545 tok/s. Add a one-token
history match after the existing four-, three-, and two-token matches.
The proposal still needs seven previously seen continuation tokens, and
every emitted token still comes from target verification. Request history
is rebuilt per generation; no extra model, graph, or cache is introduced.

A paired H100 screen on six fresh general prompts (P128/512/2048,
N32/64, batch one) measured **1.02489x** geometric-mean throughput.
Three alternating trials per variant used the same weights and frozen
projection choices. The weakest case was 0.9845x. A copying diagnostic
is reported separately and excluded from the general mean.

All 384 candidate positions, including the copying diagnostic, passed
native teacher-forced checking with worst logit deficit 0.0. Total remote
harness time was 106.3 seconds including setup and checks. This is a small
local screen, not evidence of an equivalent official-score increase.
Batched generation and GPU kernels are unchanged.

Alternatives: partial-history lookup with variable verifier widths measured
1.03380x; keeping one verifier with partial history measured 1.03172x.
The selected candidate isolates the ngram fallback without adding graph
capture or changing partial-history semantics. Raw research harness and
results remain in the gpu-interpreter worktree under
`agent/gpu_interpreter/unigram_validation.py` and
`experiments/37-gpu-interpreter/unigram_validation.json`.

Local checks: 36 unittest cases, 10 pass and 26 skip because CUDA/Torch
are unavailable locally. Tests include exhaustive incremental history
matching at widths three and seven and prompt isolation.
