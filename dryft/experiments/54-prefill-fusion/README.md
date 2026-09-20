# Prefill fusion trials and down-projection correctness correction

## Tried fusions

1. Reused existing residual/RMSNorm kernel in prefill after attention and optionally across the next-layer boundary. Six shapes, two seeds, five alternating timing trials. Both-prefill-boundary fusion produced small mixed total-generation changes (about -0.12% to +0.77% for multi-token outputs), with +2.2–2.4% on prefill-only output length one. The post-attention-only version also had a 9% B1 regression on one seed. All 6,978 output positions per mode passed the teacher-forced check. No prefill norm fusion was enabled.
2. Reused existing gate/up+SwiGLU kernel with prefill matrix tiles. Eight initial tile choices, six follow-up choices, actual layer inputs/weights, and the existing cuBLAS+SwiGLU baseline. The (128,256,64,8,4) tile won the isolated 2048/8192-row block tests by 7.5–9.6%, with identical tested BF16 outputs. Two larger configurations failed register allocation and were rejected. Full generation did not retain that gain: roughly -0.16% to +0.30% on completed large-prefill cases. No prefill MLP fusion was enabled.

## Reproduced correctness regression

The MLP fusion comparison stopped on a shared failure in both the submitted ae8739d baseline and the candidate. Both generated identical tokens. A fresh H100 run reproduced it at corpus seed 6907, B16/P512/N128, sequence 13, generated-token index 48. Token 653 was 2.5 logits below native token 1744 (judge allowance 2.0). Split-down alone and split-down plus next-norm fusion both failed; the original full-K down projection passed, with worst deficit 0.25. This is local evidence of a real correctness issue despite the official submitted run having passed its workloads.

## Retained local correction

Production fixed dispatch now uses the original down GEMM and a fused residual-add/next-RMSNorm kernel. The matrix multiplication's summation order stays unchanged. The fused elementwise kernel matches the separate norm's runtime divisor, epsilon and contraction settings, while preserving the BF16 residual and normalized-value rounding boundaries. Existing post-attention residual/RMSNorm callers retain their implementation. The split-down variants remain available only for explicit offline overrides; automatic runtime calibration remains disabled, and production defaults never select them.

This removes 35 normalization launches: the complete measured decode trace falls from 370 to 335 GPU events, including one copy event in both. It is 0.1–1.6% faster than the original full-K down path in paired generation, but about 1.2–2.5% slower than submitted split-down on batched cases. It is a correctness correction with a smaller retained fusion benefit, not a claimed improvement over the submitted score.

## Validation

Integrated three-way A/B covers five shapes and two seeds, with independent captured states and three alternating trials. All 6,976 replacement positions passed native own-prefix checks, and all paired replacement tokens exactly matched the original full-K down path. A final run using actual default dispatch, empty override tables and a calibration function that raises if called passed another 3,744 positions across five shapes, including the regression prompt. Total 10,720 candidate positions, no failures. This is evidence over those prompts, not a universal guarantee.

The initial research B1 comparison inherited verification-row overrides from a prior shape; the final integrated test sets overrides for all 1–32 row counts and fixes that comparison. Only the final integrated and unforced-dispatch results support the retained implementation.

Two GPU numerical graph-replay regression tests passed. Local unittest discovery passed (38 tests, 29 skipped without CUDA/dependencies), along with syntax and whitespace checks. The Dryft CLI validator escalation was blocked by automatic approval review over a potential-upload concern. The separately inspected pure-local agent/package.py checker and packager passed instead, checking the Engine entrypoint and a Python-source-only archive; no source upload was attempted by that fallback. No official result is claimed. Nothing has been committed or pushed by these experiments.
