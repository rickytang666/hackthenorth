# Reuse projection weights across verified output tokens

Baseline: deployed `e0e7869`, BF16. Its latest official score was 964.8 tok/s;
the 1200 tok/s objective requires about 24.4% greater ranked throughput.
The public shapes do not rank, so local improvements cannot establish that
the score target has been achieved.

The current B1 verifier processes the current token plus three proposed
tokens in a single target-model forward. A correct prefix of the proposals
and one correction/bonus token can be emitted. Projection tiles share weights
across these four input rows. The current production draft only proposes when
an exact four-token suffix appeared earlier in this generation's history.

Research policies, using exactly the existing model and verifier:

- Shorter prompt-history matches (three or two tokens), plus longest-match
  backoff from four to two. Increased coverage is useful only if accepted
  tokens pay for the verification pass and host work.
- A rolling Jacobi window that reuses unaccepted target predictions as the
  next draft. These predictions are never emitted without fresh verification.
  The target model supplies the draft during work already done by verification;
  there is no additional draft-model forward or external trained weight.

The latter is a small rolling Jacobi experiment, not a full implementation of
Lookahead Decoding's two-dimensional trajectory pool and parallel branches.
The original authors explicitly describe weak acceptance as a limitation of
plain Jacobi decoding. Both acceptance and complete generation time must win.

References:

- https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/n_gram/
- https://www.lmsys.org/blog/2023-11-21-lookahead-decoding/
- https://github.com/hao-ai-lab/LookaheadDecoding

The short screen uses six B1 P512/N64 prompts across three corpus domains and
instruction/continuation styles, alternating policies on the same calibrated
engine, and native own-prefix correctness replay. No production behavior is
changed by this screen. B4/B16 already share weights across all their decode
rows; increasing matrix row tiles does not create extra token-level reuse.

## First measured screen

The first process spent 128 seconds on baseline calibration and stopped before
any policy timings. Its empty correctness totals are not validation. The rerun
reused those baseline kernel selections for every policy and completed in
44.28 seconds, including reference checks.

The four-query target graph cost 3.92 ms versus 3.54 ms for a single decode
step in the first process. Host transfers/lookup are excluded from that ratio.
Full-generation geometric speedups across six B1 P512/N64 prompts:

| Policy | Speedup vs four-token history match |
| --- | ---: |
| Three-token match | 1.0194x |
| Two-token match | 1.0711x |
| Longest match, four down to two | 1.0774x |
| Rolling Jacobi | 0.8912x |

Backoff improved each of the six measured prompts (0.3–16.7%). Each policy
had 384 own-prefix positions checked; all passed. Backoff's worst deficit was
0.125. Across duplicate policies there were 768 unique verified positions.
Rolling Jacobi mostly emitted only one token per verifier call, explaining
its regression. It remains research-only. Records: lookup-measurements.json.

## Fresh-shape validation and adopted change

A second H100 process used six previously untested prompts, three alternating
trials per policy, and the actual `BackoffPromptLookup` class now selected by
`engine.py`. Kernel choices were frozen identically to the preceding baseline
calibration, so this compares draft policies, not different GPU tuning choices.

| Prompt / output | Seed | Backoff speedup | Adaptive speedup |
| --- | ---: | ---: | ---: |
| 128 / 64 | 3400 | 1.0434x | 1.0440x |
| 128 / 64 | 3401 | 0.9955x | 0.9940x |
| 512 / 32 | 3402 | 1.1947x | 1.1898x |
| 512 / 32 | 3403 | 1.1470x | 1.1406x |
| 2048 / 64 | 3404 | 1.1869x | 1.1841x |
| 2048 / 64 | 3405 | 1.0187x | 1.0168x |
| Geometric mean | | **1.0947x** | **1.0920x** |

Two separately labelled copying diagnostics were excluded from that mean.
Adaptive verification improved those by 1.5806x and 1.4038x versus main;
the corresponding simple-backoff gains were 1.1240x and 0.9948x. All policies
passed 448 candidate positions in this second process, worst deficit 0.0.
It finished in 54.19 seconds, excluding container startup/queue time.

The adopted code is only longest-match backoff from four to two history tokens.
The exact three-token draft verifier, KV handling, graph capture, weight format,
and B>1 generation paths are unchanged. Prompt lookup state is constructed
afresh for each generation. No adaptive graph or rolling Jacobi code is imported
by the submitted engine.

Across the two H100 processes, the adopted policy passed **832 teacher-forced
positions**, including the copying diagnostics, with worst deficit 0.125.
The local suite passes nine tests, with 26 dependency/GPU skips. New CPU tests
check longest-match priority, fresh-prompt isolation, and 200 incremental lookup
steps against a naive history search. This is measured B1 improvement, not an
established overall leaderboard improvement. See validation-measurements.json.

## HyperQwen inspiration

Read-only reference checkout: syv-ai/HyperQwen at
`feaffb676ba0ac6ba5060bd2edbd39cce952829e`. Its adaptive long-block policy is
relevant: two fully accepted short blocks plus a long lookup continuation
justify expanding verification, with request-local state and fallback on
rejection. Its trained drafter, model architecture, quantized runtime and
headline 27B/3090 performance are not a drop-in comparison for this engine.
The new research helper independently implements this policy with existing
four-/eight-query target graphs; it does not include a trained draft model.

Reference: https://github.com/syv-ai/HyperQwen/blob/feaffb676ba0ac6ba5060bd2edbd39cce952829e/patches/dflash2-lookup-drafting.patch
