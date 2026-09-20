# F01/M01 adaptation experiment — 2026-09-19

Both training arms improved over ordinary frozen Parakeet on the new development
and held-out F01/M01 data while retaining greedy decoding. Selective unfreezing
had better held-out WER than adapter-only, but did not win the development
comparison: F01 tied, while adapter-only did better on M01 and M04 development.
Do not treat one small holdout run as proof of a universal architecture winner.
F01 overall WER remains above the requested 20–30% range.

[Completed Baseten team 34 job](https://app.baseten.co/training/qel68p3/logs/wlvv40q).
One H100; two 600-step training arms. Early stopping did not trigger before the
ceiling. Adapter-only selected step 600; selective unfreezing selected step 550.
Full checkpoints remain at `results/adapter/best.nemo` and
`results/partial/best.nemo` in this job's checkpoint storage; neither is deployed.
The old adapter was not used because it had seen the new holdout recordings.


Both arms use fresh dim-32 adapters, balanced speaker/group sampling and greedy decoding. Checkpoint selection uses F01 development WER first, then M01, with other-speaker regression limits.

## dev

| Speaker | Clips / ref words | Frozen | Adapter-only | Adapter + last two blocks |
|---|---:|---:|---:|---:|
| F01 | 18 / 44 | 72.73% | 45.45% | 45.45% |
| F04 | 244 / 637 | 15.23% | 9.58% | 9.58% |
| M01 | 51 / 130 | 64.62% | 43.08% | 46.15% |
| M04 | 51 / 122 | 76.23% | 59.02% | 59.84% |

## test

| Speaker | Clips / ref words | Frozen | Adapter-only | Adapter + last two blocks |
|---|---:|---:|---:|---:|
| F01 | 19 / 36 | 69.44% | 50.00% | 44.44% |
| M01 | 55 / 135 | 82.22% | 46.67% | 42.22% |

## smoke

| Speaker | Clips / ref words | Frozen | Adapter-only | Adapter + last two blocks |
|---|---:|---:|---:|---:|
| F01 | 4 / 23 | 47.83% | 34.78% | 34.78% |
| M01 | 4 / 21 | 33.33% | 38.10% | 38.10% |
| M04 | 2 / 7 | 100.00% | 85.71% | 71.43% |

## Selected checkpoints

- adapter: step 600 of 600; trainable counts {'adapter': 1622016, 'pretrained': 0}; frozen-weight integrity passed.
- partial: step 550 of 600; trainable counts {'adapter': 1622016, 'pretrained': 50378752}; frozen-weight integrity passed.

The separate test contains unseen prompts from known speakers, not unseen speakers or new recording sessions. Development results are used for selection. Original smoke samples are diagnostic only. No weights were deployed.

## Interpretation and limitations

- On the separate test, F01 errors fall from 25/36 to 16/36 with selective
  unfreezing: 25 percentage points / 36% relative WER reduction. Adapter-only
  makes 18/36 errors, so the extra benefit is two word errors.
- M01 errors fall from 111/135 to 57/135: 40 percentage points / 48.65% relative
  WER reduction. Adapter-only makes 63/135 errors.
- The original F01 smoke score improves to 34.78% (8/23 errors); it is neither
  the new holdout metric nor evidence of 20–30% overall WER.
- The original M01 smoke score regresses by one error, from 7/21 to 8/21,
  despite larger development and holdout improvements. Both views are reported.
- F01 holdout sentence WER is 15.79%, but that subset contains only two
  sentences / 19 words. Its isolated-word WER is still 76.47% (17 words).
  This does not establish usability across spontaneous speech.
- F04 remains a speaker-disjoint development set with possible shared prompts;
  F01/M01 test prompts are excluded from all training speakers. M04 is a
  development guardrail, not a fresh independent test. M02 remains sealed.
- This is one seed and a small corpus. Sampling, learning rate, and selection
  changed from the original historical run, so improvement over that run cannot
  be attributed solely to balancing. The two new arms share those choices,
  isolating the added encoder-block fine-tuning more closely.
- Further experimentation must treat this now-inspected holdout accordingly;
  do not tune on it and continue calling it a blind final test.

## Verification

Nine local protocol/decode tests passed. Saved prediction WERs were recomputed
with the shared normalizer and scorer, and utterance identities match across
all three systems within each split. Both arms used exactly the same per-speaker
sample counts. Saved trainable names confirm that pretrained updates are confined
to encoder blocks 22/23; decoder and joint weights stayed frozen. Runtime hashes
confirm all non-trainable parameters remained unchanged.

[Raw comparison and row-level evidence](../../../results/focused/team34/wlvv40q/REPORT.md).
[Experiment implementation and protocol](README.md).
