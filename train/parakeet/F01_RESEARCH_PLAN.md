# F01-first improvement plan — 2026-09-19

## Recommendation

Prioritize a trustworthy F01 development split and
speaker-balanced/personalized adaptation before replacing Houlsby adapters with
AdaLoRA. The 20–30% goal is an experimental target, not a supported forecast.
No GPU job or new decoding run was performed for this review.

The task API exposed the title and original scope of `Start phase 1
implementation`, but returned empty items for recent turns. Detailed run context
was recovered from the repository README, NEXT_EXPERIMENTS, saved predictions,
hparams and continuation receipt.

## Recomputed evidence

Using the shared normalizer and jiwer on the saved baseline and continuation
predictions:

| Speaker / system | Reference words | Substitutions | Deletions | Insertions | WER |
|---|---:|---:|---:|---:|---:|
| F01 frozen | 23 | 10 | 1 | 0 | 47.83% |
| F01 selected adapter | 23 | 8 | 3 | 0 | 47.83% |
| M04 frozen | 7 | 0 | 7 | 0 | 100.00% |
| M04 selected adapter | 7 | 2 | 4 | 0 | 85.71% |

These are four F01 and two M04 smoke utterances, not full-speaker results.
Equal F01 WER does not mean identical predictions: `raid` changed from
`Really?` to empty; the second sentence also changed. `goat` remains `go`, and
`yet he still thinks` remains `Yes he screwed things`. This supports investigating
acoustic confusions as well as search; it does not prove their cause.

At 23 words, one error moves WER by 4.35 percentage points. Five or six errors
would score 21.74% or 26.09%; seven still scores 30.43%. Reaching the requested
band means removing five or six errors from this tiny slice, not establishing
general usability.

F04's selected internal validation WER is 23.55% over 244 rows; a matched frozen
F04 decode with the contract normalizer is missing. Do not call this a measured
frozen-to-adapted improvement or compare it directly with the smoke result.

Recomputing the existing derived training split gives F01 228 microphone rows
but only 114 utterance groups. Its head-mic view has 96 words, 18 sentences and
4.35 minutes of audio, all from session 1. F03 has 1,087 rows. F01 contributes
5.55% of training rows; uniform speaker sampling would allocate 16.67% of draws
before duration/group effects. Correlated microphones do not double diversity.

## Ordered experiments

1. **Make F01/M04 selection meaningful.** Create versioned derived splits,
   leaving frozen shared manifests intact. Withhold whole utterance groups and
   both microphones, stratifying word/sentence coverage. Group repeated normalized
   prompts as well when claiming unseen-text generalization. F01 has few sentences,
   so report split sizes and uncertainty explicitly; group cross-validation is an
   option if a single split is too unstable. Keep F04 as the unseen-speaker
   development guardrail and M02 sealed. Existing checkpoints have seen newly
   withheld rows: retrain from the frozen base for valid comparisons. New F01
   recordings from a separate session would be a stronger final test. The four
   repeatedly inspected smoke samples are now diagnostics, not a blind test.

2. **Keep greedy decoding.** The user rejected decoder tuning because of
   latency. Its implementation and local artifacts have been removed. Focus
   subsequent F01 work on training and acoustic adaptation.

3. **Run a controlled data/sampling comparison.** Start fresh dim-32 adapters
   on the new split: current sampling versus uniform-speaker/group-aware sampling.
   Choose one available microphone per group per epoch, preferentially head-mic.
   Keep optimizer, decoder, schedule and capacity fixed initially. Then test mild
   augmentation separately. Do not bundle rank, LR, augmentation and sampler
   changes into an uninterpretable comparison. Select primarily on F01 development
   WER with preregistered F04/M04 regression limits; show all three metrics.

4. **Try an F01-personalized branch.** From the new split-safe shared adapter,
   compare a short F01-specific adaptation with the shared model, using only F01
   training groups and a development-based stopping rule. Sweep adapter LR
   conservatively (e.g. 3e-6, 1e-5, 3e-5), rather than extending the saturated
   3e-7 continuation indefinitely. These are proposed values, not proven optima.
   Use some other-speaker training replay if adapting one shared model. Separate
   per-user adapters permit retaining the shared model for F04/M04, but require
   a known user/profile at inference. Do not simulate routing with hidden speaker
   labels in an unconditioned benchmark. This is personalized recognition, not
   unseen-speaker generalization.

5. **Compare adaptation methods after fixing selection.** Compare fixed LoRA
   against AdaLoRA on the same encoder projections, training distribution,
   decoder and approximately matched parameter budget. AdaLoRA needs a rank
   allocation schedule, importance updates and valid export/restore support;
   it is not a configuration toggle on the present Houlsby adapter. Log actual
   trainable names/counts and verify frozen base weights. Consider encoder
   attention projections and feed-forward matrices; leave prediction/joint
   adaptation as a separate experiment. A dim-64 Houlsby control offers a simpler
   capacity comparison. Checkpoint averaging is worthwhile only for compatible
   checkpoints from the same run/base; averaging independently initialized
   adapters or incompatible AdaLoRA rank layouts is not justified.

For M04, investigate empty outputs, waveform level/clipping and segmentation
before adding a deletion-targeted decoder modification. For F04, obtain the full
frozen/adapted contract comparison and audit malformed references without silently
changing them. A training-only LM may help repeated TORGO prompts, but report
prompt overlap and unseen-text performance; it is not evidence of recovered
acoustic recognition. Never build the LM with development/test references.

## What the papers actually support

- [Takahashi et al., Parakeet winner](https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.pdf):
  used Parakeet-TDT 1.1B and full fine-tuning with far more data. Decoding and
  checkpoint merging moved Test1 WER from 6.12 to 5.97, about 2.45% relative.
  This is not evidence that decoding alone halves our F01 WER.
- [Tan et al., CBA-Whisper](https://www.isca-archive.org/interspeech_2025/tan25b_interspeech.pdf):
  second-place Whisper large-v2 system using AdaLoRA plus curriculum/data filtering
  and preprocessing. Initial rank 12, target rank 4; the submission sequence does
  not isolate AdaLoRA against a matched LoRA control. Its total gain cannot be
  attributed solely to rank allocation or transferred numerically to Parakeet.
- [Wagner et al.](https://www.isca-archive.org/interspeech_2025/wagner25_interspeech.pdf):
  Table 1 is more informative for this decision. Without personalization and
  without SpecAugment, LoRA scores 10.87 versus AdaLoRA 11.73 on development.
  With personalization, AdaLoRA reaches 8.05 versus LoRA 10.52. This suggests
  testing personalization together with adaptive rank, not assuming AdaLoRA is
  universally superior. These are Whisper/SAP results, not predictions for F01.
- [NeMo v2.4.0 decoding source](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/asr/parts/submodules/rnnt_decoding.py):
  shows TDT beam and mAES dispatch, softmax temperature and mAES n-gram hooks.
  Runtime validation against the pinned RC container remains outstanding.

## Success criterion

Treat 20–30% as achieved only on disjoint F01 evaluation audio under a frozen
selection/decoding policy. Report word count, word/sentence WER, per-speaker
regressions and group-bootstrap uncertainty. Fresh-session speech and critical
word fidelity are needed to substantiate usability beyond TORGO read prompts.
