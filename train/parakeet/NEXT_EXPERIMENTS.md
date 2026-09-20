# Person A: reducing Parakeet WER after the 3,000-step gate

## What the completed runs establish

| Run | F04 validation | Frozen 10-row baseline slice | Decision |
|---|---:|---:|---|
| Frozen Parakeet v2 (`31mmzj3`) | Not measured | WER 0.4902, CER 0.3621 | Baseline gate passed |
| Full-model smoke (`w6vvnyq`) | Best WER 0.4490 | WER 0.6275 | Reject full-model tuning |
| Dim-32 encoder adapter (`qv557e3`) | Best WER 0.2418 near step 1,964 | WER 0.4706, CER 0.3292 | Keep adapter |
| Warm continuation (`wp11ylw`) | Best WER 0.2355 near total step 2,348 | WER 0.4706, CER 0.3333 | Stop before 4,000 |

The continuation stopped near total step 2,966 after five validations failed to
beat the step-2,348 checkpoint. This establishes saturation for the current
data distribution, dim-32 adapter, and optimizer recipe. It is not proof that
every non-best validation value rose monotonically: the retained logs expose
the best value and the sequence of non-improvements, but not every exact
non-best value. Call this a plateau or possible overfitting, not a measured
monotonic overfitting curve.

The 10-row baseline slice is a smoke test, not a model-selection set. It is
utterance-group-disjoint but uses F01, M01, and M04, which also appear in
training. F04 is the actual held-out speaker. M02 remains sealed and must not be
decoded until the final model and decoding policy are frozen.

## Recommended order of work

### 1. Close the evaluation gap before another training run

Decode both the untouched base and the selected `wp11ylw` checkpoint on all
244 F04 rows. Save row-level hypotheses and report overall WER/CER plus separate
single-word and sentence WER. The current 10-row slice has only five isolated
words, so its 1.0 isolated-word WER is too noisy to diagnose the model.

Also audit the fixed references without silently editing them. At least one F04
label is visibly malformed: `F04_2_headMic_0009` ends in a long run of `D` and
`C` characters. Preserve the frozen-contract score, and optionally report a
label-audit sensitivity score beside it. Any change to the frozen manifest
requires a team announcement; it is not necessary for the next Person A run.

### 2. Keep greedy decoding

Decoder tuning was rejected by the user due to its latency cost. Keep the
existing greedy decoder; do not reintroduce beam/mAES tuning or deploy a tuned
decoder. Continue with training-data and acoustic-adaptation improvements.

### 3. Fix sampling imbalance and correlated microphone duplication

The 4,109-row training split is not speaker-balanced: F01 has 228 rows while
F03 has 1,087, a 4.8x difference. It is also 3,151 single-word clips versus 958
sentences. Both microphone views of many utterances are present, while scoring
uses only `headMic`.

For the next fresh adapter run:

1. Sample speakers approximately uniformly rather than rows uniformly.
2. Treat an `utterance_group` as the sampling unit and choose one microphone
   view per group per epoch. Prefer `headMic` (for example 70%) while retaining
   `arrayMic` as acoustic augmentation.
3. Choose any word-versus-sentence reweighting only after the full F04 subset
   evaluation shows which subset is weak. Do not infer it from five smoke rows.

This should improve effective diversity and reduce repeated exposure to nearly
the same label/audio pair. It stays entirely within Person A's frozen training
speakers and does not touch the shared manifests.

### 4. Add mild regularization, then retry a fresh adapter

The completed recipe disabled SpecAugment (`freq_masks=0`, `time_masks=0`) and
made roughly 24 passes before early stopping. A fresh run should test modest
SpecAugment, such as two narrow frequency masks and two short time masks, while
keeping the same validation and early-stop policy. NeMo documents SpecAugment
directly in the model config. Because the current loader is Lhotse, do not add
the legacy `train_ds.augmentor` block; NeMo documents that it is unsupported by
the Lhotse loader. Use Lhotse-native augmentation or precompute conservative
speed/gain variants instead.

Start with speed factors close to the source, such as 0.95/1.0/1.05, so the
augmentation does not erase clinically meaningful rate characteristics. Add
noise/RIR only as a separate robustness experiment, not to the first WER run.

References:

- [NeMo ASR augmentation configuration](https://docs.nvidia.com/nemo/speech/nightly/asr/configs.html)
- [NeMo speed perturbation API](https://docs.nvidia.com/nemo/speech/nightly/asr/api.html)

### 5. Increase adapter capacity only after sampling and regularization

Compare the current dim-32 adapter (1.62M trainable parameters) with dim 64.
Keep the frozen base, use the same evaluation cadence, cap the run near 1,200
optimizer steps initially, and let early stopping decide. A clean A/B test is
more informative than continuing the saturated dim-32 adapter to 4,000.

Suggested small matrix:

| Variant | Sampler | SpecAugment | Adapter dim | Initial LR |
|---|---|---|---:|---:|
| Control | current | off | 32 | `3e-6` |
| Data fix | balanced/group-aware | mild | 32 | `3e-6` |
| Capacity | balanced/group-aware | mild | 64 | `1e-6` |

Do not run all cells blindly. Run 300–400 steps, compare the full F04 curve,
and continue only the best non-regressing variant.

### 6. Escalate architecture only if the controlled adapter test stalls

The next controlled escalation is selective adaptation, not another full-model
run: keep the dim-64 adapter and unfreeze only the last two encoder blocks or
their layer norms with a base-model learning rate around 10x below the adapter
rate. The earlier full-model smoke already showed that updating all 617M
parameters is unstable on this small corpus.

As a separate architecture experiment, Parakeet-TDT 1.1B is defensible because
the project plan already names it as Person A's historical comparison and a
published Speech Accessibility Project system used that family. First decode
it frozen on F04; only adapt it if the frozen result and latency justify the
extra cost. Do not change the English tokenizer.

Longer-term research directions include speaker-deficiency-conditioned
adapters and prototype-based unseen-speaker adaptation, but those require a
new experimental protocol and are not the immediate next Baseten job:

- [Structured Speaker-Deficiency Adaptation](https://arxiv.org/abs/2412.18832)
- [Prototype-Based Adaptation for unseen dysarthric speakers](https://arxiv.org/abs/2407.18461)

## What not to do

- Do not continue the unchanged dim-32 run to 4,000; its own early-stop rule
  rejected that spend.
- Do not unfreeze the entire model again; the 100-step smoke regressed sharply.
- Do not add healthy controls as positive dysarthric examples. The clean replay
  set is only a catastrophic-forgetting safeguard, capped at 10%, and should be
  activated only after measuring the separate clean-speech regression gate.
- Do not inspect M02 while choosing data, decoding, adapter size, or learning
  rate.
- Do not use Person B's Cohere code or outputs. Every recommendation above is a
  self-contained Person A experiment.
