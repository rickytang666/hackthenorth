# Person A: Parakeet lane

Phase 1 starts with a 10-row frozen-model smoke baseline. Do not submit a
fine-tuning job until `results/baseline_parakeet.jsonl` contains all 10 rows and
`contract.evaluate` has printed its scorecard.

Person A derives three non-overlapping purposes without altering the frozen
Phase 0 manifests:

- training: 4,109 rows from the six training speakers, after withholding both
  microphone views for each baseline utterance group;
- baseline smoke set: 10 head-mic rows from severe speakers F01, M01, and M04
  (five isolated words, five sentences);
- validation: all 244 F04 rows from the frozen dev manifest.

The M02 test speaker remains sealed. Baseline is group-disjoint from training;
validation remains fully speaker-disjoint from both.

## Baseline on an existing CUDA/NeMo machine

```bash
source env.sh
python -m train.parakeet.stage_baseline_data
export VOICEBRIDGE_DATA=/private/tmp/voicebridge-parakeet-baseline/data/voicebridge
python -m train.parakeet.decode --manifest torgo_dys_baseline_10.jsonl
python -m contract.evaluate results/baseline_parakeet.jsonl --json
```

## Baseline on Baseten

This follows the current Baseten ML Cookbook pattern: a `TrainingProject`, one
H100, a mounted Hugging Face checkpoint, a minimal bundled local dataset, and
checkpoint storage for durable outputs. The job is evaluation-only. The staging
step includes only 10 severe dysarthric baseline WAVs, not the training,
validation, healthy-control, or sealed-test recordings.

```bash
source env.sh
python -m train.parakeet.stage_baseline_data
export PINNED_NEMO_IMAGE='<Phase-0-tested NeMo image>'
truss train push train/parakeet/baseline_config.py
```

Retrieve the job artifacts with `truss train get_checkpoint_urls --job-id
$JOB_ID`; the predictions and scorecard are under `rank-0/results/`.

The pre-training baseline gate cleared on Baseten team 34 in job `31mmzj3`:
10/10 predictions were written, WER was 0.4902, and the real-time factor was
0.0065. Local copies are under `results/baseline/team34/31mmzj3/`.

## Fine-tuning smoke run

The first paid training run is deliberately limited to 100 steps. It mounts
only dysarthric training/validation audio plus the separate 10-row post-train
baseline set: about 494 MB total. Healthy controls and the sealed M02 test are
not present in the job. NeMo filters the two training recordings longer than 30
seconds, as required by the plan.

```bash
source env.sh
python -m train.parakeet.stage_training_data
export PINNED_NEMO_IMAGE=nvcr.io/nvidia/nemo:25.04.03
export BASETEN_TRAINING_PROJECT_NAME=voicebridge-parakeet-v2-torgo-t34
MAX_STEPS=100 truss train push train/parakeet/training_config.py \
  --remote baseten-team34 --team 34 --job-name parakeet-torgo-smoke-100
```

The job saves NeMo checkpoints and evaluates the newest `.nemo` checkpoint on
the same 10 withheld baseline utterances. Continue beyond 100 steps only if its
WER beats the frozen 0.4902 baseline.

### Smoke result: do not auto-continue

Team 34 job `w6vvnyq` completed 100 optimizer steps and restored the best
validation checkpoint (`val_wer=0.4490`) before decoding the 10 withheld
utterances. Its WER was 0.6275 versus the frozen model's 0.4902: a 0.1373
absolute regression (28.0% relative worsening). Sentence WER regressed from
0.4348 to 0.5870, isolated-word WER remained 1.0000, and latency was unchanged.

This fails the 100-step smoke criterion above, so do not automatically launch
the 800-step job or introduce clean replay to rescue it. The plan's formal kill
rule is at 200 steps against the frozen full-dev baseline; that comparison is
not available because the requested frozen baseline was limited to 10 samples.
Treat further H100 spend as an explicit decision, not a continuation of this
run. The remote checkpoint remains in Baseten; the small evidence artifacts are
under `results/training/team34/w6vvnyq/`.

## Parameter-efficient continuation

The continuation recipe uses NeMo's supported linear/Houlsby adapters rather
than full-model fine-tuning. It inserts a bottleneck adapter of dimension 32 in
each FastConformer encoder block, freezes all 617M base-model parameters, and
updates only the enabled adapter weights. This is an ASR encoder adapter, not
LLM-style LoRA.

The run has a 2,000-step ceiling, validates every 400 microbatches (about
100-125 optimizer steps with four-way accumulation and epoch boundaries),
exports the top `val_wer` checkpoint as a full `.nemo`, and stops after five
consecutive validation checks without an improvement of at least 0.001 WER.
The pinned NeMo adapter path cannot restore its sparse Lightning checkpoint
directly, so the best `.nemo` is retained and decoded without overwriting it at
train end. The requested learning rate is `3e-6`; it is intentionally
configurable because this is conservative for randomly initialized adapters.
Each microbatch is capped at 60 seconds of audio and four microbatches are
accumulated per optimizer step. A duration cap is required because Parakeet's
Lhotse dataloader discards the adapter template's clip-count batch setting.
Both dataloaders explicitly use the manifests' `text` transcript field rather
than Parakeet V2's inherited `answer` field.

```bash
source env.sh
python -m train.parakeet.stage_training_data
export PINNED_NEMO_IMAGE=nvcr.io/nvidia/nemo:25.04.03
export BASETEN_TRAINING_PROJECT_NAME=voicebridge-parakeet-adapter-torgo-t34
truss train push train/parakeet/training_config.py \
  --remote baseten-team34 --team 34 --job-name parakeet-torgo-adapter-2000
```

`MAX_STEPS`, `VAL_INTERVAL`, `EARLY_STOPPING_PATIENCE`, `ADAPTER_DIM`,
`ADAPTER_LR`, `TRAIN_BATCH_DURATION`, and `GRAD_ACCUMULATION` can be overridden
when submitting the job. The post-training gate still decodes only the same 10
frozen baseline utterances; validation uses all 244 F04 rows. The sealed M02
test speaker and all healthy controls remain absent from the job.

### Adapter result

Team 34 job `qv557e3` completed the 2,000-step ceiling. Early stopping did not
fire because F04 validation WER continued improving through step 1,964, where
the best value was 0.2418. The best exported `.nemo` was then evaluated on the
same frozen 10 utterances:

- WER: 0.4706, improved from the frozen model's 0.4902;
- sentence WER: 0.4130, improved from 0.4348;
- isolated-word WER: 1.0000, unchanged;
- critical error rate: 0.0000;
- real-time factor: 0.0077.

The compact adapter weights and small evidence artifacts are under
`results/training/team34/qv557e3/`. The merged 2.48 GB `.nemo` remains in
Baseten checkpoint storage.

### Continuation result: stop before 4,000

The saved 2,000-step adapter was warm-started in team 34 job `wp11ylw`. This
restored the adapter weights exactly, kept 617,825,926 base parameters frozen,
and trained only 1,622,016 adapter parameters. Baseten did not register the
prior custom NeMo files as resumable checkpoints, so optimizer momentum and
scheduler counters could not be restored; the continuation used a fresh AdamW
optimizer at `3e-7` and a `1e-7` cosine floor.

The continuation's F04 validation sequence was:

- local step 100: WER 0.2449;
- local step 224: no new best;
- local step 348 (about total step 2,348): WER 0.2355, the selected best;
- local steps 472, 596, 719, 842, and 966: no new best.

Early stopping fired at local step 966, approximately total step 2,966, after
five consecutive evaluations failed to improve the step-348 checkpoint. The
selected checkpoint's 10-row baseline WER remained 0.4706 and sentence WER
remained 0.4130; CER moved from 0.3292 to 0.3333. This is evidence of a
validation plateau/saturation, so the 3,000-to-4,000 stage was not launched.
The best compact adapter, TensorBoard event, logs, predictions, receipt, and
scorecard are under `results/training/team34/wp11ylw/`; the best full `.nemo`
remains in Baseten checkpoint storage. The Baseten run is visible at
<https://app.baseten.co/training/qrj8y03/logs/wp11ylw>.

The evidence-ranked plan for reducing WER is in
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md). It retains greedy decoding and prioritizes full F04 error
analysis, a speaker-balanced/group-aware sampler,
mild regularization, and a controlled dim-64 adapter comparison.
