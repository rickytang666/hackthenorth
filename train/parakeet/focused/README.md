# F01/M01 focused adaptation

Fixed greedy decoding; no decoder tuning. Both arms start from the ordinary
frozen Parakeet-TDT-0.6B-v2 checkpoint with identically initialized dim-32
Houlsby encoder adapters. The old trained adapter is deliberately not reused:
it has seen recordings newly withheld for this experiment.

The versioned split in `data.py` excludes normalized development/test prompts
from all training speakers, along with every microphone view. F01 receives
priority when assigning prompts, then M01, then M04, so later assignments do
not consume extra F01 training data. F04 retains its original speaker-disjoint
244-row development set; its prompt overlap is not removed. Original smoke
utterance groups remain excluded. M02 stays sealed.

The sampler draws speakers uniformly, then utterance groups uniformly within a
speaker. Where both microphones exist, it selects headMic with probability 0.7.
This avoids counting the two microphone views as independent training examples.

Both arms use the same seed and sample stream, four microbatches of four clips
per optimizer step, AdamW, 25-step warmup and gradient clipping at 1. Adapter
LR is 3e-5. The selective arm additionally trains encoder blocks 22 and 23
(zero-based), with pretrained LR 3e-6. Prediction/joint weights stay frozen;
all batch-normalization running statistics remain fixed. SpecAugment is off to
isolate the adaptation comparison. Frozen-weight hashes are checked at the end.

The predeclared ceiling is 600 optimizer steps per arm. Every 50 steps,
checkpoint selection minimizes (F01 dev WER, M01 dev WER) lexicographically,
subject to regressions versus frozen no greater than 5 percentage points on
M01/M04 and 2 points on F04. Stop after five checks without an eligible
improvement. Step zero is retained if training cannot beat it. The separate
F01/M01 test split is scored only after each arm's checkpoint is selected; it
must not be used to retune either arm. The frozen test baseline is scored after
both arms. These are same-speaker, unseen-prompt held-out recordings, not a
new-session or unseen-speaker test.

Run staging with `VOICEBRIDGE_DATA` set to the original local data root:

```sh
python -m train.parakeet.focused.data
truss train push train/parakeet/focused/config.py --remote baseten-team34 --team 34
```

Outputs are grouped by baseline/adapter/partial/final_frozen beneath the
Baseten checkpoint `results/` directory. Each trained arm exports a full
`best.nemo`, row-level greedy predictions, development history, parameter names,
selection details and split hashes. The selective checkpoint must remain a full
model; exporting just its adapter would discard the unfrozen-block updates.

## 1.1B comparison

`config_1b.py` uses `nvidia/parakeet-tdt-1.1b`, with up to 1,500 steps per arm
and the same five-check plateau rule. Adapter input width and final-two-layer
indices are derived from the loaded model, rather than assumed from the 0.6B
architecture. Training transcripts use the shared normalized lowercase text for
this uncased tokenizer. Greedy decoding and scoring normalization are unchanged.
The previous test split has already been inspected, so this reuse is a comparative
benchmark, not a new blind test. Compare development histories at 600 steps as
well as final selected checkpoints to distinguish training-budget differences.
This is a different pretrained checkpoint family/version, not a controlled
parameter-count-only scaling experiment.
