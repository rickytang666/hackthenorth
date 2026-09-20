# Expanded validation v1

Created at the user's request to include more male speakers in development.
These are separate derived manifests; existing Phase 0, Phase 1 and in-progress
focused-experiment files are unchanged. No training job is launched or reconfigured.

| Speaker | Development clips | Words / sentences (clips) | Reference words |
|---|---:|---:|---:|
| F01 | 23 | 19 / 4 | 55 |
| M01 | 73 | 56 / 17 | 173 |
| M03 | 81 | 62 / 19 | 199 |
| M04 | 78 | 61 / 17 | 202 |
| M05 | 94 | 71 / 23 | 249 |
| F04 | 244 | 176 / 68 | 637 |

Total: 593 development recordings, including 326 male recordings. The paired
training manifest contains 3,505 microphone rows before duration filtering.

## Using the split

Run from the repository root with VOICEBRIDGE_DATA set to the dataset root:

```bash
python -m train.parakeet.expanded_validation
```

The generator writes into `$VOICEBRIDGE_DATA/manifests`:

- `expanded_v1_train.jsonl`: use this for a new model's training data.
- `expanded_v1_dev.jsonl`: development scoring, one view per held-out utterance
  plus the unchanged full F04 validation set.
- `expanded_v1_withheld_views.jsonl`: all views of the newly withheld groups;
  for split auditing, not extra training data.
- `expanded_v1_smoke.jsonl`: the original ten smoke clips, still separate.
- `expanded_v1_receipt.json`: hashes, per-speaker counts, microphone and prompt
  overlap statistics, and the checkpoint-selection policy.

Use mean(F01 WER, M01 WER) for checkpoint selection, computing each WER from
that speaker's total word errors/reference words. Do not pool speakers or average
per-clip WER. Report F01 and M01 individually, with F04/M03/M04/M05 as regression
checks. Set any hard regression tolerances before training. Keep greedy decoding.

## Split guarantees and limits

Selection is deterministic, stratified by speaker and word/sentence type, with
approximately 20% of available utterance groups withheld. The original ten smoke
groups are already excluded before this partition. All microphone views stay on
the same side. Development prefers headMic; 24 M04 groups have only another
microphone available and are represented with that view. Report headMic-only
M04 metrics alongside the aggregate if microphone comparability matters.

The split is utterance-disjoint, not prompt-disjoint. Repeated prompts from other
recordings/speakers remain in training; overlap counts are in the receipt. F04
remains speaker-disjoint. M02 is neither read nor copied by the generator and
remains sealed for the final test.

Existing trained adapters have seen these newly withheld recordings. Start fresh
from the frozen pretrained base when evaluating training methods on this split.
Do not warm-start from the old adapted checkpoint and call these held-out results.
The original frozen model can be scored directly. F01 still has only four
sentence clips here, so its score remains noisy despite the expanded male coverage.
