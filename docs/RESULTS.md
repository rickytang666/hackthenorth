# VoiceBridge results

Fine-tuning Cohere Transcribe on dysarthric speech. Every number here was
measured on one H100 with one evaluator (`contract/evaluate.py`) and one text
normalizer (`contract/normalize.py`), so the columns compare models rather than
machines.

Raw prediction files stay out of git: they contain TORGO transcripts and this
repo is public. Regenerate them with `train/cohere/config_eval.py`.

## Headline

**Word error rate fell 38.6% relative on a speaker the model had never heard,
from a LoRA adapter touching 0.166% of the model, trained in 10.6 minutes.**

## Sealed test: speaker M02

M02 was held out from training and from every model-selection decision, then
decoded exactly once, on 2026-09-19 at 18:33, job `wd66me3`. 388 clips.

| Metric | What it means | Base | Tuned | Change |
|---|---|---:|---:|---:|
| Word error rate | Share of words wrong, corpus level | 0.5668 | **0.3481** | 38.6% better |
| Isolated-word WER | Single-word clips. Above 1.0 means it emits more words than were spoken | 1.0169 | **0.4628** | 54.5% better |
| Sentence WER | Full-sentence clips, where context helps | 0.3634 | **0.2962** | 18.5% better |
| Critical error rate | Safety-relevant words lost: negation, yes/no, numbers, medications | 0.4000 | **0.2857** | 28.6% better |
| Clips exactly right | Whole clip correct, no edits | 98 / 388 | **198 / 388** | 2.0x |
| p50 / p95 latency | Per clip, steady state | 52 / 154 ms | 70 / 216 ms | slower, see below |
| Real-time factor | Inference seconds per audio second | 0.0201 | 0.0257 | 39x real time |

Latency got worse: LoRA adds adapter matrix multiplications. At 39x real time it
does not matter, but it is not a win and is not presented as one.

## All eight dysarthric speakers

Job `qj99zjw`, 2,820 headMic clips. **Six of these speakers are in the tuned
model's training split** and their rows are optimistic by construction. Only F04
and M02 are generalization results.

| Speaker | Role | Clips | Base WER | Tuned WER | Relative | Critical base | Critical tuned |
|---|---|---:|---:|---:|---:|---:|---:|
| M03 | trained | 406 | 0.0740 | 0.0148 | 80.0% | 0.0000 | 0.0000 |
| F04 | **validation** | 244 | 0.1099 | **0.0330** | 70.0% | 0.0000 | 0.0000 |
| F03 | trained | 546 | 0.3082 | 0.1262 | 59.1% | 0.0702 | 0.0702 |
| M05 | trained | 470 | 0.3918 | 0.1480 | 62.2% | 0.0667 | 0.0444 |
| M02 | **sealed test** | 388 | 0.5668 | **0.3481** | 38.6% | 0.4000 | 0.2857 |
| F01 | trained | 118 | 0.6596 | 0.2943 | 55.4% | 0.3333 | 0.1111 |
| M01 | trained | 371 | 0.6674 | 0.2927 | 56.1% | 0.4054 | 0.1622 |
| M04 | trained | 277 | 1.1268 | 0.3276 | 70.9% | 0.4545 | 0.1818 |

**Every speaker improved. Critical errors improved or held flat in every row,
and worsened in none.**

Difficulty ranking by the frozen baseline, which is fair because it saw nobody:

```text
easiest  M03 .074 < F04 .110 < F03 .308 < M05 .392
         < M02 .567 < F01 .660 < M01 .667 < M04 1.127  hardest
```

The gain grows with severity: 70.9% on the hardest speaker, 56.1% on the second
hardest, against 38.6% on M02. That is the product thesis, measured.

## What the recognizer actually fixes

Sealed-test examples, frozen model against tuned:

| Said | Frozen heard | Tuned |
|---|---|---|
| wicked | "who were killed" | wicked |
| jagged | "they are good" | jagged |
| brawn | "oh put it on" | brawn |
| witty | "where is he ah where is he" | witty |

The frozen model does not fail quietly. It produces fluent, confident, wrong
English, which is exactly the failure the clarification step exists to catch.

## Training

| | |
|---|---|
| Base model | `CohereLabs/cohere-transcribe-03-2026`, 2.07B parameters, Apache 2.0 |
| Method | LoRA, rank 8, alpha 16, dropout 0.05. Nothing unfrozen |
| Targets | 128 modules: encoder blocks 42 to 47 of 48, plus all 8 decoder layers |
| Trainable | 3,424,256 of 2,068,023,552 = **0.166%** |
| Objective | Audio-to-transcript cross-entropy, teacher forced |
| Schedule | AdamW, lr 1e-4, 20-step warmup, linear decay, batch 4 x 4 accum, bf16 |
| Steps | 1200, 19,200 examples, **10.6 min at 30 ex/s** on one H100 |
| Selection | Lowest dev loss, 0.1428 at step 900. Not the final step, which drifted to 0.1604 |

The encoder is targeted because it holds 91.8% of the parameters and dysarthria
is an acoustic problem. Adapting the decoder alone would touch 7% of the model.

## Capacity sweep: is the adapter the right size?

Job `qv55jj3`, four configs trained from scratch for 900 steps each and scored
on F04, the validation speaker. The shipped config was re-run inside the same
job so the comparison is against a number measured on the same box, not a
remembered one. M02 was not decoded.

| Rank | Enc blocks | Trainable | % of model | F04 WER | Words | Sentences | Dev loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | **6** | 3,424,256 | **0.166%** | **0.0361** | 0.0909 | 0.0152 | 0.1387 |
| 32 | 6 | 13,697,024 | 0.659% | 0.0408 | 0.1193 | 0.0108 | 0.1513 |
| 8 | 16 | 6,291,456 | 0.304% | 0.0377 | 0.0909 | 0.0174 | 0.1399 |
| 32 | 16 | 25,165,824 | 1.204% | 0.0408 | 0.1080 | 0.0152 | 0.1446 |

**The smallest adapter wins, and every increase in capacity costs accuracy.**
WER and dev loss agree, which makes it unlikely to be noise. At 7.3x the
parameters the model is 13% worse.

The binding constraint is data, not capacity: 4.06 hours across 6 speakers is
what caps this, and rank 8 on 6 encoder blocks is already the right size for it.
The original run's dev loss bottoming at step 900 of 1200 was the same signal.

Practical consequence: the configuration is chosen by measurement rather than
inherited, and there is no accuracy left on the table from a bigger adapter.
More speakers would help; more parameters will not.

## Data

TORGO, `abnerh/TORGO-database`. Split by speaker before any sampling, frozen
before any baseline was scored, hashes in `contract/MANIFEST_HASHES` and
verified on the training box before every run.

| Split | Clips | Hours | Speakers |
|---|---:|---:|---|
| Train | 4,129 | 4.06 | F01 F03 M01 M03 M04 M05 |
| Validation | 244 | 0.24 | F04 |
| Sealed test | 388 | 0.42 | M02 |
| Control replay | 412 | 0.35 | FC01 FC02 |
| Control eval | 1,592 | 1.05 | FC03 MC04 |

Both microphone views train; only `headMic` is scored. Utterance groups never
cross a split.

## Voice rendering

OpenVoice V2, speaker embedding from five M02 clips. Job `w7rrz03`.

| | H100 | Mac CPU |
|---|---:|---:|
| 7-word synthesis | **196 ms** | 2200 ms |
| Base TTS / tone conversion | 52 / 141 ms | 703 / 1497 ms |

At 196 ms, whole-clause synthesis fits the 1.5 s first-audio budget, so chunked
streaming TTS was cut from the plan.

## Serving

| Path | First partial | Per clip | Notes |
|---|---:|---:|---|
| Baseten H100, production (`woo8697`) | ~1000 ms | 4 to 7 s | Network plus 400 ms re-decode cadence |
| Mac, MPS + bfloat16 | n/a | 145 to 292 ms | 0.5 GiB. fp16 overflows this model's attention mask |
| On-box inference alone | n/a | 70 ms | The model itself |

## End to end, live

Audio into the deployed Baseten endpoint, through the deterministic verifier,
out as the speaker's own voice. Four sealed-test clips, all transcribed
correctly, `model_id` confirming the tuned adapter.

| Clip | ASR | Confidence | Verdict | First partial | ASR total | Synthesis |
|---|---|---:|---|---:|---:|---:|
| wicked | wicked | 0.815 | clarify | 1757 ms | 5555 ms | 2900 ms |
| jagged | jagged | 0.985 | accept | 880 ms | 4413 ms | 1063 ms |
| brawn | brawn | 0.645 | clarify | 1451 ms | 4326 ms | 1005 ms |
| witty | witty | 0.794 | accept-then-ask | 907 ms | 3726 ms | 1041 ms |

**Cold start is 33 seconds.** Autoscaling is `min_replica: 0` with a 60 s
window, so the endpoint sleeps when idle and the first connection times out.
Either warm it before demoing, or set `min_replica: 1` shortly beforehand and
accept an idle H100.

Three of four route to `clarify` under the calibrated 0.879 threshold even
though all four are correct. That is the intended trade: the threshold was set
for 95% precision on accepted answers, and the cost is asking more often.

## What these numbers do not show

- Eight speakers, two of them genuinely held out. Not proof of generalization.
- TORGO is **read prompts**: isolated words and a fixed sentence set. No
  spontaneous or conversational speech, so nothing here speaks to open dialogue.
- 76% of the sealed test set is single isolated words, which is the hardest case
  and where the gain is largest. Longer utterances gain 17 to 25%.
- 34.8% WER on M02 is still high. The claim is the relative reduction on an
  unseen speaker, not that the problem is solved.
