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

All from the sealed test speaker, frozen model against tuned. The seven
sentences are the clips the demo ships, selected because the tuned model matches
the reference exactly and the frozen model does not.

| Said | Frozen heard | Tuned |
|---|---|---|
| this was easy for us | "thats whats using force" | exact |
| why yell or worry over silly items | "why yelled a warrior of a silly agent" | exact |
| he will allow a rare lie | "he will **not** allow a real life" | exact |
| we gathered shells on the beach | "forget the tails on the beach" | exact |
| their house is grey and white | "there are houses grey and white" | exact |
| two other cases also were under advisement | "two of the cases also were under advisement" | exact |
| twice each day he plays skillfully and with zest upon our small organ | "twice each day **she** played skilfully and with **that** upon **her** small organ" | exact |
| wicked | "who were killed" | wicked |
| jagged | "they are good" | jagged |
| brawn | "oh put it on" | brawn |
| witty | "where is he ah where is he" | witty |

The frozen model does not fail quietly. It produces fluent, confident, wrong
English, which is exactly the failure the clarification step exists to catch.
Row three is the sharpest case: the baseline inserts a negation that was never
spoken, inverting the meaning.

## End to end, live

`scripts/e2e_check.py` drives the real chain once per demo clip: 20 ms PCM
frames to the deployed Baseten endpoint over Protocol 1, then enrollment and
synthesis against the OpenVoice renderer on MPS. Nothing mocked, nothing
precomputed. Measured 2026-09-19 23:05 EDT.

| # | Clip | Transcript exact | Confidence | Asks | Alternatives | First partial | Final | Synthesis |
|---|---|---|---:|---|---:|---:|---:|---:|
| 1 | M02_1_headMic_0141 | yes | 0.924 | no | 1 | 645 ms | 4854 ms | 862 ms |
| 2 | M02_1_headMic_0135 | yes | 0.815 | yes | 3 | 596 ms | 10036 ms | 1078 ms |
| 3 | M02_1_headMic_0185 | yes | 0.901 | no | 1 | 604 ms | 6288 ms | 739 ms |
| 4 | M02_1_headMic_0184 | yes | 0.848 | yes | 3 | 591 ms | 6820 ms | 1725 ms |
| 5 | M02_1_headMic_0196 | yes | 0.976 | no | 1 | 598 ms | 8493 ms | 892 ms |
| 6 | M02_2_headMic_0088 | yes | 0.754 | yes | 3 | 612 ms | 11837 ms | 2002 ms |
| 7 | M02_1_headMic_0092 | yes | 0.892 | no | 1 | 608 ms | 14191 ms | 1011 ms |

**All seven transcribe exactly.** `final` is wall clock from the first frame, so
it includes streaming the clip in real time; the decode itself lands roughly 1.5
to 2.5 s after the audio ends. Three clips fall under the 0.879 threshold and
ask before speaking, each offering three genuine alternatives with the model's
own best guess ranked first.

Driven by hand in a browser over the same path: synthesis after confirmation
took 1317 ms and 735 ms.

**Cold start is 33 seconds.** Autoscaling is `min_replica: 0` with a 60 s
window, so the endpoint sleeps when idle and the first connection times out.
Either warm it before demoing, or set `min_replica: 1` beforehand and accept an
idle H100.

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

| | H100 | Mac MPS | Mac CPU |
|---|---:|---:|---:|
| 7-word synthesis | **196 ms** | 1078 ms | 2200 ms |
| Base TTS / tone conversion | 52 / 141 ms | n/a | 703 / 1497 ms |

`serve/voice/server.py` selects MPS when CUDA is absent, which is roughly 3x
faster than CPU on Apple Silicon; override with `VOICE_DEVICE=cpu` if a
converter kernel is missing on an older torch. At these speeds whole-clause
synthesis fits the 1.5 s first-audio budget, so chunked streaming TTS was cut
from the plan.

## Serving

| Path | First partial | Per clip | Notes |
|---|---:|---:|---|
| Baseten H100, production (`woo8697`) | ~600 to 1000 ms | 4 to 14 s | Network, 400 ms re-decode cadence, and real-time streaming of the clip |
| Mac, MPS + bfloat16 | n/a | 145 to 292 ms | 0.5 GiB. fp16 overflows this model's attention mask |
| On-box inference alone | n/a | 70 ms | The model itself |

## What these numbers do not show

- Eight speakers, two of them genuinely held out. Evidence, not proof of
  generalization.
- TORGO is **read prompts**: isolated words and a fixed sentence set. No
  spontaneous or conversational speech.
- We tested the adapter on spontaneous conversational speech from outside the
  corpus, an 11.8 s clip of a stroke survivor in a clinical interview. It did
  not transfer: frozen 0.654 WER, tuned 0.731. Confidence correctly collapsed to
  0.657 and the system asked rather than asserting, but the fine-tuning bought
  nothing there. The adapter is specialized to TORGO's distribution.
- 76% of the sealed test set is single isolated words, which is the hardest case
  and where the gain is largest. Longer utterances gain 17 to 25%.
- 34.8% WER on M02 is still high. The claim is the relative reduction on an
  unseen speaker, not that the problem is solved.
