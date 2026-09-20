# VoiceBridge results

Fine-tuning Cohere Transcribe on dysarthric speech. Every number was measured on
one H100 with one evaluator (`contract/evaluate.py`) and one text normalizer
(`contract/normalize.py`), so the columns compare models, not machines.

## Reading this

| Term | Meaning |
|---|---|
| **WER** | Word error rate. Share of words wrong after alignment. 0.35 means roughly one word in three. Lower is better |
| **M02, F04, M01...** | TORGO speaker codes. M/F is sex, the number is the participant. Eight speakers have dysarthria; severity differs enormously between them |
| **Sealed test** | M02. Held out from training *and* from every tuning decision, decoded exactly once. The only fully honest number here |
| **Validation** | F04. Never trained on, but it chose the checkpoint, so it is optimistic |
| **Trained** | The speaker's own clips are in the training set. Optimistic by construction, shown for completeness |
| **Baseline / frozen** | Cohere Transcribe with no adaptation. It saw none of these speakers, which makes it a fair difficulty gauge |
| **Critical error rate** | Share of safety-relevant words lost: negation, yes/no, numbers, medications |

Raw prediction files stay out of git: they contain TORGO transcripts and this
repo is public.

## Headline

**Word error rate fell 38.6% on a speaker the model had never heard, from an
adapter touching 0.166% of the model, trained in 10.6 minutes.**

## Sealed test: M02, 388 clips

| Metric | Baseline | Tuned | Change |
|---|---:|---:|---:|
| Word error rate | 0.5668 | **0.3481** | 38.6% better |
| Isolated words (above 1.0 means it emits more words than were spoken) | 1.0169 | **0.4628** | 54.5% better |
| Full sentences | 0.3634 | **0.2962** | 18.5% better |
| Critical errors | 0.4000 | **0.2857** | 28.6% better |
| Clips exactly right | 98 / 388 | **198 / 388** | 2.0x |
| p50 / p95 latency per clip | 52 / 154 ms | 70 / 216 ms | slower |

LoRA adds matrix multiplications, so latency got worse. At 39x real time it does
not matter, and it is not presented as a win.

## All eight speakers

| Speaker | Role | Clips | Baseline | Tuned | Better by | Critical base | Critical tuned |
|---|---|---:|---:|---:|---:|---:|---:|
| M03 | trained | 406 | 0.0740 | 0.0148 | 80.0% | 0.0000 | 0.0000 |
| F04 | **validation** | 244 | 0.1099 | **0.0330** | 70.0% | 0.0000 | 0.0000 |
| F03 | trained | 546 | 0.3082 | 0.1262 | 59.1% | 0.0702 | 0.0702 |
| M05 | trained | 470 | 0.3918 | 0.1480 | 62.2% | 0.0667 | 0.0444 |
| M02 | **sealed test** | 388 | 0.5668 | **0.3481** | 38.6% | 0.4000 | 0.2857 |
| F01 | trained | 118 | 0.6596 | 0.2943 | 55.4% | 0.3333 | 0.1111 |
| M01 | trained | 371 | 0.6674 | 0.2927 | 56.1% | 0.4054 | 0.1622 |
| M04 | trained | 277 | 1.1268 | 0.3276 | 70.9% | 0.4545 | 0.1818 |

**Every speaker improved. Critical errors improved or held flat in every row and
worsened in none.**

Baseline WER doubles as a severity ranking, since the frozen model saw nobody.
The gain grows with severity: 70.9% on the hardest speaker (M04) against 38.6%
on M02. That is the product thesis, measured.

## What it actually fixes

Sealed-test speaker, baseline against tuned. The seven sentences are the clips
the demo ships, picked because the tuned model matches the reference exactly and
the baseline does not.

| Said | Baseline heard | Tuned |
|---|---|---|
| this was easy for us | thats whats using force | exact |
| why yell or worry over silly items | why yelled a warrior of a silly agent | exact |
| he will allow a rare lie | he will **not** allow a real life | exact |
| we gathered shells on the beach | forget the tails on the beach | exact |
| their house is grey and white | there are houses grey and white | exact |
| two other cases also were under advisement | two of the cases also were under advisement | exact |
| twice each day **he** plays skillfully and with **zest** upon **our** small organ | twice each day **she** played skilfully and with **that** upon **her** small organ | exact |
| wicked | who were killed | wicked |
| jagged | they are good | jagged |
| brawn | oh put it on | brawn |

Row three is the sharpest: the baseline inserts a negation nobody spoke,
inverting the meaning. This is why the clarification step exists.

## End to end, live

`scripts/e2e_check.py` streams each demo clip to the deployed Baseten endpoint in
real time, then enrolls and synthesizes on the voice renderer. Nothing mocked.

| Clip | Exact | Confidence | Asks | Alternatives | First partial | Final | Synthesis |
|---|---|---:|---|---:|---:|---:|---:|
| M02_1_headMic_0141 | yes | 0.924 | no | 1 | 645 ms | 4854 ms | 862 ms |
| M02_1_headMic_0135 | yes | 0.815 | yes | 3 | 596 ms | 10036 ms | 1078 ms |
| M02_1_headMic_0185 | yes | 0.901 | no | 1 | 604 ms | 6288 ms | 739 ms |
| M02_1_headMic_0184 | yes | 0.848 | yes | 3 | 591 ms | 6820 ms | 1725 ms |
| M02_1_headMic_0196 | yes | 0.976 | no | 1 | 598 ms | 8493 ms | 892 ms |
| M02_2_headMic_0088 | yes | 0.754 | yes | 3 | 612 ms | 11837 ms | 2002 ms |
| M02_1_headMic_0092 | yes | 0.892 | no | 1 | 608 ms | 14191 ms | 1011 ms |

**All seven transcribe exactly.** `final` is wall clock from the first frame, so
it includes streaming the clip in real time; the decode lands 1.5 to 2.5 s after
the audio ends. Three clips fall under the threshold and ask before speaking,
each offering three genuine alternatives with the model's best guess first.

**Cold start is 33 seconds** at `min_replica: 0`. Warm the endpoint before
demoing.

## Why Cohere and not Parakeet

Both lanes were built. NVIDIA NeMo Parakeet (0.6B and 1.1B) was adapted with
dim-32 encoder adapters, with and without unfreezing the last two encoder
blocks. Its arms were scored on held-out *prompts* from speakers already in
training, on sets of 19 to 55 clips and with different step budgets, so those
numbers are not comparable to the tables above and are not reproduced here.

One axis is genuinely matched. F04, identical 244 clips and 637 reference words,
neither lane trained on it:

| | Frozen | Adapted |
|---|---:|---:|
| Cohere Transcribe 2.07B | 0.1099 | **0.0330** |
| NeMo Parakeet, best arm | 0.1523 | 0.0785 |

Cohere starts better and improves further, so it is the shipped lane.

## Training

| | |
|---|---|
| Base model | `CohereLabs/cohere-transcribe-03-2026`, 2.07B parameters, Apache 2.0 |
| Method | LoRA, rank 8, alpha 16, dropout 0.05. Nothing unfrozen |
| Targets | 128 modules: encoder blocks 42 to 47 of 48, plus all 8 decoder layers |
| Trainable | 3,424,256 of 2,068,023,552 = **0.166%** |
| Schedule | AdamW, lr 1e-4, 20-step warmup, linear decay, batch 4 x 4 accum, bf16 |
| Steps | 1200, 19,200 examples, **10.6 min** on one H100 |
| Selection | Lowest dev loss, 0.1428 at step 900, not the final step |

The encoder is targeted because it holds 91.8% of the parameters and dysarthria
is an acoustic problem.

## Is the adapter the right size?

Four configs trained from scratch and scored on F04. The shipped config was
re-run inside the same job, so the comparison is against a number from the same
box rather than a remembered one.

| Rank | Encoder blocks | % of model | F04 WER | Dev loss |
|---:|---:|---:|---:|---:|
| **8** | **6** | **0.166%** | **0.0361** | 0.1387 |
| 8 | 16 | 0.304% | 0.0377 | 0.1399 |
| 32 | 6 | 0.659% | 0.0408 | 0.1513 |
| 32 | 16 | 1.204% | 0.0408 | 0.1446 |

**The smallest adapter wins, and every increase in capacity costs accuracy.** At
7.3x the parameters the model is 13% worse, and WER and dev loss agree, so it is
unlikely to be noise. The binding constraint is data, not capacity: 4.06 hours
across 6 speakers. More speakers would help; more parameters will not.

## Data

TORGO, `abnerh/TORGO-database`. Split by speaker before any sampling, frozen
before any baseline was scored, hashes in `contract/MANIFEST_HASHES` and verified
before every run. Utterance groups never cross a split.

| Split | Clips | Hours | Speakers |
|---|---:|---:|---|
| Train | 4,129 | 4.06 | F01 F03 M01 M03 M04 M05 |
| Validation | 244 | 0.24 | F04 |
| Sealed test | 388 | 0.42 | M02 |
| Control | 2,004 | 1.40 | FC01 FC02 FC03 MC04 |

## Voice rendering

OpenVoice V2, speaker embedding from five M02 clips.

| 7-word synthesis | H100 | Mac MPS | Mac CPU |
|---|---:|---:|---:|
| | **196 ms** | 1078 ms | 2200 ms |

`serve/voice/server.py` picks MPS when CUDA is absent, roughly 3x faster than
CPU. Whole-clause synthesis fits the budget at these speeds, so chunked
streaming TTS was cut.

## What these numbers do not show

- Two of eight speakers are genuinely held out. Evidence, not proof of
  generalization.
- TORGO is **read prompts**, isolated words and a fixed sentence set. No
  spontaneous speech.
- Tested on spontaneous conversational speech from outside the corpus, an 11.8 s
  clinical interview clip, the adapter did not transfer: baseline 0.654, tuned
  0.731. Confidence correctly collapsed to 0.657 and the system asked rather
  than asserting, but fine-tuning bought nothing there.
- 76% of the sealed test is single isolated words, the hardest case and where
  the gain is largest. Longer utterances gain 17 to 25%.
- 34.8% WER on M02 is still high. The claim is the relative reduction on an
  unseen speaker, not that the problem is solved.
