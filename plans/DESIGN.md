# VoiceBridge technical design

Real-time speech recovery for dysarthric speech that speaks the result back in the user's own voice.

Motivation, citations, prior art, and the demo narrative live in [voicebridge-plan.md](voicebridge-plan.md). This file is the engineering agreement. Who builds what, and when, is in [OWNERSHIP.md](OWNERSHIP.md).

**Freeze: Sunday 04:00 EDT.** Final code and Devpost edits close Sunday 08:00 EDT. Judging starts 09:30 EDT at the venue.

## The one sentence the build is measured against

A person speaks, the app shows recovered text, asks a two-choice question when evidence is weak, and plays the confirmed sentence back in that person's voice, in under 1.5 seconds to first audio.

## Non-goals

- No TTS foundation model training. OpenVoice V2 is used pretrained.
- No aphasia or apraxia claims. Dysarthric read speech only.
- No clinical validation. TORGO is read prompts, not open conversation.
- No new datasets. SAP is a post-hackathon scale-up.
- No FP8 training or quantization-aware fine-tuning. Post-training quantization only, behind an accuracy gate.

## Architecture

```text
browser mic (PCM16 @ 16 kHz)
      |
      |  WebSocket  /v1/stream          <- ASR PROTOCOL (frozen in Phase 0)
      v
ASR service  (one of: mock | parakeet | cohere)
  ring buffer, overlapping windows, stable-prefix tracking
      |
      +-- confidence >= tau ---------------------> confirmed text
      |
      +-- confidence <  tau --> verifier (hosted Model API, strict JSON)
                                  accept | clarify | abstain
                                            |
                                     user taps a choice
                                            v
                                      confirmed text
      |
      |  WebSocket  /v1/synthesize      <- RENDERER PROTOCOL (frozen in Phase 0)
      v
voice service (OpenVoice V2, cached speaker embedding)
      |
      v
Opus/PCM frames -> browser playback
```

## The phases

| Phase | Wall clock | Name | Ends when |
|---|---|---|---|
| **0** | 14:30 to 15:45 | Foundation, joint, one keyboard | All six exit gates pass. **Branch here** |
| **1** | 15:45 to 18:45 | Parallel lanes: train and build | Two tuned candidates plus a working mock-driven product shell |
| **2** | 18:45 to 19:45 | Dev decode | Promotion table filled |
| **3** | 19:45 to 20:45 | Promotion and integration | **Product gate:** speech, text, confirmation, personal voice |
| **4** | 20:45 to 22:45 | Optimize and seal | Test-speaker number locked, one approved serving config |
| **5** | 22:45 to 01:45 | Verify and harden | Scorecard populated, failures documented, demo stable |
| **6** | 01:45 to 04:00 | Rehearse and buffer | Everything frozen at 04:00 |
| **7** | 04:00 to 08:00 | Submit | Video, Devpost, submitted before 08:00 |

Phase 1 is the only phase where the two people work independently. Everything before it is joint, and everything after it has at least one sync point. The 200-step kill-rule check sits inside Phase 1 at ~17:00.

Phases 0, 3, 5 and 6 are phase boundaries in the handbook sense: regenerate `README.md`, compact agent context back toward 40-60%, and re-read the demo script.

## Phase 0: the foundation, built jointly before anyone branches

Two lanes can only run independently if the seams between them already exist as running code. Prose seams drift silently; a committed evaluator and a running mock server do not. Phase 0 builds those seams with **one person at the keyboard**, because three agents on an empty repo invent three answers to the same dozen questions and git merges all three cleanly.

**Phase 0 is written on one machine.** Every file in the keyboard lane is generated once, on that machine, and reaches the other person through git. The other person does not write any of it. What they must do in the same hour is get their own machine and their own accounts to the point where they can pull that work and immediately use it, which is the readiness checklist in [OWNERSHIP.md](OWNERSHIP.md).

The other person is not idle and is not coding. Provisioning is genuinely parallel because it is downloads and approvals, and every item on it is fatal if discovered at hour 8.

**Keyboard lane, in this order:**

1. `contract/normalize.py`, `contract/predictions.py`, `contract/evaluate.py`, `contract/critical_terms.txt`
2. `data/prepare_torgo.py`, run once. Publish the five manifest SHA-256 hashes in `contract/MANIFEST_HASHES`
3. `contract/decode.py`, one decode loop taking a `transcribe(paths) -> list[str]` callable, so both lanes emit byte-comparable prediction files without sharing model code
4. `contract/confidence.py`, the shared token-log-probability scorer, since NVIDIA's TDT confidence utility is broken (see "Where `confidence` comes from")
5. `contract/protocol.md` plus `contract/mock_asr.py`, a protocol-correct fake that replays a fixture JSONL on a timer
6. `serve/_template/`, one Truss that already speaks Protocol 1 against a stub model, copied by both serving directories
7. `bench/latency.py`, driving Protocol 1 and writing the scorecard row
8. Finish `.gitignore`, the last moment it is free to edit

**Provisioning lane, same hour:** Baseten account, CLI, and H100 quota confirmed; `HF_TOKEN` working and the gated `CohereLabs/cohere-transcribe-03-2026` conditions accepted; TORGO downloaded (1.56 GB); OpenVoice V2 weights downloaded; one consented enrollment recording captured; verifier Model API key working; TORGO license checked for third-party cloud processing. Then, holding the weights already, **time one 7-word OpenVoice synthesis and write the number down**, which decides whether streaming TTS is built at all.

**Phase 0 exits when all six are true, and not on a clock:**

- `contract/evaluate.py` runs on synthetic prediction files and prints the full scorecard table
- `contract/mock_asr.py` streams protocol-correct partials to a stub page and `bench/latency.py` reports a number from it
- Both people have independently reproduced the five manifest hashes
- Both people have a trainable checkpoint loaded and one backward pass completed locally
- `contract/confidence.py` returns a score on a real decode
- One 7-word OpenVoice synthesis is timed and the number written down

Only then do the two lanes branch. Skipping any of the six moves the cost to the promotion gate, where it is unrecoverable.

## Service boundaries: why the lanes stay independent after Phase 0

Three rules, all enabled by Phase 0:

1. **The app talks HTTP and WebSocket only.** It never imports NeMo, Transformers, PEFT, or a checkpoint. Switching mock to Parakeet to Cohere to the promoted winner is one environment variable, `ASR_WS_URL`.
2. **Both model lanes serve the identical protocol from the moment they branch**, each wrapping their own model in a copy of `serve/_template/`. The hour-5 gate then picks a URL, not an integration task.
3. **The mock exists before either model does**, so product work never waits on a training job.

The dependency graph after Phase 0 has exactly four sync points: the 200-step kill-rule check, the promotion gate, the sealed test decode, and the freeze. Everything else is lane-local.

## The data contract

`contract/` is the only directory both people import from. It is frozen at the end of Phase 0. Changing it afterwards is a team announcement, not a quiet commit, because both lanes' results become incomparable the moment it drifts.

```text
contract/
  normalize.py        # THE text normalizer. both lanes import this one function
  predictions.py      # prediction row schema + writer
  evaluate.py         # WER, per-speaker WER, critical-error rate, RTF
  decode.py           # shared decode loop, pluggable transcribe callable
  confidence.py       # shared token-log-prob scorer + calibrated threshold
  critical_terms.txt  # frozen safety slice: negation, yes/no, numbers, names, meds
  protocol.md         # the three wire protocols below, normative
  mock_asr.py         # protocol-correct fake ASR server, fixture-driven
  MANIFEST_HASHES     # SHA-256 of the five frozen manifests
```

### Manifest row

Produced once by `data/prepare_torgo.py`, then immutable. Both lanes consume byte-identical files and assert the same hashes.

**Paths are relative, never absolute.** Phase 0 runs on one machine, training runs on Baseten, and the second person works on a third machine, so an absolute path baked into a manifest is wrong on two of the three. Every row stores a path relative to a data root and `contract/manifest.py` resolves it against `$VOICEBRIDGE_DATA` at load time. This is also what makes the shared hash meaningful: the same manifest hashes identically on every machine, which it cannot do if the paths differ.

```text
audio_filepath      str    path RELATIVE to $VOICEBRIDGE_DATA, 16 kHz mono PCM16 WAV
text                str    training/eval target, from TORGO transcription
duration            float  seconds
speaker_id          str    e.g. F04, M02
speech_status       str    dysarthria | control
mic                 str    headMic | arrayMic | unknown
utterance_id        str    unique per recording
utterance_group     str    same value for paired mic views of one utterance
```

Five frozen files, as actually built:

| File | Rows | Hours | Speakers |
|---|---:|---:|---|
| `torgo_dys_train.jsonl` | 4,129 | 4.06 | F01 F03 M01 M03 M04 M05 |
| `torgo_dys_dev.jsonl` | 244 | 0.24 | F04 |
| `torgo_dys_test.jsonl` | 388 | 0.42 | M02 |
| `torgo_clean_replay.jsonl` | 412 | 0.35 | FC01 FC02 |
| `torgo_clean_eval.jsonl` | 1,592 | 1.05 | FC03 MC04 |

Train keeps both mic views; dev, test and the control sets are `headMic` only.

Split by speaker, frozen before any baseline is scored: six dysarthric speakers train, `F04` dev, `M02` test. Both mic views may train; only `headMic` is scored. `utterance_group` never crosses a split. Controls are never positive training examples: at most 10% replay, and only if the clean-speech regression gate fails.

### Prediction row

Every decode from either lane writes this shape. The evaluator joins on `audio_filepath`.

```text
audio_filepath  str
speaker_id      str
text            str    reference
prediction      str
model_id        str
latency_ms      float  steady state, excludes cold start
```

Filenames are fixed so the gate table fills itself: `baseline_parakeet.jsonl`, `tuned_parakeet.jsonl`, `comparison_parakeet_1p1b.jsonl`, `baseline_cohere.jsonl`, `tuned_cohere.jsonl`.

### Protocol 1: ASR stream, `WS /v1/stream`

Client sends:

```json
{"type": "audio", "seq": 12, "pcm16_b64": "..."}
{"type": "end"}
```

Server sends:

```json
{"type": "partial", "text": "i need my", "stable_prefix_len": 9,
 "confidence": 0.62, "t_ms": 340}
{"type": "final", "text": "i need my medication",
 "confidence": 0.91, "latency_ms": 820, "model_id": "...",
 "candidates": [{"text": "i need my medication", "score": -1.2},
                {"text": "i need my medicine",  "score": -2.7}]}
```

`stable_prefix_len` is the character count the server promises not to revise. The app sends a clause to the renderer only once it is inside the stable prefix or the user has confirmed it, otherwise the system speaks a word and immediately contradicts itself.

### Where `confidence` comes from

The `confidence` field above is load-bearing: it is the sole trigger for asking instead of guessing, which is the product's central claim. Without it you either ask every time, which kills the demo rhythm, or never ask, which is the hallucination failure being pitched against.

**NVIDIA's packaged confidence utility is broken on TDT models.** It raises `IndexError` on Parakeet-TDT while working on the RNNT and CTC variants, because TDT predicts a token and a duration and skips blank frames, violating the per-frame indexing that utility assumes. The bug has been open and stale since 2024, so do not plan around it being fixed.

What is broken is the wrapper, not the model: the raw token probabilities are still available. Both lanes compute their own score, identically, in `contract/confidence.py`:

- sequence confidence = mean token log-probability, exponentiated
- word-level flag = minimum token log-probability within the word
- calibrate the accept/clarify threshold on the dev speaker, since raw probabilities run overconfident and 0.6 must actually mean about 60% correct

This is roughly twenty lines and it is a Phase 0 exit gate, not lane work, because both lanes must threshold identically or the clarification rates in the promotion table are not comparable.

### Protocol 2: Voice renderer

```text
POST /v1/enroll      {"wavs_b64": [...]}            -> {"voice_id": "..."}
WS   /v1/synthesize  {"text": "...", "voice_id": "...",
                      "pace": 0.95, "pitch_contour": [...]}
                     -> binary PCM/Opus frames, then {"type": "done"}
```

Enrollment runs once per consented session and the embedding is cached. Generated health-related content is never cached.

**Two different things are called streaming here, and only one is demo-critical.**

*Streaming ASR*, meaning text appearing word by word as the person speaks, is what a judge actually watches. It is already in Protocol 1 and it ships.

*Streaming TTS*, meaning audio starting before synthesis finishes, is not shipped by default. OpenVoice V2 has no native streaming path; the sub-second figures people report come from wrapping it in a pipeline such as Pipecat or LiveKit. Building chunked synthesis buys maybe 200 to 400 ms and costs chunk-boundary artifacts, cross-fade logic, and buffer management, which is the single most likely thing to stutter live.

A confirmed clause is 5 to 8 words. **Time one 7-word synthesis in Phase 0.** Under about 800 ms, synthesize whole confirmed clauses and delete streaming TTS from the plan. Only if that measurement misses the 1.5 s first-audio budget do you segment at punctuation or a 300 to 500 ms pause and cross-fade adjacent chunks.

### Protocol 3: Verifier

Hosted Model API, stateless, strict JSON in and out. It receives only the candidate set, confidence, an approved vocabulary and topic, and cannot emit free text.

```json
in:  {"candidates": [...], "confidence": 0.41,
      "topic": "medication", "vocabulary": [...]}
out: {"action": "accept" | "clarify" | "abstain",
      "text": "...", "choices": ["...", "..."]}
```

Enforced in code, not in the prompt: a returned `text` must be one of the supplied candidates. Negation, names, numbers, and medication terms are never substituted.

## Repo layout

```text
hackthenorth/
|
|-- contract/                  the shared ruler. built in Phase 0, then frozen
|   |-- normalize.py             THE text normalizer, both lanes import this
|   |-- predictions.py           prediction row schema + writer
|   |-- evaluate.py              scoring: WER, critical errors, RTF
|   |-- decode.py                shared decode loop, pluggable transcribe()
|   |-- confidence.py            shared scorer, NVIDIA's TDT utility is broken
|   |-- critical_terms.txt       frozen safety slice
|   |-- protocol.md              the three wire protocols, normative
|   |-- mock_asr.py              fake ASR server, lets app/ start immediately
|   `-- MANIFEST_HASHES          SHA-256 of the five frozen manifests
|
|-- data/
|   `-- prepare_torgo.py       runs once, emits the five manifests
|
|-- train/
|   |-- parakeet/              NeMo fine-tune, Baseten TrainingProject
|   `-- cohere/                PEFT LoRA, Baseten TrainingProject
|
|-- serve/
|   |-- _template/             Truss that already speaks Protocol 1. copy, never edit
|   |-- asr_parakeet/          Protocol 1
|   |-- asr_cohere/            Protocol 1
|   `-- voice/                 Protocol 2, OpenVoice V2
|
|-- app/                       browser UI + verifier client. HTTP/WS only
|-- bench/                     latency harness, scorecard writer
|-- results/                   gitignored, prediction JSONL
|
|-- plans/
|   |-- DESIGN.md              this file
|   |-- OWNERSHIP.md           who owns what, when
|   `-- voicebridge-plan.md    motivation, citations, Devpost source
|
|-- pyproject.toml             one person installs all dependencies
|-- package.json               same
|-- .gitignore
`-- README.md                  orientation, regenerated at phase boundaries
```

`app/` never imports NeMo, Transformers, PEFT, or a checkpoint. It knows one environment variable, `ASR_WS_URL`.

## Model lanes

Both lanes fine-tune from a strong pretrained English checkpoint on the identical dysarthric TORGO train split, with the same normalizer, evaluator, and H100 class.

| Lane | Base | Method | Owner |
|---|---|---|---|
| Parakeet | `nvidia/parakeet-tdt-0.6b-v2` | NeMo fine-tuning, BF16, ~800 steps | A |
| Cohere | `CohereLabs/cohere-transcribe-03-2026` | PEFT LoRA r=8 on the **top 6 encoder blocks plus the decoder**, ~400 steps | B |
| Reference | `nvidia/parakeet-tdt-1.1b` | frozen, decode only | A |

**Why the Cohere LoRA reaches into the encoder.** [voicebridge-plan.md](voicebridge-plan.md) froze the encoder and adapted the decoder only. Cohere Transcribe puts over 90% of its 2B parameters in the encoder and keeps a deliberately lightweight decoder, and dysarthria is an acoustic problem, so decoder-only adaptation would likely move very little and Lane B would lose the bake-off for a reason unrelated to the model.

The cost is backpropagation. Today the encoder runs forward with no gradient; adapting a block means backprop through it and everything above it, and backward costs roughly twice a forward. Full-encoder LoRA is about **2.3x** the step time of decoder-only. Top 6 blocks is about **1.3 to 1.5x**, which is why the step budget drops from 600 to 400 and stays inside the same 3 hours. Selective layer adaptation also has precedent: Shor et al., cited in the plan, found adapting selected layers outperformed full fine-tuning. Confirm the real multiplier and the activation-memory headroom in the 50-to-100-step smoke test before committing the paid job.

The NeMo `run.sh` and the PEFT config are transcribed from [voicebridge-plan.md](voicebridge-plan.md). Treat the numbers as starting values: run 50 to 100 steps, record examples per second and peak memory, then cap steps at roughly three to five effective passes over the small split and stop early when dev WER flattens. Cap audio at 30 seconds and bucket by length.

## The 200-step kill rule, before the gate

The promotion gate is at hour 5. Without an earlier checkpoint, a lane that is visibly failing at hour 2 still gets babysat for three more hours before anyone is permitted to say so.

Both jobs already evaluate every 100 to 200 steps and Phase 0 already produced each model's frozen baseline WER, so the comparison costs nothing extra.

**At 200 steps, if the tuned model is not beating its own untuned baseline on the dev speaker, stop the job and move that person onto the surviving lane.**

This works because a flat early curve is almost never slow learning. It is a wiring bug: LoRA pointed at module names that do not exist on the pinned revision, labels masked wrong, the encoder accidentally frozen, a learning rate off by 10x. None of those improve with more steps. At batch 4 with 4x accumulation, 200 steps is about 3,200 samples, a real fraction of an epoch on a split this small, so the signal is trustworthy.

Dropping a lane costs one tuned model. You still report both frozen baselines plus one tuned model, which is a valid comparison, and you recover roughly 3 hours of a person plus 1.5 to 2 H100-hours, since the killed job is already part-way through its 3-hour budget.

## Promotion gate

Filled from dev-speaker predictions only. The test speaker stays sealed.

| Metric | Parakeet frozen | Parakeet tuned | Cohere frozen | Cohere LoRA |
|---|---:|---:|---:|---:|
| Held-out dev-speaker WER | | | | |
| Isolated-word WER | | | | |
| Restricted-sentence WER | | | | |
| Critical error rate | | | | |
| Typical-speech WER | | | | |
| p50 / p95 final latency | | | | |
| Real-time factor | | | | |

Decision rule, predeclared:

1. Reject any candidate that adds a critical error on the frozen safety slice.
2. Reject any candidate regressing typical-speech WER by more than 1.0 absolute point.
3. Among survivors, lowest dev-speaker WER wins.
4. Within 1.0 absolute WER, fewer critical errors wins. Still tied, lower p95 wins.

Then set `ASR_WS_URL` to the winner and decode `torgo_dys_test.jsonl` exactly once, with the winner and its own frozen baseline. That single number is the headline. Preserve both baselines and both adapters so the comparison stays reproducible whichever model wins.

TORGO gives one validation speaker, so this is not speaker-macro WER and must not be described as such.

## Serving optimization, after the gate

Applied to the frozen winner only, each behind its own accuracy gate, each independently revertible.

- Overlapping rolling windows and stable-prefix tracking. Conservative by default; true encoder KV reuse is claimed only if the architecture demonstrably exposes valid chunk state.
- FP8 post-training quantization on dense projections, keeping layer norm, softmax, and logits in BF16. Accept only if dev WER rises at most 1.0 absolute point and adds no critical error. FP8 may save VRAM without helping latency at batch size one; ship BF16 if so.
- Length bucketing, capped dynamic-batch wait, preallocated tensors, pinned memory, SDPA or FlashAttention.
- `torch.compile` and CUDA Graphs only after input shapes stabilize.

## Fallback ladder

Each rung is reachable in under ten minutes and each is demoable.

| If this fails | Fall back to |
|---|---|
| Both adapters miss the gate | Better frozen model, unchanged product path |
| FP8 misses its accuracy gate | BF16 winner |
| Streaming state reuse drifts | Fixed overlapping windows |
| First-partial latency gate unreachable | `nvidia/nemotron-speech-streaming-en-0.6b`, cache-aware by design |
| First-audio budget missed with whole-clause synthesis | Add chunked streaming TTS, measured in Phase 0 before it is built |
| OpenVoice enrollment fails | Generic pretrained voice, stated honestly on stage |
| Any live service down at the booth | Fixture replay through `contract/mock_asr.py`, local file on the presenting laptop |

## Evaluation

### In plain terms

Everything reduces to one question asked four ways: **did the model type what the person actually said, and how fast?**

We hold out one speaker the model never trained on, play their recordings through it, and compare the typed output to the known correct sentence. Four numbers come out:

1. **How many words it got wrong.** Standard word error rate. 10% means one word in ten is wrong. Reported separately for single words and full sentences, because TORGO repeats a small word list and a model can look good by memorizing it.
2. **How many words that matter it got wrong.** Turning "do not give her the insulin" into "do give her the insulin" is one word wrong out of six, but it is the only error that could hurt someone. We keep a frozen list of those terms (no, yes, numbers, names, medications) and count errors on them separately. A model that improves overall but adds one of these is rejected.
3. **Whether it got worse at ordinary speech.** Fine-tuning hard on six dysarthric speakers can make a model forget everyone else. We keep a set of control recordings the model never trained on and check the score did not fall off a cliff.
4. **How fast.** How long until the first words appear, how long until the person hears their own voice, and the slow cases (p95) rather than the average, because the slow cases are what a judge notices.

### Why it is built before branching, not after

Word error rate depends on how you clean up the text first. If one lane strips punctuation and lowercases and the other does not, "Don't." and "dont" count as a mismatch and the score is wrong. Two lanes with two cleanup functions produce two numbers that cannot be compared, and at the hour-5 gate you would be choosing between normalizers instead of models with no time to re-run anything.

So `contract/normalize.py` and `contract/evaluate.py` are written once, by one person, before either lane exists. Both lanes import them. Neither lane may fork them.

### The numbers we report

| Category | What we measure |
|---|---|
| Recognition | Dev and test speaker WER and CER, split into isolated words and sentences |
| Safety | Critical error rate on the frozen term list |
| Abstention | How often it asked for clarification, and how often that recovered the sentence |
| Latency | Time to first partial text, time to first recovered audio, p50 and p95 final, RTF, peak VRAM. Cold start reported separately, never folded in |
| Voice identity | Speaker-embedding cosine similarity, F0 contour correlation, speaking-rate and pause error, plus blinded listener preference |

Cosine similarity alone is insufficient: a voice can score as the same speaker while losing its accent and rhythm entirely, which is the thing we claim to preserve.

Targets: first recovered audio under 1.5 s, RTF under 0.5, at most 250 ms between synthesized clauses.

## Risks

- TORGO has eight dysarthric speakers. One dev and one test speaker cannot establish generalization. Say this in the pitch rather than being caught on it.
- Cohere's documented production path is offline or vLLM, not a demonstrated cache-aware streaming encoder. If Cohere wins on WER it may still lose on latency.
- The Cohere collator and LoRA target-module suffixes are revision-specific. Print `model.named_modules()` and unit-test one batch and one backward pass before launching the paid job. Assert that the intended top encoder blocks are trainable and the lower ones are not.
- Cohere Transcribe is Apache 2.0 but the repo is gated behind a contact-information click-through, so it costs two minutes rather than an approval queue. Pull from `CohereLabs/cohere-transcribe-03-2026` directly: the third-party ONNX, CoreML and GGUF mirrors are CC-BY-NC and non-commercial.
- Confirm TORGO's academic non-profit license permits third-party cloud processing before mounting audio on Baseten.
- Voice enrollment audio is sensitive personal data. Explicit consent, explicit deletion, no secrets or keys in logs. The repo goes public at submission.
