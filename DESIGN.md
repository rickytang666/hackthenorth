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

## Phase 0: the foundation, built jointly before anyone branches

Two lanes can only run independently if the seams between them already exist as running code. Prose seams drift silently; a committed evaluator and a running mock server do not. Phase 0 builds those seams with **one person at the keyboard**, because three agents on an empty repo invent three answers to the same dozen questions and git merges all three cleanly.

The other person is not idle and is not coding. Provisioning is genuinely parallel because it is downloads and approvals, and every item on it is fatal if discovered at hour 8.

**Keyboard lane, in this order:**

1. `contract/normalize.py`, `contract/predictions.py`, `contract/evaluate.py`, `contract/critical_terms.txt`
2. `data/prepare_torgo.py`, run once. Publish the five manifest SHA-256 hashes in `contract/MANIFEST_HASHES`
3. `contract/decode.py`, one decode loop taking a `transcribe(paths) -> list[str]` callable, so both lanes emit byte-comparable prediction files without sharing model code
4. `contract/protocol.md` plus `contract/mock_asr.py`, a protocol-correct fake that replays a fixture JSONL on a timer
5. `serve/_template/`, one Truss that already speaks Protocol 1 against a stub model, copied by both serving directories
6. `bench/latency.py`, driving Protocol 1 and writing the scorecard row
7. Finish `.gitignore`

**Provisioning lane, same hour:** Baseten account, CLI, and H100 quota confirmed; `HF_TOKEN` working and the gated `CohereLabs/cohere-transcribe-03-2026` conditions accepted; TORGO downloaded (1.56 GB); OpenVoice V2 weights downloaded; one consented enrollment recording captured; verifier Model API key working; TORGO license checked for third-party cloud processing.

**Phase 0 exits when all four are true, and not on a clock:**

- `contract/evaluate.py` runs on synthetic prediction files and prints the full scorecard table
- `contract/mock_asr.py` streams protocol-correct partials to a stub page and `bench/latency.py` reports a number from it
- Both people have independently reproduced the five manifest hashes
- Both people have a trainable checkpoint loaded and one backward pass completed locally

Only then do the two lanes branch. Skipping any of the four moves the cost to the hour-5 gate, where it is unrecoverable.

## Service boundaries: why the lanes stay independent after Phase 0

Three rules, all enabled by Phase 0:

1. **The app talks HTTP and WebSocket only.** It never imports NeMo, Transformers, PEFT, or a checkpoint. Switching mock to Parakeet to Cohere to the promoted winner is one environment variable, `ASR_WS_URL`.
2. **Both model lanes serve the identical protocol from the moment they branch**, each wrapping their own model in a copy of `serve/_template/`. The hour-5 gate then picks a URL, not an integration task.
3. **The mock exists before either model does**, so product work never waits on a training job.

The dependency graph after Phase 0 has exactly three sync points: the promotion gate, the sealed test decode, and the freeze. Everything else is lane-local.

## The data contract

`contract/` is the only directory both people import from. It is frozen at the end of Phase 0. Changing it afterwards is a team announcement, not a quiet commit, because both lanes' results become incomparable the moment it drifts.

```text
contract/
  normalize.py        # THE text normalizer. both lanes import this one function
  predictions.py      # prediction row schema + writer
  evaluate.py         # WER, per-speaker WER, critical-error rate, RTF
  decode.py           # shared decode loop, pluggable transcribe callable
  critical_terms.txt  # frozen safety slice: negation, yes/no, numbers, names, meds
  protocol.md         # the three wire protocols below, normative
  mock_asr.py         # protocol-correct fake ASR server, fixture-driven
  MANIFEST_HASHES     # SHA-256 of the five frozen manifests
```

### Manifest row

Produced once by `data/prepare_torgo.py`, then immutable. Both lanes consume byte-identical files and assert the same hashes.

```text
audio_filepath      str    absolute path to 16 kHz mono PCM16 WAV
text                str    training/eval target, from TORGO transcription
duration            float  seconds
speaker_id          str    e.g. F04, M02
speech_status       str    dysarthria | control
mic                 str    headMic | arrayMic | unknown
utterance_id        str    unique per recording
utterance_group     str    same value for paired mic views of one utterance
```

Five frozen files: `torgo_dys_train.jsonl`, `torgo_dys_dev.jsonl`, `torgo_dys_test.jsonl`, `torgo_clean_replay.jsonl`, `torgo_clean_eval.jsonl`.

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

### Protocol 2: Voice renderer

```text
POST /v1/enroll      {"wavs_b64": [...]}            -> {"voice_id": "..."}
WS   /v1/synthesize  {"text": "...", "voice_id": "...",
                      "pace": 0.95, "pitch_contour": [...]}
                     -> binary PCM/Opus frames, then {"type": "done"}
```

Enrollment runs once per consented session and the embedding is cached. Generated health-related content is never cached. Segment at punctuation or a 300 to 500 ms pause and cross-fade adjacent chunks.

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

Every directory below Phase 0 has exactly one owner. Adding a feature means adding a file, never editing a shared list.

```text
contract/              built jointly in Phase 0, then frozen
data/prepare_torgo.py  runs once, produces the five manifests
serve/_template/       Phase 0, copied by both serving directories
train/parakeet/        Person A only
train/cohere/          Person B only
serve/asr_parakeet/    Person A only, speaks Protocol 1
serve/asr_cohere/      Person B only, speaks Protocol 1
serve/voice/           Person B only, speaks Protocol 2
app/                   Person B only, UI + verifier client
bench/                 Person A only, latency harness + scorecard writer
results/               gitignored, JSONL only
```

## Model lanes

Both lanes fine-tune from a strong pretrained English checkpoint on the identical dysarthric TORGO train split, with the same normalizer, evaluator, and H100 class.

| Lane | Base | Method | Owner |
|---|---|---|---|
| Parakeet | `nvidia/parakeet-tdt-0.6b-v2` | NeMo fine-tuning, BF16, ~800 steps | A |
| Cohere | `CohereLabs/cohere-transcribe-03-2026` | PEFT LoRA r=8 on decoder, encoder frozen, ~600 steps | B |
| Reference | `nvidia/parakeet-tdt-1.1b` | frozen, decode only | A |

The NeMo `run.sh` and the PEFT config are transcribed from [voicebridge-plan.md](voicebridge-plan.md). Treat the numbers as starting values: run 50 to 100 steps, record examples per second and peak memory, then cap steps at roughly three to five effective passes over the small split and stop early when dev WER flattens. Cap audio at 30 seconds and bucket by length.

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
| OpenVoice chunked streaming not ready | Synthesize the whole confirmed sentence, then play |
| OpenVoice enrollment fails | Generic pretrained voice, stated honestly on stage |
| Any live service down at the booth | Fixture replay through `contract/mock_asr.py`, local file on the presenting laptop |

## Evaluation

**Recognition:** dev and test speaker WER and CER, reported separately for isolated words and restricted sentences. Critical error rate. Clarification rate and recovery rate after clarification.

**Latency:** time to first partial, time to first recovered audio, p50 and p95 final latency, real-time factor, peak VRAM. Cold start reported separately, never folded into steady state.

**Voice identity:** speaker-embedding cosine similarity, F0 contour correlation, speaking-rate and pause-duration error, plus blinded listener preference. Cosine similarity alone is insufficient; a voice can score as the same speaker while losing accent and rhythm.

Targets: first recovered audio under 1.5 s, RTF under 0.5, at most 250 ms between synthesized clauses.

## Risks

- TORGO has eight dysarthric speakers. One dev and one test speaker cannot establish generalization. Say this in the pitch rather than being caught on it.
- Cohere's documented production path is offline or vLLM, not a demonstrated cache-aware streaming encoder. If Cohere wins on WER it may still lose on latency.
- The Cohere collator and LoRA target-module suffixes are revision-specific. Print `model.named_modules()` and unit-test one batch and one backward pass before launching the paid job.
- Confirm TORGO's academic non-profit license permits third-party cloud processing before mounting audio on Baseten.
- Voice enrollment audio is sensitive personal data. Explicit consent, explicit deletion, no secrets or keys in logs. The repo goes public at submission.
