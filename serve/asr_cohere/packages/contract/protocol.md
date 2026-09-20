# Wire protocols

Normative. Both ASR lanes implement Protocol 1 identically, so promoting the
winner is a change to `ASR_WS_URL` and nothing else. `app/` speaks only these
three and never imports NeMo, Transformers, PEFT, or a checkpoint.

## Protocol 1: ASR stream, `WS /v1/stream`

Client to server:

```json
{"type": "audio", "seq": 12, "pcm16_b64": "..."}
{"type": "end"}
```

`pcm16_b64` is base64 of signed 16-bit little-endian PCM, 16 kHz, mono.
`seq` starts at 0 and increments by 1. A gap means dropped audio; the server
logs it and continues.

Server to client:

```json
{"type": "partial", "text": "i need my", "stable_prefix_len": 9,
 "confidence": 0.62, "t_ms": 340}

{"type": "final", "text": "i need my medication",
 "confidence": 0.91, "latency_ms": 820, "model_id": "...",
 "candidates": [{"text": "i need my medication", "score": -1.2},
                {"text": "i need my medicine",  "score": -2.7}]}

{"type": "error", "message": "..."}
```

- `stable_prefix_len` is a count of characters in `text` the server promises not
  to revise. The client may send that prefix onward for synthesis; anything past
  it may still change.
- `confidence` comes from `contract/confidence.py` in both lanes. Never from
  NeMo's packaged confidence utility, which is broken on TDT.
- `candidates` is required on `final` and is what the verifier chooses among. At
  least one entry. `score` is a log-probability, higher is better.
- `latency_ms` is steady state and excludes cold start.

## Protocol 2: Voice renderer

```text
POST /v1/enroll      {"wavs_b64": [...]}            -> {"voice_id": "..."}
WS   /v1/synthesize  {"text": "...", "voice_id": "...",
                      "pace": 0.95, "pitch_contour": [...]}
                     -> binary PCM/Opus frames, then {"type": "done"}
```

Enrollment runs once per consented session and the embedding is cached.
Generated health-related content is never cached.

## Protocol 3: Verifier

Stateless, strict JSON in and out.

```json
in:  {"candidates": [...], "confidence": 0.41,
      "topic": "medication", "vocabulary": [...]}
out: {"action": "accept" | "clarify" | "abstain",
      "text": "...", "choices": ["...", "..."]}
```

Enforced in code, not in the prompt: a returned `text` must be byte-identical to
one of the supplied candidates. The verifier cannot emit free text, and negation,
names, numbers, and medication terms are never substituted.
