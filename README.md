# VoiceBridge

Real-time speech recovery for dysarthric speech that speaks the result back in
the person's own voice. Track: Baseten.

Measured results: [RESULTS.md](RESULTS.md).

Plans in [plans/](plans/): [DESIGN.md](plans/DESIGN.md) is the technical
agreement, [OWNERSHIP.md](plans/OWNERSHIP.md) says who builds what and when.

## Run it

```bash
uv sync
source env.sh          # VOICEBRIDGE_DATA defaults to ~/voicebridge-data, outside the repo
```

One-time data build (deterministic; hashes must match `contract/MANIFEST_HASHES`):

```bash
uv run hf download abnerh/TORGO-database --repo-type dataset --local-dir "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m data.prepare_torgo --source "$VOICEBRIDGE_DATA/torgo-dataset"
uv run python -m contract.manifest
```

The demo, three processes:

```bash
uv run python -m contract.mock_asr --port 8765          # or serve.local_runner for a real model
uv run python -m app.serve --port 5173                  # then open http://127.0.0.1:5173
cd serve/voice && uv run --no-sync python server.py --reference <wav>...
```

## Map

| Path | What |
|---|---|
| `contract/` | The shared ruler: manifests, normalizer, evaluator, confidence, protocols, mock ASR |
| `data/prepare_torgo.py` | Builds the five manifests. Deterministic, run once |
| `train/cohere/` | LoRA lane: pre-flight, collator, training, decode |
| `serve/_template/` | Truss speaking Protocol 1. Copy it, never edit it |
| `serve/asr_cohere/` | Cohere behind Protocol 1 |
| `serve/voice/` | OpenVoice V2 behind Protocol 2. Isolated env, `numpy<2` |
| `app/` | Browser UI plus the deterministic verifier |
| `bench/`, `serve/local_runner.py` | Latency harness, and running any Truss model locally |

## Things that will bite you

These were each found the hard way; the comments in the code say more.

- **Cohere does no language identification.** Without the decoder prompt prefix
  from `train/cohere/model.py` it transcribes dysarthric English into Arabic
  script at 0.9 confidence. Training and inference must use the same prefix.
- **The model does not shift labels.** Logits align 1:1 with `decoder_input_ids`,
  so the collator does teacher forcing itself.
- **The LoRA targets are Fast-Conformer names**, not `q_proj`/`k_proj`. The
  encoder holds 91.8% of the parameters, so adapting the decoder alone does
  almost nothing. Run `train.cohere.preflight` before any paid GPU minute.
- **Confidence is ours.** NVIDIA's utility is broken on TDT and Cohere ships
  none, so both lanes use `contract/confidence.py`.
- **Manifest paths are relative** to `$VOICEBRIDGE_DATA` and resolved at load.
- **Baseten project names collide org-wide**, hence the `-t34` suffix.

## Conventions

- `contract/` is frozen. Changing it voids every score already collected.
- `app/` talks HTTP and WebSocket only. Promoting a model is `ASR_WS_URL`.
- No LLM on the demo path. The verifier is deterministic and offline.
