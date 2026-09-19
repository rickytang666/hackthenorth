# VoiceBridge

Real-time speech recovery for dysarthric speech that speaks the result back in
the user's own voice. Plans live in [plans/](plans/): [DESIGN.md](plans/DESIGN.md)
is the technical agreement, [OWNERSHIP.md](plans/OWNERSHIP.md) says who builds
what and when.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
source env.sh                                    # sets VOICEBRIDGE_DATA, PYTHONPATH
.venv/bin/hf download abnerh/TORGO-database --repo-type dataset --local-dir ./data/torgo-dataset
.venv/bin/python -m data.prepare_torgo           # builds the five manifests
.venv/bin/python -m contract.manifest            # must match contract/MANIFEST_HASHES
```

Then, in two shells:

```bash
.venv/bin/python -m contract.mock_asr --port 8765
.venv/bin/python -m bench.latency --url ws://127.0.0.1:8765 --runs 5
```

## Map

| Path | What |
|---|---|
| `contract/` | The shared ruler. Both lanes import it, neither forks it |
| `data/prepare_torgo.py` | Builds the five manifests. Deterministic, run once |
| `serve/_template/` | Truss that already speaks Protocol 1. Copy, never edit |
| `train/`, `serve/`, `app/`, `bench/` | Lane work, one owner each |

## Conventions

- The contract is `contract/`. Changing it after Phase 0 is a team announcement,
  not a quiet commit, because both lanes' scores become incomparable.
- Manifest paths are relative to `$VOICEBRIDGE_DATA` and resolved at load time.
- `app/` talks HTTP and WebSocket only. Promoting a model is `ASR_WS_URL`.
- Confidence comes from `contract/confidence.py`. NeMo's own utility is broken on TDT.
