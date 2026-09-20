"""Gate 3: drive the whole demo path once per curated clip, for real.

    source env.sh && uv run python scripts/e2e_check.py

Does exactly what app/index.html does, in the same order and at the same pace:
stream 20 ms PCM frames to the live ASR over Protocol 1, take the final, enroll
the speaker with the renderer, synthesize the recovered text in their voice.
Nothing here is mocked, and it fails loudly rather than degrading, because the
point is to catch a broken link before a judge does.

BASETEN_API_KEY comes from the environment and is never printed.
"""

import argparse
import asyncio
import base64
import json
import os
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "app" / "clips"
FRAME = 320                      # samples, 20 ms at 16 kHz
DEFAULT_ASR = "wss://model-qk542geq.api.baseten.co/environments/production/websocket"
CLARIFY_BELOW = 0.879


def pcm_frames(path: Path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, f"{path} is not 16 kHz mono"
        raw = w.readframes(w.getnframes())
    for i in range(0, len(raw), FRAME * 2):
        yield raw[i:i + FRAME * 2]


async def transcribe(url: str, key: str, path: Path) -> dict:
    import websockets
    # Python can set a real header, so it does. Never put the key in a
    # subprotocol: it is not an RFC 6455 token, and the rejection message quotes
    # the value back, which is how a key ends up in a log.
    kwargs = {"additional_headers": {"Authorization": f"Api-Key {key}"}} if key else {}
    # 10 s is the library default and the replica scales to zero, so a cold
    # start loses the handshake before the container is even up.
    async with websockets.connect(url, max_size=None, open_timeout=180,
                                  ping_interval=None, **kwargs) as ws:
        began = time.perf_counter()
        first_partial, partials = None, 0

        async def pump():
            for frame in pcm_frames(path):
                await ws.send(json.dumps({"type": "audio",
                                          "pcm16_b64": base64.b64encode(frame).decode()}))
                await asyncio.sleep(0.02)      # real time, so the timings mean something
            await ws.send(json.dumps({"type": "end"}))

        sender = asyncio.create_task(pump())
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if msg["type"] == "partial":
                    partials += 1
                    if first_partial is None:
                        first_partial = (time.perf_counter() - began) * 1000
                elif msg["type"] == "final":
                    msg["first_partial_ms"] = first_partial
                    msg["final_ms"] = (time.perf_counter() - began) * 1000
                    msg["partials"] = partials
                    return msg
                elif msg["type"] == "error":
                    raise RuntimeError(f"ASR returned an error: {msg['message']}")
        finally:
            sender.cancel()


def post(base: str, path: str, payload: dict):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    began = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=180)
    return r, (time.perf_counter() - began) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default=os.environ.get("ASR_WS_URL", DEFAULT_ASR))
    ap.add_argument("--voice", default="http://127.0.0.1:8770")
    ap.add_argument("--only", type=int, help="run just this clip number, 1-based")
    args = ap.parse_args()

    key = os.environ.get("BASETEN_API_KEY", "")
    if not key and "baseten.co" in args.asr:
        raise SystemExit("BASETEN_API_KEY is not set; source env.sh or export it")
    try:
        health = json.load(urllib.request.urlopen(args.voice + "/health", timeout=5))
    except OSError as exc:
        raise SystemExit(f"voice renderer unreachable at {args.voice}: {exc}\n"
                         f"start it: cd serve/voice && ./.venv/bin/python server.py")
    print(f"voice renderer on {health['device']}")

    clips = json.loads((CLIPS / "manifest.json").read_text())
    if args.only:
        clips = [clips[args.only - 1]]
    failures, rows = [], []
    for n, clip in enumerate(clips, 1):
        wav = ROOT / "app" / clip["wav"]
        final = asyncio.run(transcribe(args.asr, key, wav))
        heard = final["text"].strip().lower()
        want = clip["tuned"]
        exact = heard == want

        _, enroll_ms = post(args.voice, "/v1/enroll",
                            {"wavs_b64": [base64.b64encode(wav.read_bytes()).decode()]})
        vid = None
        r, _ = post(args.voice, "/v1/enroll",
                    {"wavs_b64": [base64.b64encode(wav.read_bytes()).decode()]})
        vid = json.load(r)["voice_id"]
        r, synth_ms = post(args.voice, "/v1/synthesize", {"text": final["text"], "voice_id": vid})
        audio = r.read()

        clarified = final["confidence"] < CLARIFY_BELOW
        rows.append((n, clip["utterance_id"], exact, final["confidence"], clarified,
                     len(final.get("candidates", [])), final["first_partial_ms"],
                     final["final_ms"], synth_ms, len(audio)))
        if not exact:
            failures.append(f"clip {n} {clip['utterance_id']}: expected {want!r}, got {heard!r}")
        if clarified and len(final.get("candidates", [])) < 2:
            failures.append(f"clip {n} {clip['utterance_id']}: below threshold but only "
                            f"{len(final.get('candidates', []))} candidate(s); "
                            f"the clarification card would have nothing to offer")
        if len(audio) < 4096:
            failures.append(f"clip {n} {clip['utterance_id']}: synthesized only {len(audio)} bytes")

    print(f"\n| # | clip | exact | conf | clarifies | alts | 1st partial | final | synth |")
    print(f"|---|---|---|---:|---|---:|---:|---:|---:|")
    for n, uid, exact, conf, clar, alts, fp, fin, syn, _ in rows:
        print(f"| {n} | {uid} | {'yes' if exact else 'NO'} | {conf:.3f} | "
              f"{'yes' if clar else 'no'} | {alts} | "
              f"{fp:.0f} ms | {fin:.0f} ms | {syn:.0f} ms |")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print(f"\nall {len(rows)} clips passed end to end")


if __name__ == "__main__":
    main()
