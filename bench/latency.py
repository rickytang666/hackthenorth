"""Latency harness. Drives Protocol 1 and writes one scorecard row.

    python -m bench.latency --url ws://127.0.0.1:8765 --runs 5
"""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import websockets

FRAME_MS = 20
SAMPLE_RATE = 16000
BYTES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000) * 2


def _frames(audio_path: str | None, seconds: float):
    if audio_path:
        import soundfile as sf
        data, _ = sf.read(audio_path, dtype="int16")
        raw = data.tobytes()
    else:
        raw = b"\x00" * int(SAMPLE_RATE * seconds) * 2
    for offset in range(0, len(raw), BYTES_PER_FRAME):
        yield raw[offset : offset + BYTES_PER_FRAME]


async def one_run(url: str, audio_path: str | None, seconds: float) -> dict:
    began = time.perf_counter()
    first_partial = None
    final = None
    async with websockets.connect(url) as ws:
        async def send():
            for seq, frame in enumerate(_frames(audio_path, seconds)):
                await ws.send(json.dumps({
                    "type": "audio", "seq": seq,
                    "pcm16_b64": base64.b64encode(frame).decode(),
                }))
                await asyncio.sleep(FRAME_MS / 1000.0)
            await ws.send(json.dumps({"type": "end"}))

        sender = asyncio.create_task(send())
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "partial" and first_partial is None:
                    first_partial = 1000.0 * (time.perf_counter() - began)
                elif msg["type"] == "final":
                    final = msg
                    break
                elif msg["type"] == "error":
                    raise RuntimeError(msg.get("message", "server error"))
        finally:
            sender.cancel()
    total = 1000.0 * (time.perf_counter() - began)
    return {
        "first_partial_ms": first_partial,
        "final_ms": total,
        "reported_latency_ms": (final or {}).get("latency_ms"),
        "model_id": (final or {}).get("model_id"),
        "text": (final or {}).get("text"),
    }


def percentile(values, p):
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def main_async(args) -> None:
    runs = [await one_run(args.url, args.audio, args.seconds) for _ in range(args.runs)]
    row = {
        "url": args.url,
        "model_id": runs[0]["model_id"],
        "runs": args.runs,
        "first_partial_p50_ms": percentile([r["first_partial_ms"] for r in runs], 50),
        "first_partial_p95_ms": percentile([r["first_partial_ms"] for r in runs], 95),
        "final_p50_ms": percentile([r["final_ms"] for r in runs], 50),
        "final_p95_ms": percentile([r["final_ms"] for r in runs], 95),
    }
    print(json.dumps(row, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("a") as f:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--audio", default=None, help="16 kHz mono WAV, else silence")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="results/latency.jsonl")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
