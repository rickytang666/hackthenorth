"""Protocol-correct fake ASR server.

This is the reason the product lane never waits on a training job: `app/` is
built and demoed against this from minute one, and promoting a real model is a
change to ASR_WS_URL.

    python -m contract.mock_asr --fixture results/fixture.jsonl --port 8765
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import websockets

DEFAULT_LINES = [
    ("i need my medication", ["i need my medication", "i need my medicine"], 0.91),
    ("do not call the nurse", ["do not call the nurse", "do call the nurse"], 0.44),
    ("yes please", ["yes please", "s please"], 0.83),
]


def load_fixture(path: str | None) -> list[tuple[str, list[str], float]]:
    if not path:
        return DEFAULT_LINES
    out = []
    for line in Path(path).open():
        row = json.loads(line)
        text = row.get("prediction") or row["text"]
        out.append((text, row.get("candidates", [text]), row.get("confidence", 0.9)))
    return out


class MockAsr:
    def __init__(self, fixture, partial_ms: int = 220, final_ms: int = 400, seed: int = 0):
        self.fixture = fixture
        self.partial_ms = partial_ms
        self.final_ms = final_ms
        self.rng = random.Random(seed)

    async def handle(self, websocket):
        text, candidates, confidence = self.rng.choice(self.fixture)
        words = text.split()
        elapsed = 0.0
        task = asyncio.create_task(self._drain(websocket))
        try:
            for i in range(1, len(words) + 1):
                await asyncio.sleep(self.partial_ms / 1000.0)
                elapsed += self.partial_ms
                shown = " ".join(words[:i])
                # The last word is still in flight, so it is not in the stable prefix.
                stable = len(" ".join(words[: max(0, i - 1)]))
                await websocket.send(json.dumps({
                    "type": "partial",
                    "text": shown,
                    "stable_prefix_len": stable,
                    "confidence": round(min(0.95, 0.4 + 0.1 * i), 3),
                    "t_ms": round(elapsed, 1),
                }))
            await asyncio.sleep(self.final_ms / 1000.0)
            elapsed += self.final_ms
            await websocket.send(json.dumps({
                "type": "final",
                "text": text,
                "confidence": confidence,
                "latency_ms": round(elapsed, 1),
                "model_id": "mock-asr",
                "candidates": [
                    {"text": c, "score": -1.2 - 1.5 * i} for i, c in enumerate(candidates)
                ],
            }))
        finally:
            task.cancel()

    async def _drain(self, websocket):
        """Consume client audio frames so back-pressure does not stall the send side."""
        try:
            async for _ in websocket:
                pass
        except websockets.ConnectionClosed:
            pass


async def serve(host: str, port: int, fixture) -> None:
    server = MockAsr(fixture)
    async with websockets.serve(server.handle, host, port):
        print(f"mock ASR listening on ws://{host}:{port}/v1/stream")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port, load_fixture(args.fixture)))


if __name__ == "__main__":
    main()
