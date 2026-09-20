"""Protocol 1 reference implementation with a stub recognizer.

Copy this directory to serve/asr_parakeet or serve/asr_cohere and replace
`Recognizer` only. Everything else, the message shapes, the stable-prefix rule,
and the confidence source, must stay identical across both lanes or the
promotion table is comparing transports instead of models.
"""

import base64
import json

import fastapi
import time

MODEL_ID = "stub-template"
SAMPLE_RATE = 16000


class Recognizer:
    """Replace this class in a copy. Keep the method signatures."""

    def __init__(self) -> None:
        self._pcm = bytearray()

    def push(self, pcm: bytes) -> dict | None:
        """Feed audio. Return a partial hypothesis, or None if nothing changed."""
        self._pcm.extend(pcm)
        seconds = len(self._pcm) / (SAMPLE_RATE * 2)
        if seconds < 0.5:
            return None
        words = ["stub"] * max(1, int(seconds))
        text = " ".join(words)
        return {
            "text": text,
            # The last word is still in flight and may be revised.
            "stable_prefix_len": len(" ".join(words[:-1])),
            "confidence": 0.5,
        }

    def finish(self) -> dict:
        """Return the final hypothesis, its confidence, and its candidate set."""
        seconds = len(self._pcm) / (SAMPLE_RATE * 2)
        text = " ".join(["stub"] * max(1, int(seconds)))
        return {
            "text": text,
            "confidence": 0.9,
            "candidates": [{"text": text, "score": -1.0}],
        }


class Model:
    def __init__(self, **kwargs) -> None:
        self._recognizer_factory = Recognizer

    def load(self) -> None:
        """Load weights here. Called once per worker, before any request."""
        return None

    async def websocket(self, websocket: fastapi.WebSocket) -> None:
        recognizer = self._recognizer_factory()
        began = time.perf_counter()
        try:
            # Documented Baseten pattern: receive_text() in a loop, not
            # iter_text(). Baseten accepts the connection itself, so never
            # call websocket.accept().
            while True:
                message = json.loads(await websocket.receive_text())
                kind = message.get("type")
                if kind == "audio":
                    partial = recognizer.push(base64.b64decode(message["pcm16_b64"]))
                    if partial:
                        await websocket.send_text(json.dumps({
                            "type": "partial",
                            **partial,
                            "t_ms": round(1000.0 * (time.perf_counter() - began), 1),
                        }))
                elif kind == "end":
                    final = recognizer.finish()
                    await websocket.send_text(json.dumps({
                        "type": "final",
                        **final,
                        "latency_ms": round(1000.0 * (time.perf_counter() - began), 1),
                        "model_id": MODEL_ID,
                    }))
                    return
        except fastapi.WebSocketDisconnect:
            return
        except Exception as exc:  # surfaced to the client, never swallowed
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
            raise
