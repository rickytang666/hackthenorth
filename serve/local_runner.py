"""Serve any Truss ASR model class over Protocol 1, locally.

Lets bench/latency.py and app/ drive a real recognizer before anything is
deployed, so a Protocol 1 mistake surfaces here rather than at the booth.

    uv run python -m serve.local_runner --module serve.asr_cohere.model.model --port 8766
"""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

import websockets


class TextAdapter:
    """Presents a `websockets` connection with the Truss websocket interface."""

    def __init__(self, connection):
        self._connection = connection

    def iter_text(self):
        return self._connection.__aiter__()

    async def send_text(self, text: str) -> None:
        await self._connection.send(text)


async def main_async(args) -> None:
    sys.path.insert(0, str(Path(args.module.split(".")[0]).resolve().parent))
    module = importlib.import_module(args.module)
    model = module.Model()
    print(f"loading {args.module} ...", flush=True)
    model.load()
    print(f"ready. Protocol 1 on ws://{args.host}:{args.port}", flush=True)

    async def handler(connection):
        try:
            await model.websocket(TextAdapter(connection))
        except websockets.ConnectionClosed:
            pass
        except Exception as exc:  # surfaced, never swallowed
            print(f"handler error: {exc}", flush=True)

    async with websockets.serve(handler, args.host, args.port, max_size=8 * 1024 * 1024):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="dotted path to a module exposing Model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
