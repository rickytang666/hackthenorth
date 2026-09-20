"""Static server for app/, plus a WebSocket proxy to the ASR.

    source env.sh && uv run python -m app.serve

Serves the page on 5173 and proxies ws://127.0.0.1:8765 to the Baseten
endpoint. No build step, no framework, no bundler.

The proxy exists because Baseten authenticates WebSockets with an Authorization
header and a browser cannot set one. Routing through here also keeps
BASETEN_API_KEY on this machine: it is read from the environment, never sent to
the page, and never typed into a field.
"""

import argparse
import asyncio
import functools
import http.server
import os
import threading
from pathlib import Path

DEFAULT_UPSTREAM = "wss://model-qk542geq.api.baseten.co/environments/production/websocket"


def serve_static(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(Path(__file__).parent))
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


async def run_proxy(port: int, upstream: str, key: str) -> None:
    import websockets

    async def handle(browser):
        headers = {"Authorization": f"Api-Key {key}"} if key else {}
        # The replica scales to zero, so a cold start takes far longer than the
        # library's 10 s handshake default.
        async with websockets.connect(upstream, additional_headers=headers,
                                      open_timeout=180, ping_interval=None,
                                      max_size=None) as model:
            async def pipe(src, dst):
                async for message in src:
                    await dst.send(message)
            done, pending = await asyncio.wait(
                [asyncio.create_task(pipe(browser, model)),
                 asyncio.create_task(pipe(model, browser))],
                return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()

    async with websockets.serve(handle, "127.0.0.1", port, max_size=None):
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5173)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--upstream", default=os.environ.get("ASR_WS_URL", DEFAULT_UPSTREAM))
    args = ap.parse_args()

    key = os.environ.get("BASETEN_API_KEY", "")
    if not key and "baseten.co" in args.upstream:
        raise SystemExit("BASETEN_API_KEY is not set; source env.sh or export it")
    threading.Thread(target=serve_static, args=(args.port,), daemon=True).start()
    print(f"app on http://127.0.0.1:{args.port}")
    print(f"ASR proxy on ws://127.0.0.1:{args.ws_port} -> {args.upstream}")
    asyncio.run(run_proxy(args.ws_port, args.upstream, key))


if __name__ == "__main__":
    main()
