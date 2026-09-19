"""Static server for app/. No build step, no framework, no bundler.

    uv run python -m app.serve --port 5173
"""

import argparse
import http.server
import functools
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(Path(__file__).parent))
    print(f"app on http://127.0.0.1:{args.port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
