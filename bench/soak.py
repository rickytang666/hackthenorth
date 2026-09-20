"""Run repeated Protocol 1 round trips against the in-process mock ASR server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import websockets

from contract.mock_asr import MockAsr, load_fixture


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil((percent / 100) * len(ordered)) - 1)
    return round(ordered[index], 2)


def current_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    try:
        kilobytes = int(subprocess.check_output(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip())
        return kilobytes * 1024
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage if sys.platform == "darwin" else usage * 1024)


def validate_partial(message: dict[str, object]) -> None:
    text = message.get("text")
    stable = message.get("stable_prefix_len")
    confidence = message.get("confidence")
    if not isinstance(text, str) or not isinstance(stable, int):
        raise ValueError("partial is missing text or stable_prefix_len")
    if stable < 0 or stable > len(text):
        raise ValueError("partial stable_prefix_len is outside text")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("partial confidence is outside 0..1")


def validate_final(message: dict[str, object]) -> None:
    if not isinstance(message.get("text"), str) or not message["text"]:
        raise ValueError("final is missing text")
    if not isinstance(message.get("model_id"), str) or not message["model_id"]:
        raise ValueError("final is missing model_id")
    candidates = message.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("final is missing candidates")
    if any(not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str) for candidate in candidates):
        raise ValueError("final candidate is malformed")


async def round_trip(url: str, timeout_seconds: float) -> dict[str, float]:
    began = time.perf_counter()
    first_partial_ms = None
    final_ms = None
    async with websockets.connect(url, open_timeout=timeout_seconds) as websocket:
        silence = base64.b64encode(b"\x00" * 640).decode()
        await websocket.send(json.dumps({"type": "audio", "seq": 0, "pcm16_b64": silence}))
        await websocket.send(json.dumps({"type": "end"}))

        async with asyncio.timeout(timeout_seconds):
            async for raw in websocket:
                message = json.loads(raw)
                if message.get("type") == "partial":
                    validate_partial(message)
                    if first_partial_ms is None:
                        first_partial_ms = 1000 * (time.perf_counter() - began)
                elif message.get("type") == "final":
                    validate_final(message)
                    final_ms = 1000 * (time.perf_counter() - began)
                    break
                elif message.get("type") == "error":
                    raise RuntimeError(str(message.get("message", "server error")))
                else:
                    raise ValueError(f"unknown Protocol 1 message: {message.get('type')}")

    if first_partial_ms is None or final_ms is None:
        raise ValueError("round trip ended before partial and final messages")
    return {"first_partial_ms": first_partial_ms, "final_ms": final_ms}


async def run_soak(args: argparse.Namespace) -> dict[str, object]:
    fixture = load_fixture(args.fixture)
    mock = MockAsr(fixture, partial_ms=args.partial_ms, final_ms=args.final_ms, seed=args.seed)
    server = await websockets.serve(mock.handle, args.host, args.port)
    port = server.sockets[0].getsockname()[1]
    url = f"ws://{args.host}:{port}/v1/stream"
    started_at = datetime.now(UTC).isoformat()
    began = time.monotonic()
    deadline = began + args.duration_seconds
    first_partial_values: list[float] = []
    final_values: list[float] = []
    failures: Counter[str] = Counter()
    failure_examples: list[str] = []
    rss_start = current_rss_bytes()
    rss_peak = rss_start
    attempts = 0

    try:
        while time.monotonic() < deadline and (args.max_runs is None or attempts < args.max_runs):
            attempts += 1
            try:
                timing = await round_trip(url, args.timeout_seconds)
                first_partial_values.append(timing["first_partial_ms"])
                final_values.append(timing["final_ms"])
            except Exception as error:
                error_name = type(error).__name__
                failures[error_name] += 1
                if len(failure_examples) < 5:
                    failure_examples.append(f"{error_name}: {error}")
            rss_peak = max(rss_peak, current_rss_bytes())
            if args.interval_seconds:
                await asyncio.sleep(args.interval_seconds)
    finally:
        server.close()
        await server.wait_closed()

    elapsed = time.monotonic() - began
    rss_end = current_rss_bytes()
    rss_peak = max(rss_peak, rss_end)
    successful = len(final_values)
    return {
        "started_at": started_at,
        "target_duration_seconds": args.duration_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "url": url,
        "attempts": attempts,
        "successful_round_trips": successful,
        "failures": sum(failures.values()),
        "failure_rate": round(sum(failures.values()) / attempts, 6) if attempts else 0,
        "failure_types": dict(sorted(failures.items())),
        "failure_examples": failure_examples,
        "first_partial_ms": {
            "p50": percentile(first_partial_values, 50),
            "p95": percentile(first_partial_values, 95),
        },
        "final_ms": {
            "p50": percentile(final_values, 50),
            "p95": percentile(final_values, 95),
        },
        "rss_bytes": {
            "start": rss_start,
            "end": rss_end,
            "drift": rss_end - rss_start,
            "peak": rss_peak,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=30 * 60)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=8)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--partial-ms", type=int, default=220)
    parser.add_argument("--final-ms", type=int, default=400)
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/soak.json"))
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")

    report = asyncio.run(run_soak(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failure_rate"] > args.max_failure_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
