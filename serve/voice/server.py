"""Protocol 2: the voice renderer.

    uv run --no-sync python server.py --reference <wav> [<wav> ...] --port 8770

Endpoints:
    POST /v1/enroll      {"wavs_b64": [...]}  -> {"voice_id": "..."}
    POST /v1/synthesize  {"text", "voice_id"} -> audio/wav bytes
    GET  /health

Enrollment embeddings are cached per voice_id. Generated speech is never
cached: it is health-related content the person just said.
"""

import argparse
import base64
import hashlib
import io
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor" / "OpenVoice"))

import torch  # noqa: E402
from melo.api import TTS  # noqa: E402
from openvoice.api import ToneColorConverter  # noqa: E402

CKPT = HERE / "checkpoints" / "openvoice_v2"
_lock = threading.Lock()
_voices: dict[str, torch.Tensor] = {}
_engine = None


class Engine:
    def __init__(self, device: str):
        self.device = device
        self.converter = ToneColorConverter(str(CKPT / "converter" / "config.json"), device=device)
        self.converter.load_ckpt(str(CKPT / "converter" / "checkpoint.pth"))
        self.tts = TTS(language="EN", device=device)
        self.speaker_id = self.tts.hps.data.spk2id["EN-US"]
        self.source_se = torch.load(CKPT / "base_speakers" / "ses" / "en-us.pth",
                                    map_location=device)

    def enroll(self, wav_paths: list[str]) -> str:
        embedding = self.converter.extract_se(wav_paths)
        voice_id = hashlib.sha256(
            b"".join(Path(p).read_bytes() for p in wav_paths)
        ).hexdigest()[:16]
        with _lock:
            _voices[voice_id] = embedding
        return voice_id

    def synthesize(self, text: str, voice_id: str, speed: float = 1.0) -> bytes:
        target = _voices.get(voice_id)
        if target is None:
            raise KeyError(f"unknown voice_id {voice_id}; enroll first")
        with tempfile.TemporaryDirectory() as tmp:
            base, out = f"{tmp}/base.wav", f"{tmp}/out.wav"
            # One call per confirmed clause. Whole-clause, not chunked: see
            # time_synthesis.py for the measurement that decides whether
            # streaming is worth its fragility.
            self.tts.tts_to_file(text, self.speaker_id, base, speed=speed)
            self.converter.convert(audio_src_path=base, src_se=self.source_se,
                                   tgt_se=target, output_path=out)
            return Path(out).read_bytes()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_OPTIONS(self) -> None:
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "voices": len(_voices), "device": _engine.device})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "invalid JSON"})

        if self.path == "/v1/enroll":
            paths = []
            with tempfile.TemporaryDirectory() as tmp:
                for i, b64 in enumerate(body.get("wavs_b64", [])):
                    path = f"{tmp}/ref{i}.wav"
                    Path(path).write_bytes(base64.b64decode(b64))
                    paths.append(path)
                if not paths:
                    return self._json(400, {"error": "wavs_b64 is required"})
                return self._json(200, {"voice_id": _engine.enroll(paths)})

        if self.path == "/v1/synthesize":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json(400, {"error": "text is required"})
            voice_id = body.get("voice_id") or next(iter(_voices), None)
            if voice_id is None:
                return self._json(409, {"error": "no voice enrolled"})
            began = time.perf_counter()
            try:
                audio = _engine.synthesize(text, voice_id, float(body.get("pace", 1.0)))
            except KeyError as exc:
                return self._json(404, {"error": str(exc)})
            ms = 1000 * (time.perf_counter() - began)
            print(f"synthesize {len(text.split())} words in {ms:.0f} ms", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("X-Synthesis-Ms", f"{ms:.0f}")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-Synthesis-Ms")
            self.end_headers()
            return self.wfile.write(audio)

        self._json(404, {"error": "not found"})

    def log_message(self, *args) -> None:
        pass


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # MPS roughly halves synthesis on Apple Silicon, which is the difference
    # between a demo gap you notice and one you do not. Overridable because the
    # converter has hit MPS kernel gaps on older torch builds.
    import os
    if os.environ.get("VOICE_DEVICE"):
        return os.environ["VOICE_DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    global _engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--reference", nargs="*", default=[],
                        help="enroll these wavs at startup")
    args = parser.parse_args()

    device = pick_device()
    began = time.perf_counter()
    _engine = Engine(device)
    print(f"engine ready on {device} in {time.perf_counter()-began:.1f}s")
    if args.reference:
        print(f"enrolled voice_id {_engine.enroll(list(args.reference))} "
              f"from {len(args.reference)} clip(s)")
    print(f"voice service on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
