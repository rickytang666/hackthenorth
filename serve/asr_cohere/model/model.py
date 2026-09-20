"""Protocol 1, backed by Cohere Transcribe (optionally LoRA-adapted).

Only `Recognizer` differs from serve/_template. The message shapes, the
stable-prefix rule and the confidence source are identical to the Parakeet
lane by construction, which is what makes the promotion gate a URL change.

Three things here are load-bearing and were each verified the hard way in
train/cohere/preflight.py and decode_cohere.py:

  * the decoder prompt prefix forces English. Without it this model does no
    language identification and will transcribe dysarthric English into Arabic
    script at 0.9 confidence.
  * confidence comes from our own token log-probabilities via
    contract.confidence, because no usable one ships with either model.
  * the model needs an explicit decoder_input_ids; it does not shift labels.
"""

import base64
import json

import fastapi
import os
import sys
import time
from pathlib import Path

# contract/ and train/cohere/model.py are vendored into packages/ by
# sync_packages.sh so the Truss is self-contained. Locally they resolve from
# the repo root instead, which is why this is a prepend and not a replace.
_PACKAGES = Path(__file__).resolve().parents[1] / "packages"
if _PACKAGES.exists():
    sys.path.insert(0, str(_PACKAGES))

import numpy as np
import soundfile as sf
import torch

from contract.confidence import ACCEPT_THRESHOLD, from_logprobs
from train.cohere.model import MODEL_ID, load, prompt_ids

SAMPLE_RATE = 16000
# Re-run the encoder on a trailing window rather than claiming cache-aware
# streaming this architecture does not expose. Conservative and correct.
WINDOW_SECONDS = 12.0
PARTIAL_EVERY_MS = 400
MIN_AUDIO_SECONDS = 0.4


class Recognizer:
    def __init__(self, model, processor, prompt, device):
        self.model, self.processor, self.prompt, self.device = model, processor, prompt, device
        self.pcm = bytearray()
        self.last_partial = 0.0
        self.stable = ""

    def _decode(self, max_new_tokens: int):
        audio = np.frombuffer(bytes(self.pcm), dtype="<i2").astype(np.float32) / 32768.0
        audio = audio[-int(WINDOW_SECONDS * SAMPLE_RATE):]
        if audio.size < MIN_AUDIO_SECONDS * SAMPLE_RATE:
            return None
        inputs = self.processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                language="en", punctuation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            gen = self.model.generate(**inputs, decoder_input_ids=self.prompt,
                                      max_new_tokens=max_new_tokens,
                                      return_dict_in_generate=True, output_scores=True)
        sequences = gen.sequences
        prompt_len = sequences.shape[1] - len(gen.scores)
        text = self.processor.tokenizer.decode(sequences[0], skip_special_tokens=True).strip()
        logprobs = []
        for step, scores in enumerate(gen.scores):
            index = prompt_len + step
            if index >= sequences.shape[1]:
                break
            logprobs.append(
                torch.log_softmax(scores[0].float(), dim=-1)[sequences[0, index]].item()
            )
        return text, from_logprobs(logprobs)

    def push(self, pcm: bytes) -> dict | None:
        self.pcm.extend(pcm)
        now = time.perf_counter() * 1000
        if now - self.last_partial < PARTIAL_EVERY_MS:
            return None
        self.last_partial = now
        result = self._decode(max_new_tokens=48)
        if result is None:
            return None
        text, confidence = result
        # Only the shared prefix with the previous hypothesis is promised stable;
        # everything past it may still be revised by later audio.
        shared = 0
        for a, b in zip(self.stable, text):
            if a != b:
                break
            shared += 1
        self.stable = text
        return {"text": text, "stable_prefix_len": shared, "confidence": confidence.sequence}

    def _alternatives(self, best: str, n: int = 3) -> list[dict]:
        """Beam-search n-best, for when the greedy answer is not trustworthy.

        Run only below the accept threshold. The greedy pass still produces the
        text and the confidence, so the calibrated threshold stays valid and the
        common path pays nothing for this.
        """
        audio = np.frombuffer(bytes(self.pcm), dtype="<i2").astype(np.float32) / 32768.0
        audio = audio[-int(WINDOW_SECONDS * SAMPLE_RATE):]
        inputs = self.processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                language="en", punctuation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs, decoder_input_ids=self.prompt, max_new_tokens=128,
                num_beams=max(4, n + 1), num_return_sequences=n,
                return_dict_in_generate=True, output_scores=True,
            )
        scores = getattr(gen, "sequences_scores", None)
        out, seen = [], set()
        for i, seq in enumerate(gen.sequences):
            text = self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append({"text": text,
                        "score": float(scores[i]) if scores is not None else -float(i + 1)})
        # The greedy answer must be offered even if beam search ranked it out.
        if best and best.lower() not in seen:
            out.insert(0, {"text": best, "score": 0.0})
        return out[:n]

    def finish(self) -> dict:
        result = self._decode(max_new_tokens=128)
        if result is None:
            return {"text": "", "confidence": 0.0, "candidates": [{"text": "", "score": -99.0}]}
        text, confidence = result
        candidates = [{"text": text,
                       "score": float(np.log(max(confidence.sequence, 1e-6)))}]
        if confidence.sequence < ACCEPT_THRESHOLD:
            try:
                alts = self._alternatives(text)
                if len(alts) > 1:
                    candidates = alts
            except Exception as exc:
                # Alternatives are a nicety; never fail the transcript over them.
                print(f"n-best failed, falling back to single candidate: {exc}", flush=True)
        return {"text": text, "confidence": confidence.sequence, "candidates": candidates}


class Model:
    def __init__(self, **kwargs):
        # The base weights are a gated HF repo. Truss hands secrets in here, and
        # transformers only reads them from the environment, so bridge the two
        # before anything touches the hub.
        self._secrets = kwargs.get("secrets") or {}
        self._model = self._processor = self._prompt = None
        self.model_id = MODEL_ID
        self._ready = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Truss bundles packages/ at the container filesystem root (verified:
        # /packages/best), not next to model/, so the local and deployed paths
        # differ and both are searched. A missing adapter must be loud: silently
        # serving the base model would invalidate the sealed test's numbers.
        name = os.environ.get("COHERE_ADAPTER_PATH", "packages/best")
        here = Path(__file__).resolve()
        leaf = Path(name).name
        candidates = [Path(name)] if Path(name).is_absolute() else [
            here.parents[1] / name, here.parents[2] / name, Path.cwd() / name,
            here.parents[1] / leaf, Path("/packages") / leaf, Path("/app/packages") / leaf,
        ]
        self._adapter = next((str(c) for c in candidates
                              if (c / "adapter_config.json").exists()), None)
        self._adapter_searched = [str(c) for c in candidates]

    def load(self) -> None:
        token = self._secrets.get("hf_token")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        from transformers import AutoProcessor
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self._adapter is None:
            raise RuntimeError(
                "LoRA adapter not found; refusing to serve the untuned base model. "
                f"searched: {self._adapter_searched}"
            )
        # Printed because the container layout of bundled packages is not
        # documented and this is the only record of which path won.
        print(f"cohere adapter resolved to {self._adapter}", flush=True)
        self._model = load(dtype, self._adapter).eval().to(self._device)
        self.model_id = f"{MODEL_ID}+{os.path.basename(self._adapter)}"
        self._prompt = torch.tensor(
            [prompt_ids(self._processor.tokenizer, "en", punctuation=False)],
            dtype=torch.long, device=self._device,
        )
        if self._device != "cuda":
            # Refuse rather than silently serving a CPU model: it would pass a
            # smoke test and then miss every latency number on stage.
            raise RuntimeError(
                "CUDA unavailable. torch fell back to CPU, which usually means "
                "the torch build does not match the base image driver."
            )

        # Warm once so the first real request is not reported as steady state.
        warm = self._processor([np.zeros(SAMPLE_RATE, dtype=np.float32)],
                               sampling_rate=SAMPLE_RATE, return_tensors="pt",
                               language="en", punctuation=False)
        with torch.inference_mode():
            self._model.generate(**{k: v.to(self._device) for k, v in warm.items()},
                                 decoder_input_ids=self._prompt, max_new_tokens=4)
        self._ready = True

    def is_healthy(self) -> bool:
        """Declare readiness explicitly. Inferred readiness left the replica
        marked unhealthy after a 60 s load, and the router then refused every
        websocket handshake with a 500 before it reached this container."""
        return self._ready

    async def websocket(self, websocket: fastapi.WebSocket) -> None:
        recognizer = Recognizer(self._model, self._processor, self._prompt, self._device)
        began = time.perf_counter()
        try:
            # Documented Baseten pattern: receive_text() in a loop, not
            # iter_text(). Baseten accepts the connection itself, so never
            # call websocket.accept().
            while True:
                message = json.loads(await websocket.receive_text())
                if message.get("type") == "audio":
                    partial = recognizer.push(base64.b64decode(message["pcm16_b64"]))
                    if partial:
                        await websocket.send_text(json.dumps({
                            "type": "partial", **partial,
                            "t_ms": round(1000 * (time.perf_counter() - began), 1),
                        }))
                elif message.get("type") == "end":
                    await websocket.send_text(json.dumps({
                        "type": "final", **recognizer.finish(),
                        "latency_ms": round(1000 * (time.perf_counter() - began), 1),
                        "model_id": self.model_id,
                    }))
                    return
        except fastapi.WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
            raise
