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
import io
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from contract.confidence import from_logprobs
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

    def finish(self) -> dict:
        result = self._decode(max_new_tokens=128)
        if result is None:
            return {"text": "", "confidence": 0.0, "candidates": [{"text": "", "score": -99.0}]}
        text, confidence = result
        return {
            "text": text,
            "confidence": confidence.sequence,
            "candidates": [{"text": text, "score": float(np.log(max(confidence.sequence, 1e-6)))}],
        }


class Model:
    def __init__(self, **kwargs):
        self._model = self._processor = self._prompt = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._adapter = os.environ.get("COHERE_ADAPTER_PATH") or None

    def load(self) -> None:
        from transformers import AutoProcessor
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = load(dtype, self._adapter).eval().to(self._device)
        self._prompt = torch.tensor(
            [prompt_ids(self._processor.tokenizer, "en", punctuation=False)],
            dtype=torch.long, device=self._device,
        )
        # Warm once so the first real request is not reported as steady state.
        warm = self._processor([np.zeros(SAMPLE_RATE, dtype=np.float32)],
                               sampling_rate=SAMPLE_RATE, return_tensors="pt",
                               language="en", punctuation=False)
        with torch.inference_mode():
            self._model.generate(**{k: v.to(self._device) for k, v in warm.items()},
                                 decoder_input_ids=self._prompt, max_new_tokens=4)

    @property
    def model_id(self) -> str:
        return MODEL_ID + (f"+{os.path.basename(self._adapter)}" if self._adapter else "")

    async def websocket(self, websocket) -> None:
        recognizer = Recognizer(self._model, self._processor, self._prompt, self._device)
        began = time.perf_counter()
        try:
            async for raw in websocket.iter_text():
                message = json.loads(raw)
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
        except Exception as exc:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
            raise
