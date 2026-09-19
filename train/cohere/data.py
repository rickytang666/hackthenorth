"""Dataset and collator for Cohere Transcribe supervised fine-tuning.

Audio-to-transcript cross-entropy. Not DPO, not an LLM text objective.
Reads the frozen manifests through contract.manifest so paths resolve per machine.
"""

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from contract import manifest

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0  # cap so a rare long clip cannot spike memory


class TorgoDataset(Dataset):
    def __init__(self, manifest_name: str, max_seconds: float = MAX_SECONDS):
        rows = manifest.load(manifest_name, resolve=True)
        self.rows = [r for r in rows if r["duration"] <= max_seconds]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio, sr = sf.read(row["audio_filepath"], dtype="float32")
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"expected {SAMPLE_RATE} Hz, got {sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return {"audio": audio, "text": row["text"], "utterance_id": row["utterance_id"]}


class SpeechSeq2SeqCollator:
    """Feature-extract a batch and build teacher-forcing targets.

    This model does NOT shift labels internally: its logits align one-to-one
    with `decoder_input_ids`, so the shift happens here. Passing `labels`
    unshifted trains the model to predict the token it was just given, which
    produces a plausible falling loss and a useless checkpoint.

    Verified against revision b1eacc26:
      processor -> input_features (B, 128, T), length (B,)
      tokenizer -> input_ids already wrapped in BOS 4 ... EOS 3, padded with 2

    Unit-test one batch and one backward pass before launching the paid job.
    """

    def __init__(self, processor, language: str = "en", punctuation: bool = False):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.language = language
        self.punctuation = punctuation

    def __call__(self, features: list[dict]) -> dict:
        inputs = self.processor(
            [f["audio"] for f in features],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language=self.language,
            punctuation=self.punctuation,
        )
        ids = self.tokenizer(
            [f["text"] for f in features], return_tensors="pt", padding=True
        )["input_ids"]
        if ids.shape[1] < 2:
            raise RuntimeError("tokenized targets are too short to shift")

        decoder_input_ids = ids[:, :-1].contiguous()
        labels = ids[:, 1:].clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch = dict(inputs)
        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels
        if (labels != -100).sum() == 0:
            raise RuntimeError("every label is masked; the collator is wrong")
        return batch
