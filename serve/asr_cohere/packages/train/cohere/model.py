"""Single place that loads Cohere Transcribe, with the revision workarounds.

Revision b1eacc26 needs two patches against transformers 5.x. Both are narrower
than pinning transformers down a major version, and both belong here rather than
duplicated in every caller.
"""

import torch
from transformers import GenerationMixin
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

_patched = None


def model_class():
    """The remote model class, with both revision defects repaired."""
    global _patched
    if _patched is not None:
        return _patched

    cls = get_class_from_dynamic_module(
        "modeling_cohere_asr.CohereAsrForConditionalGeneration", MODEL_ID
    )

    # 1. The class declares this as a list; transformers 5.x unions it with a
    #    set during load finalization and raises after the weights are loaded.
    ignore = getattr(cls, "_keys_to_ignore_on_load_unexpected", None)
    if isinstance(ignore, list):
        cls._keys_to_ignore_on_load_unexpected = set(ignore)

    # 2. Its generate() calls super().generate(), but PreTrainedModel stopped
    #    inheriting GenerationMixin in transformers 4.50. Mixing it in after the
    #    base class keeps the model's own generate() first in the MRO.
    if not issubclass(cls, GenerationMixin):
        cls = type(cls.__name__, (cls, GenerationMixin), {})

    _patched = cls
    return cls


def load(dtype=torch.float32, adapter: str | None = None):
    model = model_class().from_pretrained(MODEL_ID, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model


def build_prompt(language: str = "en", punctuation: bool = False) -> str:
    """Decoder prompt prefix that forces the output language.

    This model does no language identification: without this prefix it free-runs
    and, on dysarthric English, will happily emit Arabic script at 0.9
    confidence. The processor's `language=` kwarg does not do this; only the
    decoder prompt does. Training and inference must use the identical prefix.
    """
    pnc = "<|pnc|>" if punctuation else "<|nopnc|>"
    return (
        "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
        f"<|{language}|><|{language}|>{pnc}<|noitn|><|notimestamp|><|nodiarize|>"
    )


def prompt_ids(tokenizer, language: str = "en", punctuation: bool = False) -> list[int]:
    return tokenizer(build_prompt(language, punctuation), add_special_tokens=False)["input_ids"]
