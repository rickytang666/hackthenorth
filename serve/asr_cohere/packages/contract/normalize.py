"""The one text normalizer. Both lanes import this and neither may fork it.

Word error rate is meaningless across lanes unless references and hypotheses are
cleaned identically, so this function is frozen at the end of Phase 0.

Worked examples:
    "Don't give her the insulin."  -> "dont give her the insulin"
    "  THE  QUICK   brown fox  "   -> "the quick brown fox"
    "It's twenty-five mg, okay?"   -> "its twenty five mg okay"
"""

import re
import unicodedata

_APOSTROPHES = dict.fromkeys(map(ord, "’ʼ´`"), "'")
_KEEP = re.compile(r"[^a-z0-9' ]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Deterministic."""
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES)
    text = text.lower()
    # Hyphens and slashes join words, so they become spaces rather than vanishing.
    text = re.sub(r"[-/_]+", " ", text)
    text = _KEEP.sub(" ", text)
    text = text.replace("'", "")
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []
