"""Shared confidence scorer.

NVIDIA's packaged confidence utility raises IndexError on Parakeet-TDT (it works
on the RNNT and CTC variants) because TDT emits a token AND a duration and skips
blank frames, breaking the per-frame indexing that utility assumes. The bug has
been open and stale since 2024, so we compute our own score from raw token
log-probabilities, which are still available from both models.

Both lanes must use this, identically, or the clarification rates in the
promotion table are not comparable.
"""

import math
from dataclasses import dataclass

# Starting values. Calibrate on the dev speaker before trusting them.
ACCEPT_THRESHOLD = 0.70
ABSTAIN_THRESHOLD = 0.30


@dataclass(frozen=True)
class Confidence:
    sequence: float          # exp(mean token log-prob), 0 to 1
    weakest_token: float     # exp(min token log-prob), 0 to 1
    n_tokens: int

    def action(self, accept: float = ACCEPT_THRESHOLD, abstain: float = ABSTAIN_THRESHOLD) -> str:
        if self.sequence >= accept:
            return "accept"
        if self.sequence < abstain:
            return "abstain"
        return "clarify"


def from_logprobs(token_logprobs: list[float]) -> Confidence:
    """Sequence confidence from per-token log-probabilities (natural log)."""
    if not token_logprobs:
        return Confidence(0.0, 0.0, 0)
    mean = sum(token_logprobs) / len(token_logprobs)
    return Confidence(
        sequence=math.exp(mean),
        weakest_token=math.exp(min(token_logprobs)),
        n_tokens=len(token_logprobs),
    )


def calibrate(scored: list[tuple[Confidence, bool]], target_precision: float = 0.95) -> float:
    """Lowest accept threshold whose accepted set is at least target_precision correct.

    `scored` is (confidence, was_the_transcript_correct) over the dev speaker.
    Raw probabilities run overconfident, so 0.6 does not mean 60% until this runs.
    """
    if not scored:
        return ACCEPT_THRESHOLD
    candidates = sorted({round(c.sequence, 3) for c, _ in scored})
    for threshold in candidates:
        accepted = [ok for c, ok in scored if c.sequence >= threshold]
        if accepted and (sum(accepted) / len(accepted)) >= target_precision:
            return threshold
    return 1.0
