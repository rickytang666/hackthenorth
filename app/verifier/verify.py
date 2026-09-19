"""Protocol 3: the constrained verifier.

When acoustic evidence is weak the system asks instead of inventing. Every rule
here is deterministic and offline: no model call, no network, no API key.

That is a deliberate narrowing. An earlier version asked an LLM to break ties in
the middle-confidence band. It was dropped because it only ever fired when
candidates differed harmlessly, where showing the person two buttons is an
equally good answer, and because a network call on the demo path is a liability
at a booth. Nothing the demo shows depended on it.

The hard guarantee, enforced in code: any text this returns is byte-identical to
something the recognizer actually proposed.
"""

from dataclasses import dataclass, field

from contract.confidence import ABSTAIN_THRESHOLD, ACCEPT_THRESHOLD
from contract.evaluate import load_critical_terms
from contract.normalize import tokens


@dataclass
class Verdict:
    action: str                       # accept | clarify | abstain
    text: str | None = None
    choices: list[str] = field(default_factory=list)
    reason: str = ""

    def to_protocol(self) -> dict:
        return {"action": self.action, "text": self.text, "choices": self.choices}


def differ_on_critical_terms(candidates: list[str]) -> bool:
    """True when candidates disagree about a word whose loss changes the meaning.

    "do not call the nurse" against "do call the nurse" is one word in six and
    the only error here that could hurt someone, so it must never be resolved
    by picking the higher score.
    """
    critical = load_critical_terms()
    seen = [{t for t in tokens(c) if t in critical} for c in candidates]
    return any(s != seen[0] for s in seen[1:])


def enforce(verdict: Verdict, candidates: list[str]) -> Verdict:
    """Reject anything the recognizer did not actually propose.

    Kept as a public function even with no model in the loop: it is the
    invariant the protocol promises, and it guards any future caller.
    """
    if verdict.action == "abstain":
        return Verdict("abstain", reason=verdict.reason or "no plausible candidate")
    if verdict.action == "accept" and verdict.text not in set(candidates):
        return Verdict("clarify", choices=candidates[:3],
                       reason="proposed text is outside the candidate set")
    if verdict.action == "clarify":
        choices = [c for c in verdict.choices if c in set(candidates)] or candidates[:3]
        return Verdict("clarify", choices=choices[:3], reason=verdict.reason)
    return verdict


def verify(candidates: list[str], confidence: float, topic: str = "") -> Verdict:
    """Decide accept, clarify, or abstain. Never returns text outside `candidates`."""
    if not candidates:
        return Verdict("abstain", reason="no candidates")
    if confidence < ABSTAIN_THRESHOLD:
        return Verdict("abstain", reason="confidence below the abstain floor")
    if differ_on_critical_terms(candidates):
        return Verdict("clarify", choices=candidates[:3],
                       reason="candidates differ on a critical term")
    if confidence >= ACCEPT_THRESHOLD:
        return Verdict("accept", text=candidates[0])
    return Verdict("clarify", choices=candidates[:3], reason="confidence below the accept threshold")
