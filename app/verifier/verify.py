"""Protocol 3: the constrained verifier.

When acoustic evidence is weak the system must ask, not invent. This module is
what makes "asks instead of guessing" a property of the code rather than a
property of the prompt.

The hard guarantee, enforced here and not in the prompt: any text this returns
is byte-identical to one of the candidates the recognizer produced. An LLM that
hallucinates a fluent sentence, or quietly drops a negation, cannot get past
`_enforce`. The model chooses; it never writes.
"""

import json
import os
from dataclasses import dataclass, field

import urllib.error
import urllib.request

from contract.confidence import ABSTAIN_THRESHOLD, ACCEPT_THRESHOLD
from contract.evaluate import load_critical_terms
from contract.normalize import tokens

API_URL = os.environ.get("VERIFIER_API_URL", "https://api.anthropic.com/v1/messages")
MODEL = os.environ.get("VERIFIER_MODEL", "claude-haiku-4-5-20251001")

SYSTEM = """You choose between speech-recognition candidates for a person with \
dysarthria. You never write new text.

Reply with JSON only: {"action": "accept"|"clarify"|"abstain", "index": <int>, \
"choices": [<int>, ...]}

- "accept" with one index when a candidate is clearly what was said.
- "clarify" with two or three indices when several are plausible. Order them \
most likely first.
- "abstain" when none is plausible.

Indices refer to the candidate list. Never invent wording. Negation, names, \
numbers and medication terms change meaning, so if candidates differ on any of \
those, prefer "clarify" over "accept"."""


@dataclass
class Verdict:
    action: str                       # accept | clarify | abstain
    text: str | None = None
    choices: list[str] = field(default_factory=list)
    reason: str = ""

    def to_protocol(self) -> dict:
        return {"action": self.action, "text": self.text, "choices": self.choices}


def _differ_on_critical_terms(candidates: list[str]) -> bool:
    """True when candidates disagree about a word whose loss changes the meaning."""
    critical = load_critical_terms()
    seen = [{t for t in tokens(c) if t in critical} for c in candidates]
    return any(s != seen[0] for s in seen[1:])


def _enforce(verdict: Verdict, candidates: list[str]) -> Verdict:
    """Reject anything the recognizer did not actually propose."""
    if verdict.action == "abstain":
        return Verdict("abstain", reason=verdict.reason or "no plausible candidate")

    allowed = set(candidates)
    if verdict.action == "accept":
        if verdict.text not in allowed:
            return Verdict("clarify", choices=candidates[:3],
                           reason="verifier returned text outside the candidate set")
        return verdict

    choices = [c for c in verdict.choices if c in allowed]
    if len(choices) < 2:
        choices = candidates[:3]
    return Verdict("clarify", choices=choices[:3], reason=verdict.reason)


def _ask_model(candidates: list[str], confidence: float, topic: str) -> Verdict:
    payload = {
        "model": MODEL,
        "max_tokens": 200,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": json.dumps({
            "candidates": candidates, "confidence": confidence, "topic": topic,
        })}],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        body = json.loads(response.read())
    text = "".join(part.get("text", "") for part in body.get("content", []))
    parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])

    action = parsed.get("action")
    if action == "accept":
        return Verdict("accept", text=candidates[int(parsed["index"])])
    if action == "clarify":
        picked = [candidates[int(i)] for i in parsed.get("choices", []) if 0 <= int(i) < len(candidates)]
        return Verdict("clarify", choices=picked)
    return Verdict("abstain")


def verify(candidates: list[str], confidence: float, topic: str = "") -> Verdict:
    """Decide accept, clarify, or abstain. Never returns text outside `candidates`."""
    if not candidates:
        return Verdict("abstain", reason="no candidates")
    if len(candidates) == 1 and confidence >= ACCEPT_THRESHOLD:
        return Verdict("accept", text=candidates[0])
    if confidence < ABSTAIN_THRESHOLD:
        return Verdict("abstain", reason="confidence below the abstain floor")

    # Cheap, deterministic, and offline: if the candidates disagree about a
    # negation or a number, no model call can make guessing safe.
    if _differ_on_critical_terms(candidates):
        return Verdict("clarify", choices=candidates[:3],
                       reason="candidates differ on a critical term")

    if confidence >= ACCEPT_THRESHOLD:
        return Verdict("accept", text=candidates[0])

    try:
        return _enforce(_ask_model(candidates, confidence, topic), candidates)
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
        # A dead verifier must degrade to asking, never to guessing.
        return Verdict("clarify", choices=candidates[:3], reason=f"verifier unavailable: {exc}")
