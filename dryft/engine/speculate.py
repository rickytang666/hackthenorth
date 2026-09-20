"""Prompt-lookup drafting and exact greedy acceptance; no draft model or weights.

Draft width follows the verify step's measured cost. Decode is weight-
bound at batch one - the step reads the same ~8 GB of weights however
many query rows ride along - so on an H100 (2026-09-20,
experiments/fp8-draft-research.md) a width-8 verify costs 3.951 ms
against 3.909 ms at width 4 and 3.715 ms for plain decode: +1.1% to carry
twice the draft. Acceptance keeps a prefix, so expected accepted length
never decreases with width, and on quote-heavy text width 8 accepts 7.75
tokens a step and runs 1.7x the width-4 rate. Cost turns up past there
(+12% at 16), so seven drafts is the knee.

A draft that misses is not free, though: the step still costs its ~6%
and emits the single bonus token. The lookup indexes generated text as
well as the prompt, so on text that never repeats it keeps proposing and
keeps missing - measured as a 3% LOSS at batch one on the benchmark's
random-token prompts. So drafting is governed rather than unconditional:
misses back the drafter off exponentially and a hit restores it, which
bounds the wasted overhead near zero on unpredictable text while leaving
the repetitive case - where every proposal lands - running at full rate.
Verification stays exact either way, which the contract admits by
construction.
"""

DRAFT_TOKENS = 7
# A verify that emits this few tokens has bought only its own overhead.
WASTED_ACCEPT = 1
# Back-off bounds, in decode steps that skip drafting entirely.
MIN_COOLDOWN = 4
MAX_COOLDOWN = 64


class DraftGovernor:
    """Suppress drafting while it is missing; restore it the moment it lands.

    Prompt-lookup hits arrive in bursts - a quoted span, a repeated
    identifier - so a miss predicts misses and a hit predicts hits. The
    governor skips proposing for a cooldown that doubles with each wasted
    verify and collapses on the first useful one, so a long unpredictable
    stretch costs at most one wasted verify per cooldown (under 0.2% at
    the cap) while a copy region is picked back up within a few tokens.
    """

    def __init__(self, min_cooldown=MIN_COOLDOWN, max_cooldown=MAX_COOLDOWN):
        self.min_cooldown = min_cooldown
        self.max_cooldown = max_cooldown
        self.cooldown = 0
        self.penalty = min_cooldown

    def should_draft(self):
        """True when the next step should propose; counts down otherwise."""
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        return True

    def record(self, accepted):
        """Feed back how many tokens a verify step actually emitted."""
        if accepted > WASTED_ACCEPT:
            self.cooldown = 0
            self.penalty = self.min_cooldown
        else:
            self.cooldown = self.penalty
            self.penalty = min(self.penalty * 2, self.max_cooldown)


def draft_width(output_length):
    """Widest draft this generation can use, or 0 when it cannot speculate.

    A verify step emits at most draft + 1 tokens, and the engine keeps a
    plain decode in reserve for the tail, so a generation must have room
    for the draft plus that decode. Clamping here (rather than refusing
    to speculate, as a fixed width would for short outputs) keeps short
    generations speculating at whatever width they can afford.
    """
    return max(0, min(DRAFT_TOKENS, output_length - 2))


class PromptLookup:
    def __init__(self, prompt, draft_length=DRAFT_TOKENS, ngram=4):
        self.tokens = list(prompt)
        self.draft_length = draft_length
        self.ngram = ngram
        self.next_index = 0
        self.following = {}

    def append(self, token):
        self.tokens.append(token)

    def propose(self):
        stop = len(self.tokens) - self.ngram - self.draft_length + 1
        for index in range(self.next_index, max(self.next_index, stop)):
            key = tuple(self.tokens[index:index + self.ngram])
            self.following[key] = index + self.ngram
        self.next_index = max(self.next_index, stop)
        if len(self.tokens) < self.ngram:
            return None
        start = self.following.get(tuple(self.tokens[-self.ngram:]))
        if start is None:
            return None
        return self.tokens[start:start + self.draft_length]


class BackoffPromptLookup:
    """Prefer the longest history match, falling back from four tokens to two."""

    def __init__(self, prompt, draft_length=DRAFT_TOKENS):
        self.matchers = [PromptLookup(prompt, draft_length, ngram)
                         for ngram in (4, 3, 2)]

    def append(self, token):
        for matcher in self.matchers:
            matcher.append(token)

    def propose(self):
        for matcher in self.matchers:
            proposal = matcher.propose()
            if proposal is not None:
                return proposal
        return None


def verified_tokens(proposal, predictions):
    """Accept matching target argmaxes, then emit one target correction/bonus.

    predictions[i] is Qwen's greedy token after current + proposal[:i].
    Once one proposal is wrong, predictions after it are on an unaccepted
    prefix and must never be emitted.
    """
    if len(predictions) != len(proposal) + 1:
        raise ValueError("verification needs one prediction per draft token plus a bonus")
    accepted = 0
    for draft, target in zip(proposal, predictions):
        if draft != target:
            break
        accepted += 1
    return predictions[:accepted + 1]
