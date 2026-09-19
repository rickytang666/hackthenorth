"""Prompt-lookup drafting and exact greedy acceptance; no draft model or weights."""

DRAFT_TOKENS = 3


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
