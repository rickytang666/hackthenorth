import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from speculate import BackoffPromptLookup, PromptLookup, verified_tokens


class SpeculationTests(unittest.TestCase):
    def test_lookup_only_proposes_previously_seen_continuations(self):
        lookup = PromptLookup([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4])
        self.assertEqual(lookup.propose(), [5, 6, 7])
        lookup.append(99)
        self.assertIsNone(lookup.propose())

    def test_lookup_learns_verified_output_without_self_matching(self):
        lookup = PromptLookup([1, 2])
        self.assertIsNone(lookup.propose())
        for token in [3, 4, 5, 6, 7, 1, 2, 3, 4]:
            lookup.append(token)
        self.assertEqual(lookup.propose(), [5, 6, 7])

    def test_acceptance_emits_only_correct_prefix_and_one_target_token(self):
        proposal = [10, 11, 12]
        for accepted in range(4):
            predictions = proposal[:accepted] + [99] + [88] * (3 - accepted)
            self.assertEqual(verified_tokens(proposal, predictions), proposal[:accepted] + [99])

    def test_verification_requires_bonus_prediction(self):
        with self.assertRaises(ValueError):
            verified_tokens([1, 2, 3], [1, 2, 3])

    def test_backoff_prefers_longer_match_over_more_recent_short_match(self):
        history = [1, 2, 3, 4, 10, 11, 12, 9, 3, 4, 20, 21, 22, 1, 2, 3, 4]
        self.assertEqual(BackoffPromptLookup(history).propose(), [10, 11, 12])

    def test_backoff_uses_short_match_and_resets_for_new_prompt(self):
        first = BackoffPromptLookup([8, 9, 10, 11, 12, 1, 8, 9])
        self.assertEqual(first.propose(), [10, 11, 12])
        self.assertIsNone(BackoffPromptLookup([1, 8, 9]).propose())

    def test_backoff_matches_naive_history_search_after_every_append(self):
        import random

        rng = random.Random(1701)
        history = [rng.randrange(5) for _ in range(24)]
        lookup = BackoffPromptLookup(history)
        for _ in range(200):
            expected = None
            for ngram in (4, 3, 2):
                if len(history) < ngram:
                    continue
                matches = [i for i in range(len(history) - ngram - 3 + 1)
                           if history[i:i + ngram] == history[-ngram:]]
                if matches:
                    start = matches[-1] + ngram
                    expected = history[start:start + 3]
                    break
            self.assertEqual(lookup.propose(), expected)
            token = rng.randrange(5)
            history.append(token)
            lookup.append(token)


if __name__ == "__main__":
    unittest.main()
