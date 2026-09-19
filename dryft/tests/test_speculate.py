import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from speculate import PromptLookup, verified_tokens


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


if __name__ == "__main__":
    unittest.main()
