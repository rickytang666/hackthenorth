"""Prefill graph replays must consume fresh prompts and restore decode state."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from decode import DecodeState
    from prefill import prefill
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class PrefillTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval().cuda()

    def test_changed_prompts_restore_cache_and_decode_position(self):
        with torch.inference_mode():
            for batch, length, output in ((2, 7, 6), (1, 1, 1), (1, 12, 9), (16, 512, 1)):
                eager = DecodeState(self.model, batch, length, output)
                graphed = DecodeState(self.model, batch, length, output)
                for repeat in range(3):
                    ids = torch.randint(0, 97, (batch, length), device="cuda")
                    for state in (eager, graphed):
                        state.token.fill_(96)
                        state.position.fill_(length + output + 7)
                        for tensor in state.cache.keys + state.cache.values:
                            tensor.fill_(1000 if repeat % 2 else -1000)
                    expected = eager.prefill(self.model, ids)
                    actual = prefill(self.model, graphed, ids)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertEqual(graphed.position.item(), length)
                    self.assertFalse(graphed.cache.prefill)
                    self.assertTrue(hasattr(graphed, "prefill_graph"))
                    for left, right in zip(eager.cache.keys + eager.cache.values,
                                           graphed.cache.keys + graphed.cache.values):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                    for _ in range(output - 1):
                        eager.graph.replay()
                        graphed.graph.replay()
                        torch.testing.assert_close(graphed.token, eager.token, atol=0, rtol=0)

    def test_large_prefill_uses_eager_path(self):
        with torch.inference_mode():
            ids = torch.randint(0, 97, (32, 513), device="cuda")
            state = DecodeState(self.model, 32, 513, 1)
            expected = state.prefill(self.model, ids).clone()
            actual = prefill(self.model, state, ids)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertFalse(hasattr(state, "prefill_graph"))


if __name__ == "__main__":
    unittest.main()
