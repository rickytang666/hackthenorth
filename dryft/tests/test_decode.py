"""Exercise cache visibility and repeated prompts against native Qwen3."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from decode import DecodeState, KVCache, forward
    from speculate import verified_tokens
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class DecodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=97,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()

    def test_fixed_cache_matches_full_prefix_and_excludes_stale_slots(self):
        with torch.inference_mode():
            for batch, prompt_length, steps in ((1, 1, 5), (2, 7, 6), (3, 4, 1)):
                capacity = prompt_length + steps - 1
                cache = KVCache(self.model, batch, capacity)
                for repeat in range(2):
                    # Simulate unrelated previous samples in unused storage.
                    for tensor in cache.keys + cache.values:
                        tensor.fill_(1000 if repeat else -1000)
                    prefix = torch.randint(0, 97, (batch, prompt_length))
                    cache.prefill = True
                    current = forward(self.model, prefix, cache, torch.arange(prompt_length))
                    cache.prefill = False
                    for step in range(steps):
                        expected = self.model(prefix, use_cache=False).logits[:, -1].argmax(-1)
                        self.assertEqual(current[:, 0].tolist(), expected.tolist())
                        prefix = torch.cat((prefix, current), dim=1)
                        if step + 1 < steps:
                            position = torch.tensor([prompt_length + step])
                            mask = torch.zeros(capacity)
                            mask.masked_fill_(torch.arange(capacity) > position, float("-inf"))
                            current = forward(
                                self.model, current, cache, position, mask.view(1, 1, 1, -1)
                            )

    def test_multi_token_verification_and_rollback_match_native(self):
        with torch.inference_mode():
            capacity = 20
            cache = KVCache(self.model, 1, capacity)
            for accept_count in range(4):
                prompt = torch.randint(0, 97, (1, 7))
                cache.prefill = True
                current = forward(self.model, prompt, cache, torch.arange(7))
                cache.prefill = False
                prefix = torch.cat((prompt, current), dim=1)
                draft_prefix = prefix.clone()
                proposal = []
                for _ in range(3):
                    draft = self.model(draft_prefix, use_cache=False).logits[:, -1].argmax(-1)
                    proposal.append(draft.item())
                    draft_prefix = torch.cat((draft_prefix, draft[:, None]), dim=1)
                if accept_count < 3:
                    proposal[accept_count] = (proposal[accept_count] + 1) % 97
                inputs = torch.tensor([[current.item(), *proposal]])
                positions = torch.arange(7, 11)
                mask = torch.zeros(4, capacity)
                mask.masked_fill_(torch.arange(capacity)[None, :] > positions[:, None], float("-inf"))
                predictions = forward(
                    self.model, inputs, cache, positions, mask[None, None], last_only=False,
                )[0].tolist()
                emitted = verified_tokens(proposal, predictions)
                self.assertEqual(len(emitted), accept_count + 1)
                for token in emitted:
                    expected = self.model(prefix, use_cache=False).logits[:, -1].argmax(-1).item()
                    self.assertEqual(token, expected)
                    prefix = torch.cat((prefix, torch.tensor([[token]])), dim=1)
                position = torch.tensor([7 + len(emitted)])
                mask = torch.zeros(capacity)
                mask.masked_fill_(torch.arange(capacity) > position, float("-inf"))
                next_token = forward(
                    self.model, torch.tensor([[emitted[-1]]]), cache, position, mask[None, None, None],
                )
                expected = self.model(prefix, use_cache=False).logits[:, -1].argmax(-1).item()
                self.assertEqual(next_token.item(), expected)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_graph_replay_resets_positions_and_prompt_state(self):
        self.model.cuda()
        with torch.inference_mode():
            state = DecodeState(self.model, 2, 7, 6)
            for _ in range(2):
                prefix = torch.randint(0, 97, (2, 7), device="cuda")
                current = state.prefill(self.model, prefix)
                for step in range(6):
                    expected = self.model(prefix, use_cache=False).logits[:, -1].argmax(-1)
                    self.assertEqual(current[:, 0].tolist(), expected.tolist())
                    prefix = torch.cat((prefix, current), dim=1)
                    if step < 5:
                        state.graph.replay()
                        current = state.token

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA and Triton")
    def test_speculative_generator_stream_and_capacity_boundaries(self):
        from engine import Engine

        engine = Engine.__new__(Engine)
        engine.model = self.model.cuda()
        engine.state = None
        with torch.inference_mode():
            for length in (1, 4, 5, 9):
                for token in (7, 11):
                    prompt = [[token] * 12]
                    outputs = list(engine.generate(prompt, length))
                    self.assertEqual(len(outputs), length)
                    prefix = torch.tensor(prompt, device="cuda")
                    for step in outputs:
                        self.assertEqual(len(step), 1)
                        expected = self.model(prefix, use_cache=False).logits[:, -1].argmax(-1).item()
                        self.assertEqual(step[0], expected)
                        prefix = torch.cat((prefix, torch.tensor([step], device="cuda")), dim=1)


if __name__ == "__main__":
    unittest.main()
