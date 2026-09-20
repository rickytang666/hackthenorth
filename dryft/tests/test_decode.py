"""Exercise cache visibility and repeated prompts against native Qwen3."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from decode import DecodeState, KVCache, RotaryTable, forward, uses_position_causality, stream_decode
    from speculate import verified_tokens
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class DecodeTests(unittest.TestCase):
    def test_stream_decode_launches_before_yield_without_extra_steps(self):
        from types import SimpleNamespace

        for batch in (1, 2, 16):
            for length in (1, 2, 5):
                token = torch.arange(batch).reshape(batch, 1)
                launches = []
                def replay():
                    token.add_(1)
                    launches.append(1)
                state = SimpleNamespace(token=token, graph=SimpleNamespace(replay=replay))
                first = token[:, 0].tolist()
                saved = []
                for step, output in enumerate(stream_decode(state, first, length)):
                    self.assertEqual(output, [row + step for row in range(batch)])
                    self.assertEqual(len(launches), min(step + 1, length - 1))
                    saved.append(output)
                self.assertEqual(saved[0], list(range(batch)))
                self.assertEqual(len(launches), length - 1)

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

    def test_rotary_table_matches_native_and_reuses_changed_positions(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = self.model.to(dtype)
            table = RotaryTable(model, 32)
            for positions in (torch.arange(7), torch.tensor([7]),
                              torch.tensor([8, 9, 10, 11]), torch.tensor([31]),
                              torch.tensor([2, 3])):
                expected = model.model.rotary_emb(
                    model.model.embed_tokens.weight[:1].unsqueeze(0), positions[None])
                for actual, reference in zip(table.select(positions), expected):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        # Unknown or position-dependent rotary schemes must not be cached.
        self.model.model.rotary_emb.rope_type = "dynamic"
        self.assertIsNone(RotaryTable(self.model, 32).select(torch.tensor([2])))

    def test_cached_rotary_preserves_logits_and_native_mask_fallback(self):
        self.assertFalse(uses_position_causality(self.model))
        with torch.inference_mode():
            table = RotaryTable(self.model, 12)
            prompt = torch.randint(0, 97, (2, 7))
            positions = torch.arange(7)
            captured = []
            handle = self.model.lm_head.register_forward_hook(
                lambda module, inputs, output: captured.append(output.clone()))
            self.addCleanup(handle.remove)
            for cached in (False, True):
                cache = KVCache(self.model, 2, 12)
                current = forward(self.model, prompt, cache, positions,
                                  position_embeddings=table.select(positions) if cached else None)
                cache.prefill = False
                position = torch.tensor([7])
                mask = torch.arange(12)[None, None, None, :] <= position
                forward(self.model, current, cache, position, mask,
                        position_embeddings=table.select(position) if cached else None)
            torch.testing.assert_close(captured[0], captured[2], atol=0, rtol=0)
            torch.testing.assert_close(captured[1], captured[3], atol=0, rtol=0)

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
            logits = []
            handle = self.model.lm_head.register_forward_hook(
                lambda module, inputs, output: logits.append(output.clone())
            )
            self.addCleanup(handle.remove)
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
                actual_logits = logits[-1]
                teacher_input = torch.cat((prompt, inputs), dim=1)
                expected_logits = self.model(teacher_input, use_cache=False).logits[:, 7:]
                torch.testing.assert_close(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)
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
                actual_logits = logits[-1]
                expected_logits = self.model(prefix, use_cache=False).logits[:, -1:]
                torch.testing.assert_close(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)
                expected = expected_logits[:, -1].argmax(-1).item()
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
