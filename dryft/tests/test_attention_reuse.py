"""Attention calibration must compare identical KV contents across timings."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA and Triton")
class AttentionReuseTests(unittest.TestCase):
    def test_calibration_checks_survive_live_cache_and_position_changes(self):
        from attention import _RopeAttention
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        with torch.inference_mode():
            torch.manual_seed(4923)
            ref = SimpleNamespace(head_dim=128, layer_idx=0,
                                  q_norm=Qwen3RMSNorm(128).cuda().bfloat16(),
                                  k_norm=Qwen3RMSNorm(128).cuda().bfloat16())
            x = torch.randn(4, 1, 6144, device="cuda", dtype=torch.bfloat16)
            keys = torch.randn(4, 8, 129, 128, device="cuda", dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            positions = torch.tensor([64], device="cuda")
            angle = torch.randn(1, 1, 128, device="cuda")
            embeddings = (angle.cos().bfloat16(), angle.sin().bfloat16())
            cache = SimpleNamespace(keys=[keys], values=[values])
            adapter = _RopeAttention(ref, (4096, 1024, 1024), x, cache, embeddings, positions)
            expected = adapter._reference(x).clone()

            # Simulate the graph timing between two calibration candidates.
            keys.normal_()
            values.normal_()
            positions.fill_(128)
            live_keys, live_values = keys.clone(), values.clone()
            for choice in adapter._candidates(4):
                actual = adapter._run(choice, x, 4)
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
            torch.testing.assert_close(keys, live_keys, atol=0, rtol=0)
            torch.testing.assert_close(values, live_values, atol=0, rtol=0)
            self.assertEqual(positions.item(), 128)

            # A decode adapter still consumes the current cache, not the snapshot.
            current = _RopeAttention(ref, (4096, 1024, 1024), x, cache, embeddings, positions)
            actual = current._run(("sglang", 4), x, 4)
            self.assertGreater(float((actual - expected).abs().max()), 0.05)


if __name__ == "__main__":
    unittest.main()
