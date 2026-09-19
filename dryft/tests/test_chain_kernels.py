"""Persistent chains must preserve all QKV channels across graph replays."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA and Triton")
class ChainKernelTests(unittest.TestCase):
    @torch.inference_mode() if torch is not None else lambda fn: fn
    def test_full_qkv_width_and_changed_inputs_across_replays(self):
        from kernels.chain import chain_a, chain_b
        from kernels.down import down_residual
        from kernels.projection import norm_projection

        torch.manual_seed(3101)
        options = dict(device="cuda", dtype=torch.bfloat16)
        att = torch.randn(1, 1, 4096, **options)
        hidden = torch.randn(1, 1, 2560, **options)
        wo = torch.randn(2560, 4096, **options) * .02
        gu = torch.randn(19456, 2560, **options) * .02
        interleaved = torch.stack(gu.chunk(2, 0), 1).contiguous()
        down = torch.randn(2560, 9728, **options) * .02
        qkv = torch.randn(6144, 2560, **options) * .02
        gain = torch.ones(2560, **options)

        def step():
            # Exactly the production launch counts: 36 A, 35 B. Both
            # barrier rings must return to their starting slot per replay.
            for layer in range(36):
                intermediate, residual = chain_a(att, wo, hidden, gain, interleaved, 1e-6)
                if layer < 35:
                    result, packed = chain_b(intermediate, down, residual, gain, qkv, 1e-6)
            return result, packed

        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_hidden, actual_qkv = step()
        self.assertEqual(actual_qkv.shape, (1, 1, 6144))
        for scale in (.25, 1., 4.):
            att.normal_(std=scale)
            hidden.normal_()
            residual = hidden + torch.nn.functional.linear(att, wo)
            intermediate = norm_projection(residual, gain, gu, 1e-6,
                                           swiglu=True, block_n=1, warps=4)
            expected_hidden = down_residual(intermediate, down, residual)
            expected_qkv = norm_projection(expected_hidden, gain, qkv, 1e-6,
                                          block_n=2, warps=4)
            for _ in range(4):
                graph.replay()
            torch.testing.assert_close(actual_hidden, expected_hidden, atol=.04, rtol=.02)
            torch.testing.assert_close(actual_qkv, expected_qkv, atol=.04, rtol=.02)


if __name__ == "__main__":
    unittest.main()
