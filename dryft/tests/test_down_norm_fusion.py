"""Cross-layer normalization must preserve BF16 values on graph replay."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA and Triton")
class DownNormFusionTests(unittest.TestCase):
    def test_residual_norm_preserves_separate_values_on_replay(self):
        from kernels.rmsnorm import add_rms_norm, rms_norm

        with torch.inference_mode():
            torch.manual_seed(6923)
            gain = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
            for batch in (2, 4, 8, 16, 32):
                x = torch.randn(batch, 1, 2560, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                add_rms_norm(x, residual, gain, 1e-6, match_separate=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, normalized = add_rms_norm(x, residual, gain, 1e-6, match_separate=True)
                for scale in (0., 0.01, 1., 16.):
                    x.normal_().mul_(scale)
                    residual.normal_().mul_(scale)
                    gain.normal_()
                    expected = x + residual
                    expected_norm = rms_norm(expected, gain, 1e-6)
                    graph.replay()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(normalized, expected_norm, atol=0, rtol=0)

    def test_matches_separate_operations_with_new_graph_inputs(self):
        from kernels.split_down import split_down_residual, split_down_residual_norm
        from kernels.rmsnorm import rms_norm

        with torch.inference_mode():
            torch.manual_seed(6401)
            weight = torch.randn(2560, 9728, device="cuda", dtype=torch.bfloat16)
            weight = (weight * 0.02).T.contiguous().T
            gain = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
            for batch in (2, 4, 7, 16, 32):
                x = torch.randn(batch, 9728, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn(batch, 1, 2560, device="cuda", dtype=torch.bfloat16)
                split_down_residual_norm(x, weight, residual, gain, 1e-6)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, normalized = split_down_residual_norm(x, weight, residual, gain, 1e-6)
                for scale in (0., 0.01, 1., 16.):
                    x.normal_().mul_(scale)
                    residual.normal_().mul_(scale)
                    gain.normal_()
                    expected = split_down_residual(x, weight, residual.reshape(batch, -1)).reshape_as(residual)
                    expected_norm = rms_norm(expected, gain, 1e-6)
                    graph.replay()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(normalized, expected_norm, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
