"""GPU numerical checks for the submitted BF16 kernels."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA and Triton")
class FusedKernelTests(unittest.TestCase):
    def test_projection_dispatch_boundaries(self):
        from projections import Projection

        torch.manual_seed(2026)
        with torch.inference_mode():
            for kind, n, k in (("qkv",6144,2560), ("gate_up",19456,2560),
                               ("o",2560,4096), ("down",2560,9728)):
                weight = torch.nn.Parameter(torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*0.02,
                                            requires_grad=False)
                projection = Projection(weight, kind)
                for rows in (1,2,3,4,5,8,12,15,16,17,24,31,32,33,64):
                    x = torch.randn(rows,1,k,device="cuda",dtype=torch.bfloat16)
                    torch.testing.assert_close(projection(x), torch.nn.functional.linear(x,weight),
                                               atol=0.04,rtol=0.02)

    def test_normalized_projections_and_swiglu(self):
        from kernels.projection import norm_projection
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        torch.manual_seed(42)
        with torch.inference_mode():
            norm = Qwen3RMSNorm(2560).cuda().bfloat16()
            norm.weight.normal_()
            for batch in (1,):
                x = torch.randn(batch, 1, 2560, device="cuda", dtype=torch.bfloat16)
                fn = norm_projection
                for width, activation in ((6144, False), (19456, True)):
                    weight = torch.randn(width, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
                    expected = torch.nn.functional.linear(norm(x), weight)
                    if activation:
                        gate, up = expected.chunk(2, -1)
                        expected = torch.nn.functional.silu(gate) * up
                    actual = fn(x, norm.weight, weight, norm.variance_epsilon,
                                swiglu=activation, block_n=1 if activation else 2, warps=4)
                    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)

    def test_grouped_attention_and_offset_causality(self):
        from kernels.attention import grouped_attention

        torch.manual_seed(3)
        with torch.inference_mode():
            for batch, queries, first, capacity in ((1, 1, 0, 129), (1, 4, 63, 513), (4, 4, 126, 257)):
                q = torch.randn(batch, queries, 32, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                k = torch.randn(batch, 8, capacity, 128, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                positions = torch.arange(first, first + queries, device="cuda")
                mask = torch.arange(capacity, device="cuda")[None, :] <= positions[:, None]
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1),
                    attn_mask=mask[None, None],
                )
                actual = grouped_attention(q, k, v, positions)
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                # Uninitialized capacity must not be read even when it has NaNs.
                k[:, :, first + queries:].fill_(float("nan"))
                v[:, :, first + queries:].fill_(float("nan"))
                torch.testing.assert_close(grouped_attention(q, k, v, positions), actual, atol=0, rtol=0)

    def test_norm_rope_preserves_reference_rounding(self):
        from kernels.fused import norm_rope
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb

        with torch.inference_mode():
            q_norm = Qwen3RMSNorm(128).cuda().bfloat16()
            k_norm = Qwen3RMSNorm(128).cuda().bfloat16()
            q_norm.weight.normal_()
            k_norm.weight.normal_()
            q = torch.randn(2, 4, 32, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, 4, 8, 128, device="cuda", dtype=torch.bfloat16)
            angles = torch.randn(1, 4, 128, device="cuda")
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            expected_q, expected_k = apply_rotary_pos_emb(
                q_norm(q).transpose(1, 2), k_norm(k).transpose(1, 2), cos, sin,
            )
            actual_q, actual_k = norm_rope(q, k, q_norm, k_norm, cos, sin)
            torch.testing.assert_close(actual_q.transpose(1, 2), expected_q, atol=0.02, rtol=0.02)
            torch.testing.assert_close(actual_k.transpose(1, 2), expected_k, atol=0.02, rtol=0.02)
            for batch, tokens in ((1, 1), (4, 1), (2, 4)):
                packed = torch.randn(batch, tokens, 6144, device="cuda", dtype=torch.bfloat16)
                q, k, _ = packed.split((4096, 1024, 1024), dim=-1)
                q, k = q.view(batch, tokens, 32, 128), k.view(batch, tokens, 8, 128)
                c, s = cos[:, :tokens].contiguous(), sin[:, :tokens].contiguous()
                expected_q, expected_k = apply_rotary_pos_emb(
                    q_norm(q).transpose(1, 2), k_norm(k).transpose(1, 2), c, s,
                )
                actual_q, actual_k = norm_rope(q, k, q_norm, k_norm, c, s)
                torch.testing.assert_close(actual_q.transpose(1, 2), expected_q, atol=0.02, rtol=0.02)
                torch.testing.assert_close(actual_k.transpose(1, 2), expected_k, atol=0.02, rtol=0.02)

    def test_swiglu_preserves_intermediate_bf16_cast(self):
        from kernels.fused import swiglu

        gate = torch.randn(4, 9728, device="cuda", dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(swiglu(gate, up), expected, atol=0.02, rtol=0.02)
        packed = torch.randn(2, 4, 19456, device="cuda", dtype=torch.bfloat16)
        gate, up = packed.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(swiglu(gate, up), expected, atol=0.02, rtol=0.02)
