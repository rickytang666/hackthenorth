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
    def test_fused_attention_prologue_handles_split_edges_and_poisoned_tail(self):
        from kernels.attention import grouped_attention, prologue_attention
        from kernels.fused import norm_rope_cache
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        torch.manual_seed(2501)
        with torch.inference_mode():
            for b, capacity, positions in ((1, 257, (0, 63, 64, 256)),
                                            (4, 2080, (0, 511, 512, 2049)),
                                            (16, 640, (0, 511, 512, 639))):
                packed = torch.randn(b, 1, 6144, device="cuda", dtype=torch.bfloat16)
                q, k, v = packed.split((4096, 1024, 1024), -1)
                q = q.view(b, 1, 32, 128)
                k, v = k.view(b, 1, 8, 128), v.view(b, 1, 8, 128)
                qn = Qwen3RMSNorm(128).cuda().bfloat16()
                kn = Qwen3RMSNorm(128).cuda().bfloat16()
                qn.weight.normal_()
                kn.weight.normal_()
                angle = torch.randn(1, 1, 128, device="cuda")
                cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
                keys = torch.zeros(b, 8, capacity, 128, device="cuda", dtype=torch.bfloat16)
                values = torch.zeros_like(keys)
                pos = torch.zeros(1, device="cuda", dtype=torch.long)
                def fused():
                    return prologue_attention(packed, qn, kn, cos, sin, keys, values,
                                              pos, 8, 4, 4096, 5120)
                fused()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = fused()
                for position in positions:
                    pos.fill_(position)
                    packed.normal_()
                    keys.normal_()
                    values.normal_()
                    keys[:, :, position:].fill_(float("nan"))
                    values[:, :, position:].fill_(float("nan"))
                    ek, ev = keys.clone(), values.clone()
                    rq, rk, rv = norm_rope_cache(q, k, v, qn, kn, cos, sin, ek, ev, pos)
                    expected = grouped_attention(rq, rk, rv, pos)
                    graph.replay()
                    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
                    torch.testing.assert_close(keys, ek, atol=0.02, rtol=0.02, equal_nan=True)
                    torch.testing.assert_close(values, ev, atol=0, rtol=0, equal_nan=True)

    def test_projection_fusions_preserve_rounding_with_changed_graph_inputs(self):
        from kernels.dotgemv import dot_projection_residual, norm_dot_projection
        from kernels.rmsnorm import rms_norm

        torch.manual_seed(2502)
        with torch.inference_mode():
            for rows in (4, 16, 24):
                x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                gain = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(257, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
                residual = torch.randn(rows, 257, device="cuda", dtype=torch.bfloat16)
                for column in (False, True):
                    weight = w.T.contiguous() if column else w
                    def run():
                        return (dot_projection_residual(x, weight, residual, rows, 257, 2560,
                                                       64, 128, 8, 4, column),
                                norm_dot_projection(x, gain, 1e-6, weight, rows, 257, 2560,
                                                    64, 128, 8, 4, column))
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for scale in (0., 0.25, 4.):
                        x.normal_(std=scale)
                        residual.normal_()
                        graph.replay()
                        expected = (torch.nn.functional.linear(x, w) + residual,
                                    torch.nn.functional.linear(rms_norm(x, gain, 1e-6), w))
                        for a, e in zip(actual, expected):
                            torch.testing.assert_close(a, e, atol=0.04, rtol=0.02)

    def test_tiled_norm_rope_matches_row_kernel_on_changed_packed_inputs(self):
        from kernels.fused import norm_rope
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        torch.manual_seed(1801)
        with torch.inference_mode():
            for batch, tokens, q_heads, k_heads in ((1, 1, 32, 8), (2, 17, 32, 8),
                                                   (1, 3, 4, 2), (1, 3, 16, 3)):
                dim = 128
                packed = torch.randn(batch, tokens, (q_heads + 2*k_heads)*dim,
                                     device="cuda", dtype=torch.bfloat16)
                q, k, _ = packed.split((q_heads*dim, k_heads*dim, k_heads*dim), -1)
                q = q.view(batch, tokens, q_heads, dim)
                k = k.view(batch, tokens, k_heads, dim)
                q_norm = Qwen3RMSNorm(dim).cuda().bfloat16()
                k_norm = Qwen3RMSNorm(dim).cuda().bfloat16()
                q_norm.weight.normal_()
                k_norm.weight.normal_()
                angles = torch.randn(1, tokens, dim, device="cuda")
                cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
                norm_rope(q, k, q_norm, k_norm, cos, sin)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = norm_rope(q, k, q_norm, k_norm, cos, sin)
                for scale in (0., 0.25, 4.):
                    packed.normal_(std=scale)
                    graph.replay()
                    expected = norm_rope(q, k, q_norm, k_norm, cos, sin, tile=False)
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right, atol=0.02, rtol=0.02)

    def test_dot_projection_layouts_tail_rows_and_graph_replay(self):
        from kernels.dotgemv import dot_projection

        torch.manual_seed(1731)
        with torch.inference_mode():
            for rows, k in ((1, 2560), (3, 4096), (17, 9728), (32, 2560)):
                # An odd output width checks masked channels as well as the
                # padded tensor-core rows in the calibration candidate.
                n = 257
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                for column in (False, True):
                    packed = weight.T.contiguous() if column else weight
                    for bn, bk in ((128, 64), (64, 128)):
                        def project():
                            return dot_projection(x, packed, rows, n, k, bn, bk, 8, 4, column)
                        project()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = project()
                        for scale in (0.25, 4.):
                            x.normal_(std=scale)
                            graph.replay()
                            expected = torch.nn.functional.linear(x, weight)
                            torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)

    def test_greedy_head_rounding_ties_and_graph_replay(self):
        from kernels.greedy import greedy_token

        torch.manual_seed(1709)
        with torch.inference_mode():
            weight=torch.randn(151936,2560,device="cuda",dtype=torch.bfloat16)*0.02
            for rows in (1,3,16,17,32,33):
                x=torch.randn(rows,1,2560,device="cuda",dtype=torch.bfloat16)
                greedy_token(x,weight)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual=greedy_token(x,weight)
                for scale in (0.,1.,4.):
                    x.normal_(std=scale)
                    graph.replay()
                    expected=torch.nn.functional.linear(x,weight).argmax(-1)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            weight.zero_()
            weight[3].fill_(1)
            weight[-1].fill_(1)
            x.fill_(1)
            self.assertTrue(bool((greedy_token(x[:3],weight)==3).all()))

    def test_vector_down_residual_native_bf16_rounding(self):
        from kernels.down import down_residual

        torch.manual_seed(1709)
        with torch.inference_mode():
            weight=torch.randn(2560,9728,device="cuda",dtype=torch.bfloat16)*0.02
            x=torch.randn(1,1,9728,device="cuda",dtype=torch.bfloat16)
            residual=torch.randn(1,1,2560,device="cuda",dtype=torch.bfloat16)
            down_residual(x,weight,residual)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual=down_residual(x,weight,residual)
            for scale in (0.25,1.,4.):
                x.normal_(std=scale)
                residual.normal_()
                graph.replay()
                expected=torch.nn.functional.linear(x,weight)+residual
                torch.testing.assert_close(actual,expected,atol=0.04,rtol=0.02)

    def test_gateup_epilogue_matches_native_under_graph_replay(self):
        from kernels.gateup import gate_up_swiglu

        torch.manual_seed(719)
        with torch.inference_mode():
            for rows, width in ((2,96), (3,9728), (16,9728), (17,96), (31,96), (32,9728)):
                weight = torch.randn(2*width,2560,device="cuda",dtype=torch.bfloat16)*0.02
                gate,up = weight.chunk(2,0)
                packed = torch.stack((gate.T,up.T),-1).flatten(1).contiguous()
                x = torch.randn(rows,1,2560,device="cuda",dtype=torch.bfloat16)
                gate_up_swiglu(x,packed)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = gate_up_swiglu(x,packed)
                for scale in (0.25,1.,4.):
                    x.normal_(std=scale)
                    graph.replay()
                    g,u = torch.nn.functional.linear(x,weight).chunk(2,-1)
                    expected = torch.nn.functional.silu(g)*u
                    torch.testing.assert_close(actual,expected,atol=0.04,rtol=0.02)

    def test_add_rms_norm_preserves_residual_rounding_and_inputs(self):
        from kernels.rmsnorm import add_rms_norm, rms_norm

        torch.manual_seed(19)
        for batch in (1, 2, 4, 8, 16, 24, 33):
            x = torch.randn(batch, 1, 2560, device='cuda', dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            gain = torch.randn(2560, device='cuda', dtype=torch.bfloat16)
            original_x, original_residual = x.clone(), residual.clone()
            summed, actual = add_rms_norm(x, residual, gain, 1e-6)
            torch.testing.assert_close(summed, x + residual, atol=0, rtol=0)
            torch.testing.assert_close(actual, rms_norm(x + residual, gain, 1e-6),
                                       atol=0.02, rtol=0.02)
            torch.testing.assert_close(x, original_x, atol=0, rtol=0)
            torch.testing.assert_close(residual, original_residual, atol=0, rtol=0)

    def test_norm_rope_cache_matches_separate_updates_under_graph_replay(self):
        from kernels.fused import norm_rope, norm_rope_cache
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        torch.manual_seed(111)
        with torch.inference_mode():
            q_norm = Qwen3RMSNorm(128).cuda().bfloat16()
            k_norm = Qwen3RMSNorm(128).cuda().bfloat16()
            q_norm.weight.normal_()
            k_norm.weight.normal_()
            for batch, tokens in ((1,1), (4,1), (16,1), (1,4), (3,4)):
                packed = torch.randn(batch,tokens,6144,device="cuda",dtype=torch.bfloat16)
                q,k,v = (x.view(batch,tokens,-1,128) for x in packed.split((4096,1024,1024),-1))
                angles = torch.randn(1,tokens,128,device="cuda")
                cos,sin = angles.cos().bfloat16(), angles.sin().bfloat16()
                positions = torch.arange(tokens,device="cuda")*2+3
                keys = torch.full((batch,8,41,128),-123.,device="cuda",dtype=torch.bfloat16)
                values = torch.full_like(keys,123.)
                expected_keys,expected_values = keys.clone(),values.clone()
                def call():
                    return norm_rope_cache(q,k,v,q_norm,k_norm,cos,sin,keys,values,positions)
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result,_,_ = call()
                for offset in (0,13,-13):
                    positions.add_(offset)
                    packed.normal_()
                    # Decode cache writes still use the row kernel. Keep this
                    # comparison bit-exact; tiled prefill is checked separately.
                    expected_q,expected_k = norm_rope(q,k,q_norm,k_norm,cos,sin,tile=False)
                    expected_keys.index_copy_(2,positions,expected_k.transpose(1,2))
                    expected_values.index_copy_(2,positions,v.transpose(1,2))
                    graph.replay()
                    torch.testing.assert_close(result,expected_q.transpose(1,2),atol=0,rtol=0)
                    torch.testing.assert_close(keys,expected_keys,atol=0,rtol=0)
                    torch.testing.assert_close(values,expected_values,atol=0,rtol=0)

    def test_flash_prefill_matches_reference_and_preserves_causality(self):
        from attention import GroupedAttention
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        class Cache:
            prefill = True

            def update(self, keys, values, layer_idx, cache_kwargs):
                self.keys, self.values = keys, values
                return keys, values

        torch.manual_seed(17)
        config = Qwen3Config(hidden_size=2560, num_attention_heads=32,
                             num_key_value_heads=8, head_dim=128)
        config._attn_implementation = "sdpa"
        with torch.inference_mode():
            reference = Qwen3Attention(config, 0).cuda().bfloat16().eval()
            attention = GroupedAttention(reference)
            for batch, length in ((1, 1), (2, 17), (4, 129)):
                x = torch.randn(batch, length, 2560, device="cuda", dtype=torch.bfloat16)
                angles = torch.randn(1, length, 128, device="cuda")
                embeddings = angles.cos().bfloat16(), angles.sin().bfloat16()
                positions = torch.arange(length, device="cuda")
                native_cache, flash_cache = Cache(), Cache()
                expected = reference(x, embeddings, None, past_key_value=native_cache,
                                     cache_position=positions)[0]
                actual = attention(x, embeddings, past_key_value=flash_cache,
                                   cache_position=positions)[0]
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                self.assertEqual(flash_cache.keys.shape, (batch, 8, length, 128))
                torch.testing.assert_close(flash_cache.keys, native_cache.keys, atol=0.02, rtol=0.02)
                torch.testing.assert_close(flash_cache.values, native_cache.values, atol=0.02, rtol=0.02)
                if length > 1:
                    split = length // 2
                    x[:, split:].normal_()
                    changed = attention(x, embeddings, past_key_value=Cache(),
                                        cache_position=positions)[0]
                    torch.testing.assert_close(changed[:, :split], actual[:, :split], atol=0, rtol=0)
                    mask = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
                    expected = reference(x, embeddings, mask, past_key_value=Cache(),
                                         cache_position=positions)[0]
                    actual = attention(x, embeddings, mask, past_key_value=Cache(),
                                       cache_position=positions)[0]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

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
