"""BF16 gate/up GEMM with an interleaved SwiGLU epilogue.

Uses the tiled GEMM approach from Triton 3.1's matmul. Weights are packed
offline as [K, I, 2] (gate, up); FP32 dot accumulators round to BF16 before
SiLU, and SiLU rounds to BF16 before the multiplication by up.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gateup(X, W, Y, M: tl.constexpr, K: tl.constexpr, I: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    row = (tile % tiles_m) * BM + tl.arange(0, BM)
    col = (tile // tiles_m) * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    xp = X + (row % M)[:, None] * K + kr[None, :]
    wp = W + kr[:, None] * (2 * I) + col[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // BK):
        x = tl.load(xp)
        w = tl.load(wp, col[None, :] < 2 * I, 0)
        acc = tl.dot(x, w, acc)
        xp += BK
        wp += BK * (2 * I)
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    gate, up = tl.split(tl.reshape(rounded, (BM, BN // 2, 2)))
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    out_col = (tile // tiles_m) * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(Y + row[:, None] * I + out_col[None, :], activated * up,
             (row[:, None] < M) & (out_col[None, :] < I))


def gate_up_swiglu(x, interleaved_weight):
    """BF16 contiguous x[...,K] and interleaved weight[K,2*I] -> x[...,I].

    Decode-only: 2–32 rows, K divisible by 64. The caller packs weights
    once during warmup; inputs and weights are never mutated.
    """
    rows = x.numel() // x.shape[-1]
    width = interleaved_weight.shape[1] // 2
    out = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    _gateup[(triton.cdiv(rows, 16) * triton.cdiv(2 * width, 128),)](
        x, interleaved_weight, out, rows, x.shape[-1], width, 16, 128, 64,
        num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out
