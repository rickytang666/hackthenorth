"""Fused Q/K normalization + RoPE and SwiGLU with native BF16 cast boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_row(X, W, Y, COS, SIN, row, HEADS: tl.constexpr,
                   TOKENS: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    dim = tl.arange(0, D)
    other = (dim + D // 2) % D
    x = tl.load(X + row * D + dim).to(tl.float32)
    rotated = tl.load(X + row * D + other).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    weight = tl.load(W + dim).to(tl.float32)
    other_weight = tl.load(W + other).to(tl.float32)
    dtype: tl.constexpr = Y.dtype.element_ty
    normed = ((x * inv).to(dtype).to(tl.float32) * weight).to(dtype).to(tl.float32)
    rotated = ((rotated * inv).to(dtype).to(tl.float32) * other_weight).to(dtype).to(tl.float32)
    rotated = tl.where(dim < D // 2, -rotated, rotated)
    token = (row // HEADS) % TOKENS
    cos = tl.load(COS + token * D + dim).to(tl.float32)
    sin = tl.load(SIN + token * D + dim).to(tl.float32)
    left = (normed * cos).to(dtype).to(tl.float32)
    right = (rotated * sin).to(dtype).to(tl.float32)
    tl.store(Y + row * D + dim, left + right)


@triton.jit
def _norm_rope(Q, K, WQ, WK, OQ, OK, COS, SIN, Q_ROWS: tl.constexpr,
               Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, TOKENS: tl.constexpr,
               D: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr):
    row = tl.program_id(0)
    if row < Q_ROWS:
        _norm_rope_row(Q, WQ, OQ, COS, SIN, row, Q_HEADS, TOKENS, D, Q_EPS)
    else:
        _norm_rope_row(K, WK, OK, COS, SIN, row - Q_ROWS, K_HEADS, TOKENS, D, K_EPS)


def norm_rope(q, k, q_norm, k_norm, cos, sin):
    """Contiguous Q/K [B,T,H,D] to normalized, rotated tensors of the same shape."""
    b, tokens, q_heads, dim = q.shape
    k_heads = k.shape[2]
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    _norm_rope[(b * tokens * (q_heads + k_heads),)](
        q, k, q_norm.weight, k_norm.weight, out_q, out_k, cos, sin,
        b * tokens * q_heads, q_heads, k_heads, tokens, dim,
        q_norm.variance_epsilon, k_norm.variance_epsilon,
        num_warps=4, enable_fp_fusion=False,
    )
    return out_q, out_k


@triton.jit
def _swiglu(GATE, UP, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(GATE + index, mask=index < N, other=0.).to(tl.float32)
    up = tl.load(UP + index, mask=index < N, other=0.).to(tl.float32)
    activation = (gate * tl.sigmoid(gate)).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + index, activation * up, mask=index < N)


def swiglu(gate, up):
    out = torch.empty_like(gate)
    _swiglu[(triton.cdiv(gate.numel(), 1024),)](gate, up, out, gate.numel(), 1024)
    return out
