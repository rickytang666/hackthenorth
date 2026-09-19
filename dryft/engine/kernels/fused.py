"""Fused Q/K normalization + RoPE and SwiGLU with native BF16 cast boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_row(X, W, Y, COS, SIN, row, HEADS: tl.constexpr,
                   TOKENS: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
                   BATCH_STRIDE: tl.constexpr, TOKEN_STRIDE: tl.constexpr):
    dim = tl.arange(0, D)
    other = (dim + D // 2) % D
    token = (row // HEADS) % TOKENS
    batch = row // (HEADS * TOKENS)
    base = batch * BATCH_STRIDE + token * TOKEN_STRIDE + (row % HEADS) * D
    x = tl.load(X + base + dim).to(tl.float32)
    rotated = tl.load(X + base + other).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    weight = tl.load(W + dim).to(tl.float32)
    other_weight = tl.load(W + other).to(tl.float32)
    dtype: tl.constexpr = Y.dtype.element_ty
    normed = ((x * inv).to(dtype).to(tl.float32) * weight).to(dtype).to(tl.float32)
    rotated = ((rotated * inv).to(dtype).to(tl.float32) * other_weight).to(dtype).to(tl.float32)
    rotated = tl.where(dim < D // 2, -rotated, rotated)
    cos = tl.load(COS + token * D + dim).to(tl.float32)
    sin = tl.load(SIN + token * D + dim).to(tl.float32)
    left = (normed * cos).to(dtype).to(tl.float32)
    right = (rotated * sin).to(dtype).to(tl.float32)
    tl.store(Y + row * D + dim, left + right)


@triton.jit
def _norm_rope(Q, K, WQ, WK, OQ, OK, COS, SIN, Q_ROWS: tl.constexpr,
               Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, TOKENS: tl.constexpr,
               D: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr,
               Q_BATCH: tl.constexpr, Q_STRIDE: tl.constexpr,
               K_BATCH: tl.constexpr, K_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    if row < Q_ROWS:
        _norm_rope_row(Q, WQ, OQ, COS, SIN, row, Q_HEADS, TOKENS, D, Q_EPS, Q_BATCH, Q_STRIDE)
    else:
        _norm_rope_row(K, WK, OK, COS, SIN, row - Q_ROWS, K_HEADS, TOKENS, D, K_EPS, K_BATCH, K_STRIDE)


def norm_rope(q, k, q_norm, k_norm, cos, sin):
    """Q/K [B,T,H,D], including packed projection views, to contiguous outputs."""
    b, tokens, q_heads, dim = q.shape
    k_heads = k.shape[2]
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _norm_rope[(b * tokens * (q_heads + k_heads),)](
        q, k, q_norm.weight, k_norm.weight, out_q, out_k, cos, sin,
        b * tokens * q_heads, q_heads, k_heads, tokens, dim,
        q_norm.variance_epsilon, k_norm.variance_epsilon,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        num_warps=4, enable_fp_fusion=False,
    )
    return out_q, out_k


@triton.jit
def _swiglu(GATE, UP, OUT, N: tl.constexpr, COLS: tl.constexpr,
             GATE_STRIDE: tl.constexpr, UP_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // COLS, index % COLS
    gate = tl.load(GATE + row * GATE_STRIDE + col, mask=index < N, other=0.).to(tl.float32)
    up = tl.load(UP + row * UP_STRIDE + col, mask=index < N, other=0.).to(tl.float32)
    activation = (gate * tl.sigmoid(gate)).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + index, activation * up, mask=index < N)


def swiglu(gate, up):
    out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    cols = gate.shape[-1]
    gate_rows, up_rows = gate.reshape(-1, cols), up.reshape(-1, cols)
    _swiglu[(triton.cdiv(gate.numel(), 1024),)](
        gate_rows, up_rows, out, gate.numel(), cols, gate_rows.stride(0), up_rows.stride(0), 1024,
    )
    return out
