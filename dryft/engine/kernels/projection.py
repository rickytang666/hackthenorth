"""Fused BF16 hidden RMSNorm and small-batch projections.

Inputs and weights are contiguous BF16 CUDA tensors. The kernels read all
inputs, return new outputs, and preserve the native normalization/projection/
activation BF16 rounding boundaries. No inputs or weights are mutated.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm_projection(X, GAIN, W, Y, K: tl.constexpr, N: tl.constexpr,
                     EPS: tl.constexpr, SWIGLU: tl.constexpr,
                     BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    k = tl.arange(0, BLOCK_K)
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    gain = tl.load(GAIN + k, k < K, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
    normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
    normalized = (normalized * gain).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W + n[:, None] * K + k[None, :],
                     (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    projected = tl.sum(weight * normalized[None, :], 1).to(tl.bfloat16)
    if SWIGLU:
        up_weight = tl.load(W + (n[:, None] + N) * K + k[None, :],
                            (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        up = tl.sum(up_weight * normalized[None, :], 1).to(tl.bfloat16)
        gate = projected.to(tl.float32)
        activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
        projected = (activated.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + n, projected, n < N)


def norm_projection(x, gain, weight, eps=1e-6, *, swiglu=False, block_n=4, warps=8):
    """Normalize one hidden row and project, optionally through packed SwiGLU."""
    if x.numel() != x.shape[-1] or not x.is_contiguous():
        raise ValueError("the batch-one prototype requires one contiguous hidden row")
    n, k = weight.shape
    if x.shape[-1] != k or gain.numel() != k or not weight.is_contiguous():
        raise ValueError("incompatible projection dimensions or weight stride")
    if swiglu:
        if n % 2:
            raise ValueError("packed gate/up width must be even")
        n //= 2
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    _norm_projection[(triton.cdiv(n, block_n),)](
        x, gain, weight, out, k, n, eps, swiglu,
        triton.next_power_of_2(k), block_n, num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _output_residual(X, W, RES, OUT, K: tl.constexpr, N: tl.constexpr,
                     BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    k = tl.arange(0, BLOCK_K)
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    activation = tl.load(X + k, k < K, 0).to(tl.float32)
    weight = tl.load(W + n[:, None] * K + k[None, :],
                     (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    # Native projection rounds before residual addition; the store rounds again.
    projected = tl.sum(weight * activation[None, :], 1).to(tl.bfloat16).to(tl.float32)
    residual = tl.load(RES + n, n < N, 0).to(tl.float32)
    tl.store(OUT + n, projected + residual, n < N)


def output_projection_residual(x, weight, residual):
    """Batch-one BF16 attention output projection and residual in one launch."""
    n, k = weight.shape
    if x.numel() != k or residual.numel() != n:
        raise ValueError("output projection requires one activation and residual row")
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous()
           for t in (x, weight, residual)):
        raise ValueError("output projection requires contiguous BF16 tensors")
    out = torch.empty_like(residual)
    _output_residual[(triton.cdiv(n, 2),)](
        x, weight, residual, out, k, n, triton.next_power_of_2(k), 2,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
