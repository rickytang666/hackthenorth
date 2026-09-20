"""Two-way BF16 down projection with FP32 merge and fused residual add.

Each CTA reduces half of K; the merge rounds the projection to BF16 before
adding the residual. This preserves the native BF16 boundaries while changing
FP32 summation order. Load-time calibration arbitrates against the old path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_down(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             WN: tl.constexpr, WK: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    tiles_m = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    rows = (tile % tiles_m) * BM + tl.arange(0, BM)
    cols = (tile // tiles_m) * BN + tl.arange(0, BN)
    chunks = tl.cdiv(tl.cdiv(K, BK), SPLITS)
    base_k = tl.program_id(1) * chunks * BK
    kr = base_k + tl.arange(0, BK)
    xp = X + (rows % M)[:, None] * K + kr[None, :]
    wp = W + kr[:, None] * WK + cols[None, :] * WN
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(chunks):
        x = tl.load(xp, kr[None, :] + block * BK < K, 0.)
        w = tl.load(wp, (kr[:, None] + block * BK < K) & (cols[None, :] < N), 0.)
        acc = tl.dot(x, w, acc)
        xp += BK
        wp += BK * WK
    tl.store(Y + tl.program_id(1) * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _finish(P, Y, RES, M: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr,
            ADD_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLITS)
    values = tl.load(P + split[:, None] * M * N + row * N + col[None, :],
                     col[None, :] < N, 0.)
    value = tl.sum(values, 0).to(tl.bfloat16).to(tl.float32)
    if ADD_RES:
        value += tl.load(RES + row * N + col, col < N, 0).to(tl.float32)
    tl.store(Y + row * N + col, value, col < N)


def split_down_residual(x, weight, residual):
    """Contiguous BF16 x[M,K], residual[M,N], and strided weight[N,K]."""
    m, k = x.shape
    n = weight.shape[0]
    partial = torch.empty((2, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _split_down[(triton.cdiv(m, 16) * triton.cdiv(n, 32), 2)](
        x, weight, partial, m, n, k, weight.stride(0), weight.stride(1),
        16, 32, 128, 2, num_warps=2, num_stages=5, enable_fp_fusion=False,
    )
    _finish[(m, triton.cdiv(n, 256))](
        partial, out, residual, m, n, 2, True, 256, num_warps=4,
    )
    return out


@triton.jit
def _finish_norm(P, RES, GAIN, Y, NORM, M: tl.constexpr, N,
                 EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < N
    a = tl.load(P + row * N + col, valid, 0.)
    b = tl.load(P + M * N + row * N + col, valid, 0.)
    projected = (a + b).to(tl.bfloat16).to(tl.float32)
    residual = tl.load(RES + row * N + col, valid, 0.).to(tl.float32)
    summed = (projected + residual).to(tl.bfloat16)
    tl.store(Y + row * N + col, summed, valid)
    value = summed.to(tl.float32)
    inverse = tl.rsqrt(tl.sum(value * value, 0) / N + EPS)
    normalized = (value * inverse).to(tl.bfloat16)
    gain = tl.load(GAIN + col, valid, 0.)
    tl.store(NORM + row * N + col, normalized * gain, valid)


def split_down_residual_norm(x, weight, residual, gain, eps, warps=16):
    """Also produce the next layer's normalized input in the merge kernel."""
    shape = residual.shape
    x = x.reshape(-1, x.shape[-1])
    m, k = x.shape
    n = weight.shape[0]
    partial = torch.empty((2, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    normalized = torch.empty_like(out)
    _split_down[(triton.cdiv(m, 16) * triton.cdiv(n, 32), 2)](
        x, weight, partial, m, n, k, weight.stride(0), weight.stride(1),
        16, 32, 128, 2, num_warps=2, num_stages=5, enable_fp_fusion=False)
    # Runtime N/EPS and default contraction match the existing RMSNorm kernel.
    _finish_norm[(m,)](
        partial, residual, gain, out, normalized, m, n, eps,
        triton.next_power_of_2(n), num_warps=warps)
    return out, normalized
