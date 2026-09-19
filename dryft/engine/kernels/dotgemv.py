"""Software-pipelined tensor-core GEMM for small decode row counts.

Inputs are contiguous BF16 CUDA tensors. Accumulation is FP32 with one BF16
rounding at the store, the same boundary as native linear. Rows are masked
at load and store, so callers never pad. The column layout reads a (K, N)
contiguous transpose, which shares storage with the projection module's
cached column weight.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dot_tile(X, W, Y, R, N, K: tl.constexpr, MP: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, COLUMN: tl.constexpr):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, MP)
    live = rows[:, None] < R
    accumulator = tl.zeros((MP, BLOCK_N), tl.float32)
    for offset in range(0, K, BLOCK_K):
        k = offset + tl.arange(0, BLOCK_K)
        x = tl.load(X + rows[:, None] * K + k[None, :], live, 0.0)
        if COLUMN:
            weight = tl.load(W + k[:, None] * N + n[None, :], n[None, :] < N, 0.0)
            accumulator = tl.dot(x, weight, accumulator)
        else:
            weight = tl.load(W + n[:, None] * K + k[None, :], n[:, None] < N, 0.0)
            accumulator = tl.dot(x, tl.trans(weight), accumulator)
    tl.store(Y + rows[:, None] * N + n[None, :], accumulator.to(tl.bfloat16),
             live & (n[None, :] < N))


def dot_projection(x, weight, rows, n, k, block_n, block_k, warps, stages, column):
    """Project up to 32 rows through tensor cores; weight is (N, K) or (K, N)."""
    if k % block_k:
        raise ValueError("K must divide the K tile")
    out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    padded = max(triton.next_power_of_2(rows), 16)
    _dot_tile[(triton.cdiv(n, block_n),)](
        x, weight, out, rows, n, k, padded, block_n, block_k, column,
        num_warps=warps, num_stages=stages,
    )
    return out
