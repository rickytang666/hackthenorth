import torch
import triton
import triton.language as tl


@triton.jit
def _stats_tile(A, B, C, GAIN, RES, STATS, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                EPS: tl.constexpr, NORM: tl.constexpr, RESIDUAL: tl.constexpr,
                WRITE_STATS: tl.constexpr, READ_STATS: tl.constexpr, PARTS: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    inverse = tl.zeros((BLOCK_M,), tl.float32)
    if NORM:
        if READ_STATS:
            p = tl.arange(0, triton.next_power_of_2(PARTS))
            partial = tl.load(STATS + ram[:, None] * PARTS + p[None, :],
                              p[None, :] < PARTS, 0.0)
            inverse = tl.math.rsqrt(tl.sum(partial, 1) / K + EPS)
        else:
            # Per-row RMS statistic over the full activation row, FP32 like the
            # reference. Wrapped rows (ram) repeat the same statistic they load.
            sumsq = tl.zeros((BLOCK_M,), tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                if EVEN_K:
                    row = tl.load(A + ram[:, None] * stride_am
                                  + (k * BLOCK_K + rk)[None, :] * stride_ak)
                else:
                    row = tl.load(A + ram[:, None] * stride_am
                                  + (k * BLOCK_K + rk)[None, :] * stride_ak,
                                  mask=(k * BLOCK_K + rk)[None, :] < K, other=0.0)
                value = row.to(tl.float32)
                sumsq += tl.sum(value * value, 1)
            inverse = tl.math.rsqrt(sumsq / K + EPS)
    Aptr = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    Bptr = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(Aptr)
            b = tl.load(Bptr)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(Aptr, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(Bptr, mask=rk[:, None] < k_remaining, other=_0)
        if NORM:
            # Reference order: round the normalized value to BF16, then a
            # BF16 multiply by the gain.
            normalized = (a.to(tl.float32) * inverse[:, None]).to(tl.bfloat16)
            if EVEN_K:
                gain = tl.load(GAIN + k * BLOCK_K + rk)
            else:
                gain = tl.load(GAIN + k * BLOCK_K + rk,
                               mask=(k * BLOCK_K + rk) < K, other=0.0)
            a = normalized * gain[None, :]
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="ieee")
        Aptr += BLOCK_K * stride_ak
        Bptr += BLOCK_K * stride_bk
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    Cptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if RESIDUAL:
        projected = acc.to(tl.bfloat16).to(tl.float32)
        residual = tl.load(RES + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
                           mask, 0.0).to(tl.float32)
        summed = (projected + residual).to(tl.bfloat16)
        tl.store(Cptr, summed, mask=mask)
        if WRITE_STATS:
            values = tl.where(rn[None, :] < N, summed.to(tl.float32), 0.0)
            partial = tl.sum(values * values, 1)
            tl.store(STATS + rm * PARTS + pid_n, partial, rm < M)
    else:
        tl.store(Cptr, acc.to(C.dtype.element_ty), mask=mask)



def project(x, weight, tile, gain=None, eps=1e-6, residual=None,
            stats=None, write_stats=False):
    rows, k = x.shape
    n = weight.shape[0]
    bn, bk, warps, stages = tile
    out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    read_stats = stats is not None
    if write_stats:
        stats = torch.empty((rows, triton.cdiv(n, bn)), device=x.device, dtype=torch.float32)
    parts = 0 if stats is None else stats.shape[1]
    compiled = _stats_tile[(triton.cdiv(rows, 16) * triton.cdiv(n, bn),)](
        x, weight, out, x if gain is None else gain,
        out if residual is None else residual, out if stats is None else stats,
        rows, n, k, x.stride(0), x.stride(1), weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1), eps, gain is not None, residual is not None,
        write_stats, read_stats, parts, 16, bn, bk, 8, k % bk == 0,
        num_warps=warps, num_stages=stages)
    return out, stats, compiled
