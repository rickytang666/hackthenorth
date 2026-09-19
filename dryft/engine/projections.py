"""H100 decode projections using the matmul shipped with pinned Triton 3.1.

Tiles and weight layouts were measured with rotating weights beyond L2.
Prefill and larger batches retain native cuBLAS. Original BF16 weights remain
available for native prefill and the batch-one normalization fusion.
"""
import torch
import triton
import triton.language as tl
from triton.ops.matmul import _kernel as _upstream_matmul

# Bypass the upstream autotuner/heuristics with the measured configurations.
# https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/ops/matmul.py
_matmul = _upstream_matmul.fn.fn


def configuration(kind, rows):
    """Return column-layout flag and (N tile, K tile, warps, stages), if any."""
    if rows > 32:
        return False, None
    if kind == "down":
        return True, (32, 128, 2, 5) if 4 <= rows <= 16 else None
    if rows == 1:
        return False, None
    if kind == "qkv":
        return (True, (64, 128, 4, 5)) if rows <= 16 else (False, (32, 128, 2, 5))
    if kind == "gate_up":
        return True, (128, 64, 4, 3)
    if kind == "o":
        return rows <= 16, (64, 128, 4, 5) if rows == 2 else (32, 128, 2, 5)
    return False, None


class Projection(torch.nn.Module):
    def __init__(self, weight, kind):
        super().__init__()
        self.weight = weight
        self.kind = kind
        self.register_buffer("column_weight", None, persistent=False)

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        column, tile = configuration(self.kind, rows)
        weight = self.weight
        if column:
            if self.column_weight is None:
                self.column_weight = weight.detach().T.contiguous().T
            weight = self.column_weight
        if tile is None:
            return torch.nn.functional.linear(x, weight)
        n, k = weight.shape
        flat = x.reshape(rows, k)
        out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
        bn, bk, warps, stages = tile
        _matmul[(triton.cdiv(rows, 16) * triton.cdiv(n, bn), 1)](
            flat, weight, out, rows, n, k,
            flat.stride(0), flat.stride(1), weight.stride(1), weight.stride(0),
            out.stride(0), out.stride(1),
            acc_dtype=tl.float32, input_precision="ieee", fp8_fast_accum=True,
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLIT_K=1,
            EVEN_K=k % bk == 0, AB_DTYPE=tl.bfloat16,
            num_warps=warps, num_stages=stages,
        )
        return out.reshape(*x.shape[:-1], n)
