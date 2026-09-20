"""Fixed H100 decode projection choices, measured in full generation.

Engine startup does not run calibration. The calibration helpers remain for
offline experiments; ordinary execution uses the fixed dispatch below.
Projections accumulate FP32 and preserve the native BF16 rounding boundaries.
"""
import json

import torch
import triton
import triton.language as tl
from triton.ops.matmul import _kernel as _upstream_matmul

from kernels.dotgemv import dot_projection

# Bypass the upstream autotuner/heuristics with the measured configurations.
# https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/ops/matmul.py
_matmul = _upstream_matmul.fn.fn

DOT_CONFIGS = ((128, 64, 8, 4), (64, 128, 8, 4))
# Upstream-matmul (N tile, K tile, warps, stages) grid for calibration. The
# frozen per-kind picks sit inside this grid; EVEN_K masking in the upstream
# kernel handles any K remainder, so no divisibility filter is needed.
TILE_CANDIDATES = (
    (32, 128, 2, 5), (64, 128, 4, 5), (128, 64, 4, 3), (64, 64, 4, 4),
    (128, 64, 8, 4), (64, 128, 8, 4), (256, 64, 8, 3), (32, 64, 4, 5),
)
_choices = {}
_forced = {}
_probes = {}
_probing = False


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


def _legacy(kind, rows):
    column, tile = configuration(kind, rows)
    return ("tile", column, tile) if tile is not None else ("linear", column)


def _measure(step, iters=8, warmup=2):
    for _ in range(warmup):
        step()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def probe_point(kind, instance, x, rows):
    """Register a calibration point for any module with the probe interface.

    The instance must provide _candidates(rows) with the frozen choice
    first, _run(choice, x, rows), and optionally _reference(x) for the
    numeric check (native linear against .weight otherwise).
    """
    if _probing and kind not in _probes:
        _probes[kind] = (instance, x.detach().clone(), rows)


def resolve(kind, rows):
    """Return an offline override or the fixed production choice."""
    choice = _forced.get(kind)
    if choice is None:
        choice = _choices.get((kind, rows))
    if choice is None:
        if kind == "down_next_norm" and 2 <= rows <= 32:
            return ("fused", 16)
        if kind == "down_residual" and 2 <= rows <= 32:
            return ("splitk",)
        if kind == "rope_attention":
            if rows == 8:
                return ("sglang", 8)
            if rows == 16:
                return ("sglang", 4)
            return ("fused",)
    return choice


def calibrate(step, graph_factory=None, margin=0.005):
    """Pick the fastest projection per kind by timing whole decode steps.

    `step` runs one eager step and leaves the state reusable; it identifies
    which kinds and row counts this state exercises. When `graph_factory`
    is given it must capture the step into a fresh CUDA graph and return a
    replay callable: graphed replays are the only timing that matches the
    production regime (eager steps are launch-bound and misrank kernels).
    The frozen configuration wins any comparison within `margin` to reduce
    sensitivity to timing noise. Call before capturing the production graph.
    """
    global _probing
    _probes.clear()
    _probing = True
    step()
    _probing = False
    torch.cuda.synchronize()
    for kind, (instance, x, rows) in _probes.items():
        if (kind, rows) in _choices:
            continue
        reference_fn = getattr(instance, "_reference", None)
        reference = (reference_fn(x) if reference_fn is not None
                     else torch.nn.functional.linear(x, instance.weight)).float()
        scale = max(reference.abs().max().item(), 1.0)
        options = instance._candidates(rows)
        legacy = options[0]
        timings = []
        for choice in options:
            try:
                output = instance._run(choice, x, rows).float()
                if (output - reference).abs().max().item() > 0.05 * scale:
                    continue
                _forced[kind] = choice
                if graph_factory is not None:
                    replay = graph_factory()
                    elapsed = _measure(replay, iters=20, warmup=4)
                    del replay
                else:
                    elapsed = _measure(step)
                timings.append((elapsed, choice))
            except Exception:
                continue
            finally:
                _forced.pop(kind, None)
        if not timings:
            continue
        timings.sort(key=lambda pair: pair[0])
        elapsed, best = timings[0]
        legacy_elapsed = [t for t, c in timings if c == legacy]
        if legacy_elapsed and legacy_elapsed[0] <= elapsed * (1 + margin):
            elapsed, best = legacy_elapsed[0], legacy
        _choices[(kind, rows)] = best
        print(json.dumps(dict(kind="projection_calibration", projection=kind,
                              rows=rows, choice=str(best),
                              step_ms=[[round(t, 4), str(c)] for t, c in timings[:3]])),
              flush=True)


class Projection(torch.nn.Module):
    def __init__(self, weight, kind):
        super().__init__()
        self.weight = weight
        self.kind = kind
        self.register_buffer("column_weight", None, persistent=False)

    def _column(self):
        if self.column_weight is None:
            self.column_weight = self.weight.detach().T.contiguous().T
        return self.column_weight

    def _tile(self, x, rows, column, tile):
        weight = self._column() if column else self.weight
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

    def _dot(self, x, rows, config, column):
        n, k = self.weight.shape
        bn, bk, warps, stages = config
        weight = self._column().T if column else self.weight
        flat = x.reshape(rows, k)
        out = dot_projection(flat, weight, rows, n, k, bn, bk, warps, stages, column)
        return out.reshape(*x.shape[:-1], n)

    def _run(self, choice, x, rows):
        if choice[0] == "tile":
            return self._tile(x, rows, choice[1], choice[2])
        if choice[0] == "dot":
            return self._dot(x, rows, choice[1], choice[2])
        weight = self._column() if choice[1] else self.weight
        return torch.nn.functional.linear(x, weight)

    def _candidates(self, rows):
        options = [_legacy(self.kind, rows)]
        for column in (False, True):
            choice = ("linear", column)
            if choice not in options:
                options.append(choice)
        for tile in TILE_CANDIDATES:
            for column in (False, True):
                choice = ("tile", column, tile)
                if choice not in options:
                    options.append(choice)
        _, k = self.weight.shape
        for config in DOT_CONFIGS:
            if k % config[1]:
                continue
            options.append(("dot", config, False))
            options.append(("dot", config, True))
        return options

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        if rows > 32:
            return torch.nn.functional.linear(x, self.weight)
        if _probing and self.kind not in _probes:
            _probes[self.kind] = (self, x.detach().clone(), rows)
        choice = _forced.get(self.kind)
        if choice is None:
            choice = _choices.get((self.kind, rows))
        if choice is None:
            choice = _legacy(self.kind, rows)
        return self._run(choice, x, rows)
