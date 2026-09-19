"""Exact grouped-query attention over fixed KV storage, including short verifies.

Q is [batch, query_heads, queries, head_dim]; K/V are contiguous
[batch, kv_heads, capacity, head_dim]. Positions contains one absolute cache
position per query. Output has Q's shape and dtype; inputs are not mutated.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_attention(
    Q, K, V, POS, PART, LSE,
    Q_B: tl.constexpr, Q_H: tl.constexpr, Q_T: tl.constexpr,
    KV_HEADS: tl.constexpr, GROUPS: tl.constexpr, QUERIES: tl.constexpr,
    CAPACITY: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    row = tl.arange(0, ROWS)
    dim = tl.arange(0, D)
    head = kv_head * GROUPS + row // QUERIES
    query = row % QUERIES
    row_valid = row < GROUPS * QUERIES
    position = tl.load(POS + query)
    q = tl.load(
        Q + batch * Q_B + head[:, None] * Q_H + query[:, None] * Q_T + dim[None, :],
        mask=row_valid[:, None], other=0,
    )
    kv_base = (batch * KV_HEADS + kv_head) * CAPACITY * D
    acc = tl.zeros((ROWS, D), tl.float32)
    maximum = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    start = split * SPLIT_SIZE
    last_position = tl.load(POS + QUERIES - 1)
    if start <= last_position:
        for offset in range(0, SPLIT_SIZE, BLOCK):
            col = start + offset + tl.arange(0, BLOCK)
            initialized = (col < CAPACITY) & (col <= last_position)
            key = tl.load(K + kv_base + dim[:, None] + col[None, :] * D,
                          mask=initialized[None, :], other=0)
            value = tl.load(V + kv_base + col[:, None] * D + dim[None, :],
                            mask=initialized[:, None], other=0)
            visible = initialized[None, :] & (col[None, :] <= position[:, None])
            scores = tl.dot(q, key) * (1.4426950408889634 * D ** -0.5)
            scores = tl.where(visible, scores, float("-inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            correction = tl.where(maximum == float("-inf"), 0., tl.exp2(maximum - new_maximum))
            probabilities = tl.where(visible, tl.exp2(scores - new_maximum[:, None]), 0.)
            acc = acc * correction[:, None]
            acc = tl.dot(probabilities.to(q.dtype), value, acc)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            maximum = new_maximum
    normalized = acc / tl.where(denominator > 0, denominator, 1.)[:, None]
    logsum = tl.where(denominator > 0, maximum + tl.log2(denominator), float("-inf"))
    index = ((batch * KV_HEADS * GROUPS + head) * QUERIES + query) * SPLITS + split
    tl.store(PART + index[:, None] * D + dim[None, :], normalized, mask=row_valid[:, None])
    tl.store(LSE + index, logsum, mask=row_valid)


@triton.jit
def _prologue_attention(
    QKV, WQ, WK, COS, SIN, K, V, POS, PART, LSE,
    KV_HEADS: tl.constexpr, GROUPS: tl.constexpr, WIDTH: tl.constexpr,
    K_OFFSET: tl.constexpr, V_OFFSET: tl.constexpr, CAPACITY: tl.constexpr,
    D: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr,
    SPLITS: tl.constexpr, SPLIT_SIZE: tl.constexpr, ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    dim = tl.arange(0, D)
    other = (dim + D // 2) % D
    sign = dim < D // 2
    cos = tl.load(COS + dim).to(tl.float32)
    sin = tl.load(SIN + dim).to(tl.float32)
    position = tl.load(POS)
    row = tl.arange(0, ROWS)
    row_valid = row < GROUPS
    # Q heads for this KV head: norm + rope with _norm_rope_row's boundaries.
    base = batch * WIDTH + (kv_head * GROUPS + row) * D
    x = tl.load(QKV + base[:, None] + dim[None, :], row_valid[:, None], 0).to(tl.float32)
    rotated = tl.load(QKV + base[:, None] + other[None, :], row_valid[:, None], 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + Q_EPS)
    gain = tl.load(WQ + dim).to(tl.float32)
    other_gain = tl.load(WQ + other).to(tl.float32)
    normed = ((x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
              * gain[None, :]).to(tl.bfloat16).to(tl.float32)
    rotated = ((rotated * inv[:, None]).to(tl.bfloat16).to(tl.float32)
               * other_gain[None, :]).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(sign[None, :], -rotated, rotated)
    left = (normed * cos[None, :]).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sin[None, :]).to(tl.bfloat16).to(tl.float32)
    q = (left + right).to(tl.bfloat16)
    # Newest K for this KV head, same boundaries; V passes through.
    k_base = batch * WIDTH + K_OFFSET + kv_head * D
    kx = tl.load(QKV + k_base + dim).to(tl.float32)
    k_rotated = tl.load(QKV + k_base + other).to(tl.float32)
    k_inv = tl.rsqrt(tl.sum(kx * kx, 0) / D + K_EPS)
    k_gain = tl.load(WK + dim).to(tl.float32)
    k_other_gain = tl.load(WK + other).to(tl.float32)
    k_normed = ((kx * k_inv).to(tl.bfloat16).to(tl.float32)
                * k_gain).to(tl.bfloat16).to(tl.float32)
    k_rotated = ((k_rotated * k_inv).to(tl.bfloat16).to(tl.float32)
                 * k_other_gain).to(tl.bfloat16).to(tl.float32)
    k_rotated = tl.where(sign, -k_rotated, k_rotated)
    k_left = (k_normed * cos).to(tl.bfloat16).to(tl.float32)
    k_right = (k_rotated * sin).to(tl.bfloat16).to(tl.float32)
    k_new = (k_left + k_right).to(tl.bfloat16)
    v_new = tl.load(QKV + batch * WIDTH + V_OFFSET + kv_head * D + dim)
    kv_base = (batch * KV_HEADS + kv_head) * CAPACITY * D
    # Online softmax over the cache, strictly before the newest position.
    acc = tl.zeros((ROWS, D), tl.float32)
    maximum = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    scale = 1.4426950408889634 * D ** -0.5
    start = split * SPLIT_SIZE
    if start < position:
        for offset in range(0, SPLIT_SIZE, BLOCK):
            col = start + offset + tl.arange(0, BLOCK)
            visible = (col < CAPACITY) & (col < position)
            key = tl.load(K + kv_base + dim[:, None] + col[None, :] * D,
                          mask=visible[None, :], other=0)
            value = tl.load(V + kv_base + col[:, None] * D + dim[None, :],
                            mask=visible[:, None], other=0)
            scores = tl.dot(q, key) * scale
            scores = tl.where(visible[None, :], scores, float("-inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            correction = tl.where(maximum == float("-inf"), 0., tl.exp2(maximum - new_maximum))
            probabilities = tl.where(visible[None, :], tl.exp2(scores - new_maximum[:, None]), 0.)
            acc = acc * correction[:, None]
            acc = tl.dot(probabilities.to(q.dtype), value, acc)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            maximum = new_maximum
    if split == SPLITS - 1:
        # The last split folds in the newest token and persists its KV.
        score = tl.sum(q.to(tl.float32) * k_new.to(tl.float32)[None, :], 1) * scale
        new_maximum = tl.maximum(maximum, score)
        correction = tl.where(maximum == float("-inf"), 0., tl.exp2(maximum - new_maximum))
        probability = tl.exp2(score - new_maximum)
        acc = acc * correction[:, None] + probability[:, None] * v_new.to(tl.float32)[None, :]
        denominator = denominator * correction + probability
        maximum = new_maximum
        cache_row = kv_base + position * D
        tl.store(K + cache_row + dim, k_new)
        tl.store(V + cache_row + dim, v_new)
    normalized = acc / tl.where(denominator > 0, denominator, 1.)[:, None]
    logsum = tl.where(denominator > 0, maximum + tl.log2(denominator), float("-inf"))
    head = kv_head * GROUPS + row
    index = (batch * KV_HEADS * GROUPS + head) * SPLITS + split
    tl.store(PART + index[:, None] * D + dim[None, :], normalized, mask=row_valid[:, None])
    tl.store(LSE + index, logsum, mask=row_valid)


def prologue_attention(qkv, q_norm, k_norm, cos, sin, keys, values, positions,
                       kv_heads, groups, k_offset, v_offset):
    """Single-token decode attention with norm/rope/cache-write fused in.

    qkv is the packed (batch, 1, width) decode projection output. Writes the
    newest K/V into the fixed cache slot and returns [batch, heads, 1, D].
    """
    batch = qkv.shape[0]
    width = qkv.shape[-1]
    heads = kv_heads * groups
    dim = keys.shape[-1]
    capacity = keys.shape[2]
    target_splits = max(1, 128 // (batch * kv_heads))
    split_size = max(64, min(512, triton.next_power_of_2(triton.cdiv(capacity, target_splits))))
    splits = triton.cdiv(capacity, split_size)
    partial = torch.empty((batch, heads, 1, splits, dim), dtype=torch.float32, device=qkv.device)
    logsum = torch.empty((batch, heads, 1, splits), dtype=torch.float32, device=qkv.device)
    out = torch.empty((batch, heads, 1, dim), dtype=qkv.dtype, device=qkv.device)
    _prologue_attention[(batch, kv_heads, splits)](
        qkv, q_norm.weight, k_norm.weight, cos, sin, keys, values, positions,
        partial, logsum, kv_heads, groups, width, k_offset, v_offset,
        capacity, dim, q_norm.variance_epsilon, k_norm.variance_epsilon,
        splits, split_size, max(16, triton.next_power_of_2(groups)), 64,
        num_warps=4, num_stages=2,
    )
    _merge_attention[(batch * heads,)](
        partial, logsum, out, splits, dim, triton.next_power_of_2(splits), num_warps=4,
    )
    return out


@triton.jit
def _merge_attention(PART, LSE, OUT, SPLITS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, BLOCK)
    dim = tl.arange(0, D)
    logs = tl.load(LSE + row * SPLITS + split, mask=split < SPLITS, other=float("-inf"))
    weights = tl.exp2(logs - tl.max(logs, 0))
    weights = weights / tl.sum(weights, 0)
    partial = tl.load(PART + (row * SPLITS + split[:, None]) * D + dim[None, :],
                      mask=split[:, None] < SPLITS, other=0.)
    out = tl.sum(partial * weights[:, None], 0)
    tl.store(OUT + row * D + dim, out)


def grouped_attention(q, k, v, positions):
    batch, heads, queries, dim = q.shape
    kv_heads, capacity = k.shape[1:3]
    groups = heads // kv_heads
    target_splits = max(1, 128 // (batch * kv_heads))
    split_size = max(64, min(512, triton.next_power_of_2(triton.cdiv(capacity, target_splits))))
    splits = triton.cdiv(capacity, split_size)
    partial = torch.empty((batch, heads, queries, splits, dim), dtype=torch.float32, device=q.device)
    logsum = torch.empty((batch, heads, queries, splits), dtype=torch.float32, device=q.device)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _partial_attention[(batch, kv_heads, splits)](
        q, k, v, positions, partial, logsum,
        q.stride(0), q.stride(1), q.stride(2),
        kv_heads, groups, queries, capacity, dim, splits, split_size,
        max(16, triton.next_power_of_2(groups * queries)), 64,
        num_warps=4, num_stages=2,
    )
    _merge_attention[(batch * heads * queries,)](
        partial, logsum, out, splits, dim, triton.next_power_of_2(splits), num_warps=4,
    )
    return out
