import torch
import triton
import triton.language as tl
from kernels.attention import _merge_attention

@triton.jit
def _direct_attention(
    QKV, WQ, WK, COS, SIN, K, V, POS, PART, LSE,
    KV_HEADS: tl.constexpr, GROUPS: tl.constexpr, WIDTH: tl.constexpr,
    K_OFFSET: tl.constexpr, V_OFFSET: tl.constexpr, CAPACITY: tl.constexpr,
    D: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr,
    SPLITS: tl.constexpr, SPLIT_SIZE: tl.constexpr, ROWS: tl.constexpr,
    BLOCK: tl.constexpr, DIRECT: tl.constexpr,
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
    if DIRECT:
        index = (batch * KV_HEADS * GROUPS + head) * D
        tl.store(PART + index[:, None] + dim[None, :], normalized, mask=row_valid[:, None])
    else:
        index = (batch * KV_HEADS * GROUPS + head) * SPLITS + split
        tl.store(PART + index[:, None] * D + dim[None, :], normalized, mask=row_valid[:, None])
        tl.store(LSE + index, logsum, mask=row_valid)

def attention(qkv, q_norm, k_norm, cos, sin, keys, values, positions,
              kv_heads, groups, k_offset, v_offset, max_chunk=512):
    batch=qkv.shape[0];width=qkv.shape[-1];heads=kv_heads*groups
    dim=keys.shape[-1];capacity=keys.shape[2]
    target_splits=max(1,128//(batch*kv_heads))
    split_size=max(64,min(max_chunk,triton.next_power_of_2(triton.cdiv(capacity,target_splits))))
    splits=triton.cdiv(capacity,split_size)
    out=torch.empty((batch,heads,1,dim),dtype=qkv.dtype,device=qkv.device)
    if splits==1:
        partial=out
        logsum=out
    else:
        partial=torch.empty((batch,heads,1,splits,dim),dtype=torch.float32,device=qkv.device)
        logsum=torch.empty((batch,heads,1,splits),dtype=torch.float32,device=qkv.device)
    _direct_attention[(batch,kv_heads,splits)](
        qkv,q_norm.weight,k_norm.weight,cos,sin,keys,values,positions,partial,logsum,
        kv_heads,groups,width,k_offset,v_offset,capacity,dim,
        q_norm.variance_epsilon,k_norm.variance_epsilon,
        splits,split_size,max(16,triton.next_power_of_2(groups)),64,splits==1,
        num_warps=4,num_stages=2)
    if splits>1:
        _merge_attention[(batch*heads,)](partial,logsum,out,splits,dim,
            triton.next_power_of_2(splits),num_warps=4)
    return out
