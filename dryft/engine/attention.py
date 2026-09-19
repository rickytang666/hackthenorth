"""Flash prefill and grouped decode without expanding the KV heads."""

import json

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from kernels.attention import grouped_attention
from kernels.fused import norm_rope, norm_rope_cache
from projections import Projection

_prefill_backends = {}


def _prefill_attention(q, k, v):
    """Causal grouped SDPA with a per-shape measured backend.

    Flash and cuDNN both accumulate FP32 over BF16; which is faster at a
    given prefill shape varies by GPU, so the first call per shape times
    both on the real tensors (during the untimed warmup, before any graph
    capture) and caches the winner. Flash remains the fallback.
    """
    key = (q.device, q.dtype, tuple(q.shape), tuple(k.shape),
           q.stride(), k.stride(), v.stride())
    backend = _prefill_backends.get(key)
    if backend is None:
        if torch.cuda.is_current_stream_capturing():
            backend = SDPBackend.FLASH_ATTENTION
        else:
            timings = []
            for candidate in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
                try:
                    with sdpa_kernel(candidate):
                        for _ in range(2):
                            torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)
                        start = torch.cuda.Event(True)
                        end = torch.cuda.Event(True)
                        start.record()
                        for _ in range(3):
                            torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)
                        end.record()
                        end.synchronize()
                    timings.append((start.elapsed_time(end) / 3, candidate))
                except Exception:
                    continue
            timings.sort(key=lambda pair: pair[0])
            backend = timings[0][1] if timings else SDPBackend.FLASH_ATTENTION
            _prefill_backends[key] = backend
            print(json.dumps(dict(kind="prefill_attention_calibration",
                                  q_shape=list(q.shape), backend=str(backend),
                                  milliseconds=[[round(t, 4), str(b)] for t, b in timings])),
                  flush=True)
    with sdpa_kernel(backend):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)


class GroupedAttention(torch.nn.Module):
    # Decode and verification use absolute positions inside grouped_attention.
    uses_position_causality = True

    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        projections = (reference.q_proj, reference.k_proj, reference.v_proj)
        self.widths = tuple(proj.weight.shape[0] for proj in projections)
        self.qkv_weight = torch.nn.Parameter(
            torch.cat([proj.weight.detach() for proj in projections]), requires_grad=False,
        )
        # The reference fallback and packed paths share weight storage.
        for proj, weight in zip(projections, self.qkv_weight.split(self.widths)):
            proj.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.qkv = Projection(self.qkv_weight, "qkv")
        reference.o_proj = Projection(reference.o_proj.weight, "o")

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        ref = self.reference
        prefill = past_key_value.prefill
        if prefill and attention_mask is not None:
            return ref(
                hidden_states, position_embeddings, attention_mask,
                past_key_value=past_key_value, cache_position=cache_position, **kwargs,
            )
        shape = (*hidden_states.shape[:-1], -1, ref.head_dim)
        qkv = (torch.nn.functional.linear(hidden_states, self.qkv_weight)
               if prefill else self.qkv(hidden_states))
        q, k, v = qkv.split(self.widths, dim=-1)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        if prefill:
            q, k = norm_rope(q, k, ref.q_norm, ref.k_norm, *position_embeddings)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            k, v = past_key_value.update(k, v, ref.layer_idx, {"cache_position": cache_position})
            result = _prefill_attention(q, k, v)
        else:
            q, k, v = norm_rope_cache(
                q, k, v, ref.q_norm, ref.k_norm, *position_embeddings,
                past_key_value.keys[ref.layer_idx], past_key_value.values[ref.layer_idx], cache_position,
            )
            result = grouped_attention(q, k, v, cache_position)
        result = result.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return ref.o_proj(result), None
