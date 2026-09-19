"""Flash prefill and grouped decode without expanding the KV heads."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from kernels.attention import grouped_attention
from kernels.fused import norm_rope
from projections import Projection


class GroupedAttention(torch.nn.Module):
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
        q, k = q.view(shape), k.view(shape)
        v = v.view(shape).transpose(1, 2)
        q, k = norm_rope(q, k, ref.q_norm, ref.k_norm, *position_embeddings)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        k, v = past_key_value.update(k, v, ref.layer_idx, {"cache_position": cache_position})
        if prefill:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True,
                )
        else:
            result = grouped_attention(q, k, v, cache_position)
        result = result.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return ref.o_proj(result), None
