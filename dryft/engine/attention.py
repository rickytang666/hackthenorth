"""Flash prefill and grouped decode without expanding the KV heads."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from kernels.attention import grouped_attention, prologue_attention
from kernels.sglang_attention import grouped_attention as sglang_attention
from kernels.fused import norm_rope, norm_rope_cache
from kernels.rmsnorm import rms_norm
from kernels.tile import tile_projection
from projections import Projection, probe_point, resolve


class _NormQkv:
    """Calibration adapter: input RMSNorm fused into the QKV projection tile."""

    TILES = ((64, 128, 4, 5), (32, 128, 2, 5))

    def __init__(self, attention, norm):
        self.attention = attention
        self.norm = norm

    def _candidates(self, rows):
        return [("legacy",)] + [("normtile", column, tile)
                                for tile in self.TILES for column in (True, False)]

    def _run(self, choice, x, rows):
        if choice[0] == "legacy":
            return self.attention.qkv(
                rms_norm(x, self.norm.weight, self.norm.variance_epsilon))
        _, column, tile = choice
        projection = self.attention.qkv
        weight = projection._column() if column else projection.weight
        k = weight.shape[1]
        flat = tile_projection(x.reshape(rows, k), weight, rows, tile,
                               gain=self.norm.weight,
                               eps=self.norm.variance_epsilon)
        return flat.reshape(*x.shape[:-1], weight.shape[0])

    def _reference(self, x):
        value = x.float()
        inverse = torch.rsqrt(value.pow(2).mean(-1, keepdim=True)
                              + self.norm.variance_epsilon)
        normalized = (value * inverse).to(torch.bfloat16)
        return torch.nn.functional.linear(normalized * self.norm.weight,
                                          self.attention.qkv_weight)

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
                past_key_value=None, cache_position=None, norm=None,
                normalized_input=None, **kwargs):
        ref = self.reference
        prefill = past_key_value.prefill
        if prefill and attention_mask is not None:
            return ref(
                hidden_states, position_embeddings, attention_mask,
                past_key_value=past_key_value, cache_position=cache_position, **kwargs,
            )
        shape = (*hidden_states.shape[:-1], -1, ref.head_dim)
        if normalized_input is not None and not prefill:
            qkv = self.qkv(normalized_input)
        elif norm is not None and not prefill:
            rows = hidden_states.numel() // hidden_states.shape[-1]
            adapter = _NormQkv(self, norm)
            probe_point("norm_qkv", adapter, hidden_states, rows)
            qkv = adapter._run(resolve("norm_qkv", rows) or ("legacy",),
                               hidden_states, rows)
        else:
            if norm is not None:
                hidden_states = rms_norm(hidden_states, norm.weight,
                                         norm.variance_epsilon)
            qkv = (torch.nn.functional.linear(hidden_states, self.qkv_weight)
                   if prefill else self.qkv(hidden_states))
        q, k, v = qkv.split(self.widths, dim=-1)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        if prefill:
            q, k = norm_rope(q, k, ref.q_norm, ref.k_norm, *position_embeddings)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            k, v = past_key_value.update(k, v, ref.layer_idx, {"cache_position": cache_position})
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)
        else:
            adapter = _RopeAttention(ref, self.widths, qkv, past_key_value,
                                     position_embeddings, cache_position)
            if hidden_states.shape[1] == 1:
                rows = hidden_states.shape[0]
                probe_point("rope_attention", adapter, qkv, rows)
                choice = resolve("rope_attention", rows) or ("legacy",)
            else:
                choice = ("legacy",)
            result = adapter._run(choice, qkv, hidden_states.shape[0])
        result = result.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return ref.o_proj(result), None


class _RopeAttention:
    """Calibrate fused prologue and upstream SGLang decode attention."""

    def __init__(self, ref, widths, qkv, past_key_value, embeddings, positions):
        self.ref = ref
        self.widths = widths
        self.rows = qkv.shape[0]
        self.cache = past_key_value
        self.embeddings = embeddings
        self.positions = positions

    def _candidates(self, rows):
        options = [("legacy",), ("fused",)]
        if rows >= 4 and self.ref.head_dim == 128:
            options.extend([("sglang", 4), ("sglang", 8)])
        return options

    def _run(self, choice, x, rows):
        ref = self.ref
        keys = self.cache.keys[ref.layer_idx]
        values = self.cache.values[ref.layer_idx]
        positions = self.positions
        validation = getattr(self, "_validation", None)
        if validation is not None and x is validation[0]:
            keys, values, positions = validation[1:]
        if choice[0] == "fused":
            dim = ref.head_dim
            return prologue_attention(
                x, ref.q_norm, ref.k_norm, *self.embeddings, keys, values,
                positions, self.widths[1] // dim, self.widths[0] // self.widths[1],
                self.widths[0], self.widths[0] + self.widths[1])
        shape = (*x.shape[:-1], -1, ref.head_dim)
        q, k, v = x.split(self.widths, dim=-1)
        q, k, v = norm_rope_cache(
            q.view(shape), k.view(shape), v.view(shape), ref.q_norm, ref.k_norm,
            *self.embeddings, keys, values, positions)
        if choice[0] == "sglang":
            return sglang_attention(q, k, v, positions, splits=choice[1], block=64)
        return grouped_attention(q, k, v, positions)

    def _reference(self, x):
        # Timing replays mutate the live cache. Compare every candidate against
        # the same cache contents; ordinary decode adapters never take this copy.
        layer = self.ref.layer_idx
        self._validation = (x, self.cache.keys[layer].clone(),
                            self.cache.values[layer].clone(), self.positions.clone())
        return self._run(("legacy",), x, self.rows)
