"""Fuse small-batch decoder projections while retaining native prefill."""
import torch

from kernels.attention import grouped_attention
from kernels.fused import norm_rope_cache
from kernels.projection import norm_projection
from kernels.rmsnorm import add_rms_norm


class FusedProjectionLayer(torch.nn.Module):
    @property
    def uses_position_causality(self):
        return getattr(self.original.self_attn, "uses_position_causality", False)

    def __init__(self, layer, selections):
        super().__init__()
        self.original = layer
        self.selections = selections
        self.fuse_residual_norm = False  # Opt-in until full-generation GPU A/B passes.

    def project(self, x, norm, weight, config, swiglu=False):
        return norm_projection(x, norm.weight, weight, norm.variance_epsilon,
                  swiglu=swiglu, block_n=config["block_n"], warps=config["warps"])

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        layer = self.original
        rows = hidden_states.numel() // hidden_states.shape[-1]
        selected = self.selections.get(str(rows), {})
        if past_key_value.prefill or (not selected and not self.fuse_residual_norm):
            return layer(hidden_states, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_value=past_key_value,
                         output_attentions=output_attentions, use_cache=use_cache,
                         cache_position=cache_position, position_embeddings=position_embeddings,
                         **kwargs)
        attention = layer.self_attn
        if "qkv" in selected:
            ref = attention.reference
            packed = self.project(hidden_states, layer.input_layernorm,
                                  attention.qkv_weight, selected["qkv"])
            q,k,v = packed.split(attention.widths, -1)
            shape = (*hidden_states.shape[:-1], -1, ref.head_dim)
            q,k,v = norm_rope_cache(
                q.view(shape), k.view(shape), v.view(shape), ref.q_norm, ref.k_norm,
                *position_embeddings, past_key_value.keys[ref.layer_idx],
                past_key_value.values[ref.layer_idx], cache_position,
            )
            a = grouped_attention(q,k,v,cache_position)
            a = a.transpose(1,2).reshape(*hidden_states.shape[:-1], -1)
            a = ref.o_proj(a)
        else:
            a = attention(layer.input_layernorm(hidden_states), position_embeddings,
                          attention_mask, past_key_value=past_key_value,
                          cache_position=cache_position)[0]
        if "mlp" in selected:
            residual = hidden_states + a
            intermediate = self.project(residual, layer.post_attention_layernorm,
                                        layer.mlp.gate_up_weight, selected["mlp"], swiglu=True)
            result = layer.mlp.down_proj(intermediate)
        else:
            if self.fuse_residual_norm:
                norm = layer.post_attention_layernorm
                residual, normalized = add_rms_norm(
                    a, hidden_states, norm.weight, norm.variance_epsilon)
            else:
                residual = hidden_states + a
                normalized = layer.post_attention_layernorm(residual)
            result = layer.mlp(normalized)
        result = residual + result
        return (result,None) if output_attentions else (result,)


def install(model):
    selections = {
        "1": {
            "qkv": {"block_n": 2, "warps": 4},
            "mlp": {"block_n": 1, "warps": 4},
        },
    }
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = FusedProjectionLayer(layer, selections)
