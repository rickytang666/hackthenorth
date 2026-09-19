"""Fuse small-batch decoder projections while retaining native prefill."""
import torch

from kernels.attention import grouped_attention
from kernels.fused import norm_rope
from kernels.projection import norm_projection, norm_tensor_projection


class FusedProjectionLayer(torch.nn.Module):
    def __init__(self, layer, selections):
        super().__init__()
        self.original = layer
        self.selections = selections

    def project(self, x, norm, weight, config, swiglu=False):
        fn = norm_projection if config["mode"] == "vector" else norm_tensor_projection
        return fn(x, norm.weight, weight, norm.variance_epsilon,
                  swiglu=swiglu, block_n=config["block_n"], warps=config["warps"])

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        layer = self.original
        rows = hidden_states.numel() // hidden_states.shape[-1]
        selected = self.selections.get(str(rows), {})
        if past_key_value.prefill or not selected:
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
            q,k = norm_rope(q.view(shape), k.view(shape), ref.q_norm, ref.k_norm,
                            *position_embeddings)
            q,k,v = q.transpose(1,2), k.transpose(1,2), v.view(shape).transpose(1,2)
            k,v = past_key_value.update(k,v,ref.layer_idx,{"cache_position":cache_position})
            a = grouped_attention(q,k,v,cache_position)
            a = a.transpose(1,2).reshape(*hidden_states.shape[:-1], -1)
            a = ref.o_proj(a)
        else:
            a = attention(layer.input_layernorm(hidden_states), position_embeddings,
                          attention_mask, past_key_value=past_key_value,
                          cache_position=cache_position)[0]
        residual = hidden_states + a
        if "mlp" in selected:
            intermediate = self.project(residual, layer.post_attention_layernorm,
                                        layer.mlp.gate_up_weight, selected["mlp"], swiglu=True)
            result = layer.mlp.down_proj(intermediate)
        else:
            result = layer.mlp(layer.post_attention_layernorm(residual))
        result = residual + result
        return (result,None) if output_attentions else (result,)


def install(model):
    selections = {
        str(rows): {
            kind: {"mode": "vector" if rows == 1 else "tensor",
                   "block_n": 4 if rows == 1 else 64, "warps": 8}
            for kind in ("qkv", "mlp")
        }
        for rows in range(1, 17)
    }
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = FusedProjectionLayer(layer, selections)
