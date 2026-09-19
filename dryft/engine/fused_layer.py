"""Fuse small-batch decoder projections while retaining native prefill."""
import torch

from kernels.attention import grouped_attention
from kernels.chain import chain_a, chain_b
from kernels.fused import norm_rope_cache
from kernels.projection import norm_projection
from kernels.down import down_residual
from kernels.rmsnorm import add_rms_norm
from projections import probe_point, resolve

# Batch-one QKV rows precomputed by the previous layer's chain_b, keyed by
# the consuming layer's id. Written and consumed strictly in layer order.
_STASH = {}


class _ChainA:
    """Calibration adapter: o+residual+norm+gate_up+swiglu as one kernel."""

    def __init__(self, module, hidden_states):
        self.module = module
        self.hidden = hidden_states

    def _candidates(self, rows):
        return [("legacy",), ("chain",)]

    def _run(self, choice, x, rows):
        layer = self.module.original
        ref = layer.self_attn.reference
        norm = layer.post_attention_layernorm
        if choice[0] == "legacy":
            residual = self.hidden + ref.o_proj(x)
            return self.module.project(
                residual, norm, layer.mlp.gate_up_weight,
                self.module.selections["1"]["mlp"], swiglu=True)
        intermediate, _ = chain_a(x, ref.o_proj.weight, self.hidden,
                                  norm.weight, self.module._chain_gu(),
                                  norm.variance_epsilon)
        return intermediate

    def _reference(self, x):
        return self._run(("legacy",), x, 1)


class _ChainB:
    """Calibration adapter: down+residual+next norm+next QKV as one kernel."""

    def __init__(self, module, residual):
        self.module = module
        self.residual = residual

    def _candidates(self, rows):
        return [("legacy",), ("chain",)]

    def _run(self, choice, x, rows):
        layer = self.module.original
        follower = self.module.follower.original
        norm = follower.input_layernorm
        if choice[0] == "legacy":
            hidden = down_residual(x, layer.mlp.down_proj.weight, self.residual)
            return self.module.follower.project(
                hidden, norm, follower.self_attn.qkv_weight,
                self.module.selections["1"]["qkv"])
        _, packed = chain_b(x, layer.mlp.down_proj.weight, self.residual,
                            norm.weight, follower.self_attn.qkv_weight,
                            norm.variance_epsilon)
        return packed

    def _reference(self, x):
        return self._run(("legacy",), x, 1)


class FusedProjectionLayer(torch.nn.Module):
    @property
    def uses_position_causality(self):
        return getattr(self.original.self_attn, "uses_position_causality", False)

    def __init__(self, layer, selections):
        super().__init__()
        self.original = layer
        self.selections = selections
        # Full-generation H100 A/B (2026-09-19, experiments/fp8-draft-research.md):
        # +0.73-0.85% at batches 1/4/8/16, teacher-forced clean.
        self.fuse_residual_norm = True
        self.follower = None
        self.chain_gu = None

    def _chain_gu(self):
        # Row-interleaved [I, 2, K] gate/up copy: chain_a streams each tile
        # as one contiguous block. Built lazily inside the untimed load.
        if self.chain_gu is None:
            weight = self.original.mlp.gate_up_weight.detach()
            half = weight.shape[0] // 2
            self.chain_gu = torch.stack((weight[:half], weight[half:]), 1).contiguous()
        return self.chain_gu

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
            packed = _STASH.pop(id(self), None)
            if packed is None:
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
        else:
            a = attention(hidden_states, position_embeddings,
                          attention_mask, past_key_value=past_key_value,
                          cache_position=cache_position,
                          norm=layer.input_layernorm)[0]
        if "mlp" in selected and "qkv" in selected:
            probe_point("chain_a", _ChainA(self, hidden_states), a, rows)
            if (resolve("chain_a", rows) or ("legacy",))[0] == "chain":
                norm = layer.post_attention_layernorm
                intermediate, residual = chain_a(
                    a, layer.self_attn.reference.o_proj.weight, hidden_states,
                    norm.weight, self._chain_gu(), norm.variance_epsilon)
            else:
                residual = hidden_states + layer.self_attn.reference.o_proj(a)
                intermediate = self.project(residual, layer.post_attention_layernorm,
                                            layer.mlp.gate_up_weight, selected["mlp"], swiglu=True)
            if self.follower is not None:
                probe_point("chain_b", _ChainB(self, residual), intermediate, rows)
            if (self.follower is not None
                    and (resolve("chain_b", rows) or ("legacy",))[0] == "chain"):
                follower = self.follower.original
                norm = follower.input_layernorm
                result, packed_next = chain_b(
                    intermediate, layer.mlp.down_proj.weight, residual,
                    norm.weight, follower.self_attn.qkv_weight,
                    norm.variance_epsilon)
                _STASH[id(self.follower)] = packed_next
            else:
                result = down_residual(intermediate, layer.mlp.down_proj.weight, residual)
        elif "mlp" in selected:
            residual = hidden_states + a
            intermediate = self.project(residual, layer.post_attention_layernorm,
                                        layer.mlp.gate_up_weight, selected["mlp"], swiglu=True)
            result = down_residual(intermediate, layer.mlp.down_proj.weight, residual)
        else:
            if "qkv" in selected:
                a = layer.self_attn.reference.o_proj(a)
            if self.fuse_residual_norm:
                norm = layer.post_attention_layernorm
                residual, normalized = add_rms_norm(
                    a, hidden_states, norm.weight, norm.variance_epsilon)
            else:
                residual = hidden_states + a
                normalized = layer.post_attention_layernorm(residual)
            if hasattr(layer.mlp, "forward_residual"):
                result = layer.mlp.forward_residual(normalized, residual)
            else:
                result = residual + layer.mlp(normalized)
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
    layers = model.model.layers
    for i in range(len(layers) - 1):
        # Plain attribute: registering the next layer as a submodule would
        # nest the whole stack recursively.
        object.__setattr__(layers[i], "follower", layers[i + 1])
