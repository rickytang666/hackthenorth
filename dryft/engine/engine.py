"""BF16 Qwen3 with fused RMSNorm, preallocated KV storage and CUDA graphs."""

import torch
from transformers import AutoModelForCausalLM

from decode import DecodeState
from attention import GroupedAttention
from kernels.fused import swiglu
from kernels.gateup import gate_up_swiglu
from kernels.rmsnorm import rms_norm
from speculate import DRAFT_TOKENS, PromptLookup
from fused_layer import install as install_fused_projections
from projections import Projection
from prefill import prefill


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


class FusedMLP(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.gate_up_weight = torch.nn.Parameter(
            torch.cat((reference.gate_proj.weight.detach(), reference.up_proj.weight.detach())),
            requires_grad=False,
        )
        self.gate_up = Projection(self.gate_up_weight, "gate_up")
        self.down_proj = Projection(reference.down_proj.weight, "down")
        self.register_buffer("interleaved_weight", None, persistent=False)

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        if 2 <= rows <= 32:
            if self.interleaved_weight is None:
                gate, up = self.gate_up_weight.detach().chunk(2, 0)
                self.interleaved_weight = torch.stack((gate.T, up.T), -1).flatten(1).contiguous()
            return self.down_proj(gate_up_swiglu(x, self.interleaved_weight))
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down_proj(swiglu(gate, up))


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
            layer.self_attn = GroupedAttention(layer.self_attn)
            layer.mlp = FusedMLP(layer.mlp)
        install_fused_projections(self.model)
        self.state = None

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        with torch.inference_mode():
            shape = (len(input_ids), len(input_ids[0]), max_new_tokens)
            if self.state is None or self.state.shape != shape:
                self.state = None
                self.state = DecodeState(self.model, *shape, speculative=True)
            state = self.state
            prompt = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
            current = prefill(self.model, state, prompt)
            tokens = current[:, 0].tolist()
            yield tokens
            lookup = PromptLookup(input_ids[0]) if state.verify_graph is not None else None
            if lookup is not None:
                lookup.append(tokens[0])
            emitted = 1
            while emitted < max_new_tokens:
                proposal = None
                if lookup is not None and max_new_tokens - emitted > DRAFT_TOKENS:
                    proposal = lookup.propose()
                if proposal is not None:
                    verified = state.verify(tokens[0], proposal, shape[1] + emitted - 1)
                    for token in verified:
                        tokens = [token]
                        lookup.append(token)
                        emitted += 1
                        yield tokens
                else:
                    state.graph.replay()
                    tokens = state.token[:, 0].tolist()
                    if lookup is not None:
                        lookup.append(tokens[0])
                    emitted += 1
                    yield tokens
