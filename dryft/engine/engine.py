"""BF16 Qwen3 with fused RMSNorm, preallocated KV storage and CUDA graphs."""

import torch
from transformers import AutoModelForCausalLM

from decode import DecodeState
from attention import GroupedAttention
from kernels.fused import swiglu
from kernels.rmsnorm import rms_norm


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
        self.gate_proj = reference.gate_proj
        self.up_proj = reference.up_proj
        self.down_proj = reference.down_proj

    def forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


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
                self.state = DecodeState(self.model, *shape)
            state = self.state
            prompt = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
            current = state.prefill(self.model, prompt)
            yield current[:, 0].tolist()
            for _ in range(max_new_tokens - 1):
                state.graph.replay()
                yield state.token[:, 0].tolist()
