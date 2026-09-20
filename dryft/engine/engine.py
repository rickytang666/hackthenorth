"""BF16 Qwen3 with fused RMSNorm, preallocated KV storage and CUDA graphs."""

import torch
from transformers import AutoModelForCausalLM

from decode import DecodeState, stream_decode
from attention import GroupedAttention
from kernels.fused import swiglu
from kernels.gateup import LEGACY_CONFIG, gate_up_swiglu
from kernels.rmsnorm import rms_norm
from speculate import BackoffPromptLookup as PromptLookup
from fused_layer import install as install_fused_projections
from kernels.tile import tile_projection
from kernels.split_down import split_down_residual, split_down_residual_norm
from projections import Projection, probe_point, resolve
from prefill import prefill

# (row tile, column tile, K tile, warps, stages) candidates for the fused
# gate/up SwiGLU GEMM; the frozen tile leads and calibration arbitrates
# in-graph on the run GPU.
GATEUP_CANDIDATES = (
    LEGACY_CONFIG,
    (16, 128, 64, 4, 4), (16, 256, 64, 4, 3), (16, 256, 64, 8, 4),
    (16, 128, 128, 4, 4), (16, 64, 64, 4, 3), (16, 128, 64, 8, 3),
    (16, 256, 128, 8, 3), (32, 128, 64, 4, 3),
)


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

    def _candidates(self, rows):
        return [config for config in GATEUP_CANDIDATES
                if self.gate_up_weight.shape[1] % config[2] == 0]

    def _run(self, choice, x, rows):
        return gate_up_swiglu(x, self.interleaved_weight, choice)

    def _reference(self, x):
        # The kernel's documented rounding boundaries, composed natively.
        gate, up = torch.nn.functional.linear(x, self.gate_up_weight).chunk(2, dim=-1)
        value = gate.float()
        activated = (value * torch.sigmoid(value)).to(torch.bfloat16).float()
        return (activated * up.float()).to(torch.bfloat16)

    def _intermediate(self, x, rows):
        if self.interleaved_weight is None:
            gate, up = self.gate_up_weight.detach().chunk(2, 0)
            self.interleaved_weight = torch.stack((gate.T, up.T), -1).flatten(1).contiguous()
        probe_point("gate_up_fused", self, x, rows)
        config = resolve("gate_up_fused", rows) or LEGACY_CONFIG
        return gate_up_swiglu(x, self.interleaved_weight, config)

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        if 2 <= rows <= 32:
            return self.down_proj(self._intermediate(x, rows))
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down_proj(swiglu(gate, up))

    def forward_residual(self, x, residual):
        """MLP with the residual folded into the down projection's store."""
        rows = x.numel() // x.shape[-1]
        if not 2 <= rows <= 32:
            return residual + self.forward(x)
        intermediate = self._intermediate(x, rows)
        return self._down_residual(intermediate, residual, rows)

    def _down_residual(self, intermediate, residual, rows):
        probe_point("down_residual", _DownResidual(self.down_proj, residual),
                    intermediate, rows)
        choice = resolve("down_residual", rows) or ("legacy",)
        if choice[0] in ("restile", "splitk"):
            return _DownResidual(self.down_proj, residual)._run(choice, intermediate, rows)
        return residual + self.down_proj(intermediate)

    def forward_residual_next_norm(self, x, residual, next_norm):
        rows = x.numel() // x.shape[-1]
        intermediate = self._intermediate(x, rows)
        adapter = _DownNextNorm(self, residual, next_norm)
        probe_point("down_next_norm", adapter, intermediate, rows)
        choice = resolve("down_next_norm", rows) or ("legacy",)
        if choice[0] == "legacy":
            return self._down_residual(intermediate, residual, rows), None
        return adapter.compute(choice, intermediate, rows)


class _DownNextNorm:
    """Compare the whole decode step with cross-layer down/norm fusion."""

    def __init__(self, mlp, residual, norm):
        self.mlp = mlp
        self.residual = residual
        self.norm = norm

    def _candidates(self, rows):
        return [("legacy",), ("fused", 16)]

    def compute(self, choice, x, rows):
        norm = self.norm
        if choice[0] == "legacy":
            out = self.mlp._down_residual(x, self.residual, rows)
            return out, rms_norm(out, norm.weight, norm.variance_epsilon)
        return split_down_residual_norm(x, self.mlp.down_proj._column(),
                                       self.residual, norm.weight,
                                       norm.variance_epsilon, choice[1])

    def _run(self, choice, x, rows):
        # Validate both outputs; timing uses the ordinary decode graph.
        return torch.stack(self.compute(choice, x, rows))

    def _reference(self, x):
        out = self.residual + torch.nn.functional.linear(x, self.mlp.down_proj.weight)
        value = out.float()
        inverse = torch.rsqrt(value.square().mean(-1, keepdim=True)
                              + self.norm.variance_epsilon)
        normalized = (value * inverse).to(out.dtype) * self.norm.weight
        return torch.stack((out, normalized))


class _DownResidual:
    """Calibration adapter: the residual add fused into the down tile's store."""

    TILES = ((32, 128, 2, 5), (64, 128, 4, 5))

    def __init__(self, projection, residual):
        self.projection = projection
        self.residual = residual

    def _candidates(self, rows):
        options = [("legacy",)] + [("restile", column, tile)
                                   for tile in self.TILES for column in (True, False)]
        if rows in (4, 16):
            options.append(("splitk",))
        return options

    def _run(self, choice, x, rows):
        if choice[0] == "legacy":
            return self.residual + self.projection(x)
        if choice[0] == "splitk":
            flat = split_down_residual(
                x.reshape(rows, -1), self.projection._column(),
                self.residual.reshape(rows, -1),
            )
            return flat.reshape(self.residual.shape)
        _, column, tile = choice
        n, k = self.projection.weight.shape
        weight = self.projection._column() if column else self.projection.weight
        flat = tile_projection(x.reshape(rows, k), weight, rows, tile,
                               residual=self.residual.reshape(rows, n))
        return flat.reshape(self.residual.shape)

    def _reference(self, x):
        return self.residual + torch.nn.functional.linear(x, self.projection.weight)


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
            if state.verify_graph is None:
                yield from stream_decode(state, tokens, max_new_tokens)
                return
            yield tokens
            lookup = (PromptLookup(input_ids[0], draft_length=state.draft_tokens)
                      if state.verify_graph is not None else None)
            if lookup is not None:
                lookup.append(tokens[0])
            emitted = 1
            while emitted < max_new_tokens:
                proposal = None
                if lookup is not None:
                    proposal = lookup.propose()
                if proposal is not None:
                    verified = state.verify(tokens[0], proposal, shape[1] + emitted - 1)
                    # The cache carries scratch slots for the draft, so a
                    # verify may run at any position; only the tokens this
                    # generation still owes are emitted.
                    for token in verified[:max_new_tokens - emitted]:
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
