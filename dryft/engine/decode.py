"""Fixed-address KV storage and a captured single-token decode step."""

import torch

from speculate import draft_width, verified_tokens


def stream_decode(state, first_tokens, output_length):
    """Overlap the next graph with delivery of an already-materialized token.

    Only the Python list crosses yield; it cannot alias the graph's token buffer.
    The final yield has no outstanding replay, and output_length=1 launches none.
    This path deliberately excludes speculative verification.
    """
    tokens = first_tokens
    for step in range(output_length):
        has_next = step + 1 < output_length
        if has_next:
            state.graph.replay()
        yield tokens
        if has_next:
            tokens = state.token[:, 0].tolist()


class KVCache:
    def __init__(self, model, batch_size, capacity):
        config = model.config
        shape = (batch_size, config.num_key_value_heads, capacity, config.head_dim)
        options = {"device": model.device, "dtype": model.dtype}
        # Masked slots must still be finite: NaNs can survive attention masking.
        self.keys = [torch.zeros(shape, **options) for _ in model.model.layers]
        self.values = [torch.zeros(shape, **options) for _ in model.model.layers]
        self.prefill = True

    def update(self, keys, values, layer_idx, cache_kwargs):
        key_cache, value_cache = self.keys[layer_idx], self.values[layer_idx]
        if self.prefill:
            length = keys.shape[2]
            key_cache[:, :, :length].copy_(keys)
            value_cache[:, :, :length].copy_(values)
            # Prefill attends only to the prompt, preserving causal Flash SDPA.
            return keys, values
        position = cache_kwargs["cache_position"]
        key_cache.index_copy_(2, position, keys)
        value_cache.index_copy_(2, position, values)
        return key_cache, value_cache


class RotaryTable:
    """Prompt-independent default RoPE, computed with the reference module.

    Dynamic/scaled rotary variants retain the native per-call calculation.
    Table rows remain on device; index_select supports graph replay positions.
    """
    def __init__(self, model, capacity, enabled=True):
        rotary = model.model.rotary_emb
        self.tables = None
        if enabled and getattr(rotary, "rope_type", None) == "default":
            positions = torch.arange(capacity, device=model.device).unsqueeze(0)
            with torch.inference_mode():
                exemplar = model.model.embed_tokens.weight[:1].unsqueeze(0)
                self.tables = rotary(exemplar, positions)

    def select(self, positions):
        if self.tables is None:
            return None
        return tuple(table.index_select(1, positions) for table in self.tables)


def uses_position_causality(model):
    return all(
        getattr(layer, "uses_position_causality", False)
        or getattr(getattr(layer, "self_attn", None), "uses_position_causality", False)
        for layer in model.model.layers
    )


def forward(model, token_ids, cache, positions, attention_mask=None, last_only=True,
            position_embeddings=None):
    base = model.model
    x = base.embed_tokens(token_ids)
    position_ids = positions.unsqueeze(0)
    embeddings = (base.rotary_emb(x, position_ids)
                  if position_embeddings is None else position_embeddings)
    for layer in base.layers:
        x = layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=embeddings,
        )[0]
    x = base.norm(x[:, -1:, :] if last_only else x)
    if x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == 2560:
        from kernels.greedy import greedy_token
        return greedy_token(x, model.lm_head.weight)
    return model.lm_head(x).argmax(dim=-1)


class DecodeState:
    def __init__(self, model, batch_size, prompt_length, output_length, speculative=False,
                 cache_rotary=True, omit_unused_masks=True):
        self.shape = (batch_size, prompt_length, output_length)
        self.draft_tokens = (draft_width(output_length)
                             if speculative and batch_size == 1 else 0)
        # Tail verification may write beyond the last requested output.
        # Causal attention hides these scratch slots from earlier queries.
        capacity = prompt_length + output_length - 1 + self.draft_tokens
        device = model.device
        self.cache = KVCache(model, batch_size, capacity)
        self.prompt_positions = torch.arange(prompt_length, device=device)
        self.token = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
        self.position = torch.full((1,), prompt_length, dtype=torch.int64, device=device)
        self.slots = torch.arange(capacity, device=device)
        self.rotary = RotaryTable(model, capacity, enabled=cache_rotary)
        self.position_causality = omit_unused_masks and uses_position_causality(model)
        self.graph = None
        self.verify_graph = None
        if output_length > 1:
            self.capture(model)
        if self.draft_tokens >= 1:
            self.capture_verifier(model)

    def step(self, model):
        mask = None
        if not self.position_causality:
            visible = self.slots <= self.position
            mask = torch.zeros_like(self.slots, dtype=model.dtype)
            mask.masked_fill_(~visible, torch.finfo(model.dtype).min)
            mask = mask.view(1, 1, 1, -1)
        next_token = forward(
            model, self.token, self.cache, self.position, mask,
            position_embeddings=self.rotary.select(self.position),
        )
        self.token.copy_(next_token)
        self.position.add_(1)

    def capture(self, model):
        self.cache.prefill = False
        prompt_length = self.shape[1]
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream):
            # Initialize CUDA libraries and Triton specializations off capture.
            for _ in range(2):
                self.position.fill_(prompt_length)
                self.step(model)

            self.position.fill_(prompt_length)
        torch.cuda.current_stream(model.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.step(model)

    def prefill(self, model, input_ids):
        self.cache.prefill = True
        embeddings = (None if self.rotary.tables is None else
                      tuple(table[:, :self.shape[1]] for table in self.rotary.tables))
        token = forward(model, input_ids, self.cache, self.prompt_positions,
                        position_embeddings=embeddings)
        self.cache.prefill = False
        self.token.copy_(token)
        self.position.fill_(self.shape[1])
        return self.token

    def verify_step(self, model):
        positions = self.position + self.verify_offsets
        mask = None
        if not self.position_causality:
            visible = self.slots[None, :] <= positions[:, None]
            mask = torch.zeros(visible.shape, dtype=model.dtype, device=model.device)
            mask.masked_fill_(~visible, torch.finfo(model.dtype).min)
            mask = mask[None, None, :, :]
        return forward(
            model, self.verify_input, self.cache, positions,
            mask, last_only=False, position_embeddings=self.rotary.select(positions),
        )

    def capture_verifier(self, model):
        length = self.draft_tokens + 1
        self.verify_input = torch.zeros((1, length), dtype=torch.int64, device=model.device)
        self.verify_offsets = torch.arange(length, device=model.device)
        self.position.fill_(self.shape[1])
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.verify_step(model)

        torch.cuda.current_stream(model.device).wait_stream(stream)
        self.verify_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.verify_graph, stream=stream):
            self.verify_output = self.verify_step(model)

    def verify(self, current_token, proposal, position):
        inputs = torch.tensor([[current_token, *proposal]], dtype=torch.int64, device=self.token.device)
        self.verify_input.copy_(inputs)
        self.verify_graph.replay()
        emitted = verified_tokens(proposal, self.verify_output[0].tolist())
        self.token.copy_(self.verify_output[:, len(emitted) - 1:len(emitted)])
        # Discard the speculative tail logically. Future attention masks it,
        # and subsequent decode/verification overwrites those slots.
        self.position.fill_(position + len(emitted))
        return emitted
