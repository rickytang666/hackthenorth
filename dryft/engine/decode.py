"""Fixed-address KV storage and a captured single-token decode step."""

import torch


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


def forward(model, token_ids, cache, positions, attention_mask=None):
    base = model.model
    x = base.embed_tokens(token_ids)
    position_ids = positions.unsqueeze(0)
    embeddings = base.rotary_emb(x, position_ids)
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
    x = base.norm(x[:, -1:, :])
    return model.lm_head(x)[:, -1, :].argmax(dim=-1, keepdim=True)


class DecodeState:
    def __init__(self, model, batch_size, prompt_length, output_length):
        self.shape = (batch_size, prompt_length, output_length)
        capacity = prompt_length + output_length - 1
        device = model.device
        self.cache = KVCache(model, batch_size, capacity)
        self.prompt_positions = torch.arange(prompt_length, device=device)
        self.token = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
        self.position = torch.full((1,), prompt_length, dtype=torch.int64, device=device)
        self.slots = torch.arange(capacity, device=device)
        self.graph = None
        if output_length > 1:
            self.capture(model)

    def step(self, model):
        visible = self.slots <= self.position
        mask = torch.zeros_like(self.slots, dtype=model.dtype)
        mask.masked_fill_(~visible, torch.finfo(model.dtype).min)
        next_token = forward(
            model, self.token, self.cache, self.position, mask.view(1, 1, 1, -1)
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
        token = forward(model, input_ids, self.cache, self.prompt_positions)
        self.cache.prefill = False
        self.token.copy_(token)
        self.position.fill_(self.shape[1])
        return self.token
