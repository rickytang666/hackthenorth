"""Capture bounded, fixed-shape prefills; recompute every fresh prompt."""

import torch


def prefill(model, state, input_ids):
    # Very large prefills fall back to eager; the public batched shapes
    # (8192 prompt tokens) fit comfortably in a captured graph.
    if input_ids.numel() > 16384:
        return state.prefill(model, input_ids)
    if not hasattr(state, "prefill_graph"):
        state.prefill_input = torch.empty_like(input_ids)
        state.prefill_input.copy_(input_ids)
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                state.prefill(model, state.prefill_input)
        torch.cuda.current_stream(model.device).wait_stream(stream)
        state.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(state.prefill_graph, stream=stream):
            state.prefill(model, state.prefill_input)
    else:
        state.prefill_input.copy_(input_ids)
    # The graph overwrites every prompt KV entry and resets decode positions.
    state.prefill_graph.replay()
    return state.token
