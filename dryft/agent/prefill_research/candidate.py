"""Research-only fixed-shape prefill graph; decode kernels are unchanged."""
import torch

from decode import DecodeState
from engine import Engine


class PrefillGraphState(DecodeState):
    def prefill(self, model, input_ids):
        if input_ids.numel() > 2048:
            return super().prefill(model, input_ids)
        if not hasattr(self, "prefill_graph"):
            self.prefill_input = torch.empty_like(input_ids)
            self.prefill_input.copy_(input_ids)
            stream = torch.cuda.Stream(device=model.device)
            stream.wait_stream(torch.cuda.current_stream(model.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    super().prefill(model, self.prefill_input)
            torch.cuda.current_stream(model.device).wait_stream(stream)
            self.prefill_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.prefill_graph, stream=stream):
                super().prefill(model, self.prefill_input)
        else:
            self.prefill_input.copy_(input_ids)
        self.prefill_graph.replay()
        return self.token


class PrefillGraphEngine(Engine):
    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        with torch.inference_mode():
            shape = (len(input_ids), len(input_ids[0]), max_new_tokens)
            if self.state is None or self.state.shape != shape:
                self.state = None
                self.state = PrefillGraphState(self.model, *shape, speculative=True)
            yield from super().generate(input_ids, max_new_tokens)
