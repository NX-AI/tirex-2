# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Replay a forward pass from CUDA graphs: one launch instead of hundreds of small kernels."""

from collections import OrderedDict

import torch


def _key(tensor: torch.Tensor | None):
    return None if tensor is None else (tuple(tensor.shape), tensor.dtype, tensor.cpu().numpy().tobytes())


class CUDAGraphRunner:
    """Run ``build(*structure)(x)`` from a CUDA graph per shape of ``x`` and content of ``structure``.

    ``build`` runs eagerly and may synchronize with the host; the function it
    returns must not. The ``max_graphs`` most recently used graphs are kept.
    """

    def __init__(self, build, max_graphs: int = 32):
        self.build = build
        self.max_graphs = max_graphs
        self._graphs = OrderedDict()
        # Graphs replay one at a time and their outputs are cloned at once, so they can share memory.
        self._pool = torch.cuda.graph_pool_handle()
        # One warm-up stream for all captures: cuBLAS keeps a workspace per stream.
        self._stream = torch.cuda.Stream()

    def __len__(self) -> int:
        return len(self._graphs)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, *structure: torch.Tensor | None) -> torch.Tensor:
        key = (tuple(x.shape), x.dtype, x.device, *map(_key, structure))
        if key not in self._graphs:
            self._graphs[key] = self._capture(x, structure)
            if len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        self._graphs.move_to_end(key)

        graph, static_input, static_output, _ = self._graphs[key]
        static_input.copy_(x)
        graph.replay()
        return static_output.clone()

    def _capture(self, x: torch.Tensor, structure: tuple) -> tuple:
        static_input = x.clone()
        constants = tuple(None if t is None else t.clone() for t in structure)
        fn = self.build(*constants)

        # Warm up outside the capture, which also runs lazy compilation and autotuning.
        self._stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(self._stream):
            fn(static_input)
        torch.cuda.current_stream(x.device).wait_stream(self._stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_output = fn(static_input)
        # The graph reads tensors by address: keep the structure and what fn closes over alive.
        return graph, static_input, static_output, (constants, fn)
