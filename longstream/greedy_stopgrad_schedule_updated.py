"""Greedy v2 stop-gradient support for long-stream HebbGate experiments.

Implements a per-task gradient mask derived from the greedy unknown-(T)
allocation plan. Channels allocated to previous tasks remain active in forward
(masking is still handled by task gates), but their gradients are stopped.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from Utility_functions import GatedLayer


class GreedyStopGradController:
    """Stops gradients on channels allocated to previous tasks.

    Allocation convention: channels are assigned contiguously by task in each
    gated layer, using allocation sizes from the unknown-(T) greedy scheduler.
    """

    def __init__(self, model: nn.Module, allocations: List):
        self.model = model
        self.allocations = allocations
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._layers: Dict[str, nn.Module] = {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, GatedLayer)
        }

        self._task_trainable_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self._build_masks()

    def _build_masks(self):
        pointers: Dict[str, int] = {name: 0 for name in self._layers}

        for t, alloc in enumerate(self.allocations):
            per_layer: Dict[str, torch.Tensor] = {}
            for layer_name, module in self._layers.items():
                out_dim = int(module.weight.shape[0])
                mask = torch.zeros(out_dim, dtype=torch.float32)

                k = int(alloc.per_layer_k.get(layer_name, 0))
                start = pointers[layer_name]
                end = min(out_dim, start + max(0, k))
                if end > start:
                    mask[start:end] = 1.0
                pointers[layer_name] = end
                per_layer[layer_name] = mask

            self._task_trainable_masks[t] = per_layer

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_weight_hook(self, trainable_mask: torch.Tensor):
        def _hook(grad: torch.Tensor):
            if grad is None:
                return grad
            m = trainable_mask.to(grad.device)
            if grad.dim() == 4:  # Conv2d [out, in, kh, kw]
                return grad * m.view(-1, 1, 1, 1)
            if grad.dim() == 2:  # Linear [out, in]
                return grad * m.view(-1, 1)
            return grad

        return _hook

    def _make_bias_hook(self, trainable_mask: torch.Tensor):
        def _hook(grad: torch.Tensor):
            if grad is None:
                return grad
            return grad * trainable_mask.to(grad.device)

        return _hook

    def on_task_start(self, task_idx: int):
        self._clear_hooks()
        if task_idx not in self._task_trainable_masks:
            return

        for layer_name, module in self._layers.items():
            trainable_mask = self._task_trainable_masks[task_idx][layer_name]
            self._hooks.append(module.weight.register_hook(self._make_weight_hook(trainable_mask)))
            if module.bias is not None:
                self._hooks.append(module.bias.register_hook(self._make_bias_hook(trainable_mask)))

    def on_task_end(self, task_idx: int):
        _ = task_idx
        self._clear_hooks()
