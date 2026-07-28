"""Updated capacity-scheduling utilities for long-stream HebbGate rebuttal experiments.

This module adds an unknown-(T) greedy schedule that does not depend on the
final stream horizon. It tracks remaining channels per gated layer and creates
per-task allocation plans that can be mapped to HebbGate's global kappa API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import torch.nn as nn

from Utility_functions import GatedLayer


@dataclass
class LayerCapacityState:
    layer_name: str
    total_channels: int
    min_per_task: int
    remaining: int
    saturated_at_task: int | None = None
    allocations: List[int] = field(default_factory=list)


@dataclass
class TaskCapacityAllocation:
    task_idx: int
    per_layer_k: Dict[str, int]
    global_kappa_proxy: float


class UnknownTHorizonGreedySchedule:
    """Task-horizon-independent channel allocator.

    For each new task and each layer l:
        k_{l,t} = max(m_l, r * R_l), clipped to R_l

    where R_l is remaining channels in layer l.
    """

    def __init__(self, model: nn.Module, decay_ratio: float, min_channels: Dict[str, int] | int = 1):
        self.decay_ratio = float(decay_ratio)
        if not (0.0 < self.decay_ratio <= 1.0):
            raise ValueError("decay_ratio must be in (0, 1].")

        self.layers: List[Tuple[str, GatedLayer]] = []
        for name, module in model.named_modules():
            if isinstance(module, GatedLayer):
                self.layers.append((name, module))

        if not self.layers:
            raise ValueError("No GatedLayer modules found in model.")

        self.state: Dict[str, LayerCapacityState] = {}
        for name, module in self.layers:
            total = int(module.gate_for(0).numel()) if len(module._gate_bank) > 0 else int(module._dim)
            if isinstance(min_channels, dict):
                m_l = int(min_channels.get(name, 1))
            else:
                m_l = int(min_channels)
            m_l = max(1, min(m_l, total))
            self.state[name] = LayerCapacityState(
                layer_name=name,
                total_channels=total,
                min_per_task=m_l,
                remaining=total,
            )

    def allocate_for_task(self, task_idx: int) -> TaskCapacityAllocation:
        per_layer_k: Dict[str, int] = {}
        total_alloc = 0
        total_capacity = 0

        for layer_name, _ in self.layers:
            s = self.state[layer_name]
            total_capacity += s.total_channels

            if s.remaining <= 0:
                k = 0
                if s.saturated_at_task is None:
                    s.saturated_at_task = task_idx
            else:
                # Meaningful chunk from remaining capacity.
                # If even this natural chunk falls below the minimum per-task budget,
                # freeze this layer for subsequent tasks (do not force tiny 1-channel
                # allocations forever).
                requested = int(self.decay_ratio * s.remaining)
                if requested < s.min_per_task:
                    k = 0
                    s.saturated_at_task = task_idx
                else:
                    k = min(requested, s.remaining)
                    s.remaining -= k
                    if s.remaining <= 0 and s.saturated_at_task is None:
                        s.saturated_at_task = task_idx

            s.allocations.append(int(k))
            per_layer_k[layer_name] = int(k)
            total_alloc += int(k)

        # Proxy because current HebbGate API expects a scalar kappa.
        # We map layer-wise allocations to a weighted global fraction.
        global_kappa_proxy = float(total_alloc / max(1, total_capacity))

        return TaskCapacityAllocation(
            task_idx=task_idx,
            per_layer_k=per_layer_k,
            global_kappa_proxy=global_kappa_proxy,
        )

    def export_layer_summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for layer_name, _ in self.layers:
            s = self.state[layer_name]
            rows.append(
                {
                    "layer": layer_name,
                    "total_channels": s.total_channels,
                    "per_task_greedy_allocations": s.allocations,
                    "freeze_or_saturation_task": s.saturated_at_task,
                    "residual_capacity": s.remaining,
                }
            )
        return rows


def known_t_uniform_schedule(n_tasks: int, kappa_value: float) -> List[float]:
    """Simple fixed-(T) oracle baseline schedule with constant per-task kappa."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    k = float(kappa_value)
    return [k for _ in range(n_tasks)]


def known_t_decay_schedule(n_tasks: int) -> List[float]:
    """Simple fixed-(T) oracle baseline schedule with decay per-task kappa."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")

    kappa_un = float(1/n_tasks)
    kappa_max = kappa_un * 1.2
    kappa_min = kappa_un * 0.8
    kappa_range = kappa_max - kappa_min

    kappas = []
    for t_n in range(n_tasks):
        k = 0.99 * (kappa_max - (kappa_range / (n_tasks-1) * t_n))
        kappas.append(k)

    # print(kappa_un, kappa_min, kappa_max)
    # print(kappas)
    # print(sum(kappas))

    return kappas
