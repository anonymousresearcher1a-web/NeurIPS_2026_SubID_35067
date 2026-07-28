from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def _forward_masked(model, x, task_id: int, kappa: float):
    return model(x, task_id=task_id, k_frac=float(kappa))


def _normalize_kappas(kappa_targets, n_tasks: int) -> List[float]:
    if kappa_targets is None:
        return [1.0 / n_tasks for _ in range(n_tasks)]
    if len(kappa_targets) < n_tasks:
        raise ValueError(f"Expected at least {n_tasks} kappa targets, got {len(kappa_targets)}")
    return [float(kappa_targets[t]) for t in range(n_tasks)]


@torch.no_grad()
def eval_CIL_ptKappa_taskwise(
    model,
    tasks_data: List[Tuple],
    n_tasks: int | None = None,
    device: str = "cpu",
    kappa_targets=None,
):
    """
    Class-IL with candidate-task-specific kappa.

    Key compatibility fix for long-stream runs:
    when probing candidate gate t, use kappa_targets[t] (not kappa_targets[task_ref]).
    """
    model.eval()
    if n_tasks is None:
        n_tasks = len(tasks_data)

    kappas = _normalize_kappas(kappa_targets, n_tasks)
    acc = [0.0] * n_tasks
    cnt = [0] * n_tasks

    for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
        for xb, yb in tqdm(loader, desc=f"Eval | CIL-taskwise | task {task_ref}"):
            xb, yb = xb.to(device), yb.to(device)

            best_score = -1e9
            best_logits = None
            for t in range(n_tasks):
                logits = _forward_masked(model, xb, t, kappas[t])
                score = F.softmax(logits, dim=1).max(1).values.mean()
                if score > best_score:
                    best_score = score
                    best_logits = logits

            pred = best_logits.argmax(1)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)

    return [100.0 * a / c if c else 0.0 for a, c in zip(acc, cnt)]


@torch.no_grad()
def _batch_task_confidences_taskwise(model, xb, n_tasks: int, kappas: List[float]):
    scores = []
    for t in range(n_tasks):
        logits = _forward_masked(model, xb, t, kappas[t])
        conf = torch.softmax(logits, dim=1).max(1).values
        scores.append(conf)
    return torch.stack(scores, dim=0)


@torch.no_grad()
def _compute_conf_stats_taskwise(model, tasks_data, n_tasks: int, kappas: List[float], device: str, max_batches=None):
    eps = 1e-8
    sums = torch.zeros(n_tasks, device=device)
    sums2 = torch.zeros(n_tasks, device=device)
    counts = torch.zeros(n_tasks, device=device)

    model.eval()
    for loader, _ in tasks_data[:n_tasks]:
        for i, (xb, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            S = _batch_task_confidences_taskwise(model, xb, n_tasks, kappas)
            sums += S.sum(dim=1)
            sums2 += (S ** 2).sum(dim=1)
            counts += S.size(1)

    mu = sums / (counts + eps)
    var = (sums2 / (counts + eps)) - mu ** 2
    var.clamp_(min=0.0)
    sigma = (var + eps).sqrt()
    return mu.detach(), sigma.detach()


@torch.no_grad()
def eval_CIL_ptKappa_znorm_taskwise(
    model,
    tasks_data: List[Tuple],
    device: str = "cpu",
    kappa_targets=None,
    n_tasks: int | None = None,
    norm_mode: str = "z",
    max_batches: int = 10,
):
    """Z-normalized Class-IL evaluation with candidate-task-specific kappa values."""
    model.eval()
    if n_tasks is None:
        n_tasks = len(tasks_data)

    kappas = _normalize_kappas(kappa_targets, n_tasks)
    acc = [0.0] * n_tasks
    cnt = [0] * n_tasks
    eps = 1e-8

    mu, sigma = _compute_conf_stats_taskwise(
        model,
        tasks_data,
        n_tasks=n_tasks,
        kappas=kappas,
        device=device,
        max_batches=max_batches,
    )

    for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
        for xb, yb in tqdm(loader, desc=f"Eval | CIL-znorm-taskwise | task {task_ref}"):
            xb, yb = xb.to(device), yb.to(device)

            best_score = -1e9
            best_logits = None
            for t in range(n_tasks):
                logits = _forward_masked(model, xb, t, kappas[t])
                conf = torch.softmax(logits, dim=1).max(1).values

                if norm_mode == "z":
                    score = ((conf - mu[t]) / (sigma[t] + eps)).mean()
                elif norm_mode == "mean_ratio":
                    score = (conf / (mu[t] + eps)).mean()
                else:
                    score = conf.mean()

                if score > best_score:
                    best_score = score
                    best_logits = logits

            pred = best_logits.argmax(1)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)

    return [100.0 * a / c if c else 0.0 for a, c in zip(acc, cnt)]
