# -*- coding: utf-8 -*-
from __future__ import annotations

# -----------------------------------------------------------------
# Reward-sign modes
# -----------------------------------------------------------------
REWARD_BATCH = "batch"  # +1 if batch accuracy > 0.5 else −1
REWARD_SAMPLE = "sample"  # sign per sample, then average
REWARD_MARGIN = "margin"  # raw classification margin, no ±1 clipping

from datetime import datetime
from ResNet_comp_X import *
from Datasets import *
from Helper_functions import *

from capacity_schedules_updated import (
    UnknownTHorizonGreedySchedule,
    known_t_uniform_schedule,
    known_t_decay_schedule
)

def set_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def build_class_slices(n_tasks, classes_per_task, start=0):
    return {t: torch.arange(start + t * classes_per_task, start + (t + 1) * classes_per_task)
            for t in range(n_tasks)}

def _build_capacity_aware_kappa_initials(
    n_tasks: int,
    kappa_peak: float,
) -> List[float]:
    """Match `main_loop.py` task-start kappa schedule exactly."""
    initials: List[float] = []
    for t_idx in range(n_tasks):
        capacity_left = (n_tasks - t_idx) / n_tasks
        initials.append(float(min(capacity_left, float(kappa_peak))))
    return initials


if __name__ == "__main__":
    device_n = 0
    torch.cuda.set_device(device_n)
    print(torch.cuda.current_device())

    n_tasks = 10                        # Choices: {10, 20}   #
    datasets = ['CIF100', 'imagenet']   # Choices: {For tiny-imagenet-200: 'imagenet', For CIFAR-100: 'CIF100'}   #
    architecture = 'Resnet'             # 'WideCNN'
    seed = 22                           # , , 1952, 30, 7, 1961, 13, 32]

    save = False
    load_ = True
    train_ = False
    save_model = False
    eval_speed = True

    #
    counter = 0
    conf = 1

    #  #  ------ CONFIG -------  #  #

    norm = "in"
    per_task_bn = False
    gate_update_rate = 10
    kappa_update_rate = 1
    reward = REWARD_MARGIN
    scalemode = 'power'
    gating_mode = 'hard'
    beta = 2
    lr = 0.001,
    g_lr = 0.0003  # , 0.0005]
    kappa_peak = 0.5

    if n_tasks == 20:
        kappa_targets = known_t_decay_schedule(n_tasks=n_tasks)
    else:
        kappa_v = 1/n_tasks
        kappa_targets = [kappa_v for _ in range(n_tasks)]

    #  #  ------ ------ -------  #  #

    for dataset in datasets:
        if dataset == 'CIF10':
            batch_size = 64
            tasks_data = generate_split_cifar10(batch_size=batch_size, data_root="./data", num_workers=0)
            num_classes = 10
            k_fractions = kappa_targets

            class_indices_per_task = build_class_slices(n_tasks=5, classes_per_task=2, start=0)
        elif dataset == 'imagenet':
            batch_size = 128
            tasks_data = generate_split_tiny_imagenet(
                root_dir="./../../../data/tiny-imagenet-200",
                n_tasks=n_tasks, batch_size=batch_size, num_workers=0,
                shuffle_classes=False, seed=seed
            )
            num_classes = 200
            k_fractions = kappa_targets
            class_indices_per_task = build_class_slices(n_tasks=n_tasks, classes_per_task=int(num_classes/n_tasks), start=0)
        else:
            batch_size = 64
            tasks_data = generate_split_cifar100(batch_size=batch_size, data_root="./../../../HebbGate_X2/data", n_tasks=n_tasks,
                                                 num_workers=0)
            num_classes = 100
            k_fractions = kappa_targets
            class_indices_per_task = build_class_slices(n_tasks=n_tasks, classes_per_task=int(num_classes/n_tasks), start=0)

        num_tasks = len(tasks_data)
        device_eval = "cuda" if torch.cuda.is_available() else "cpu"
        time_code = datetime.now().strftime(
            "%H%M_%d%m%y")  # e.g. "1642_070325"

        counter += 1
        seed_everything(seed)
        k = k_fractions[0]

        print(
            f"Experiment: \n Weights_LR: {lr}\n Gating_LR:{g_lr}\n"
            f" Gate_Update_Rate: {gate_update_rate}\n Reward_Mode: {reward}\n"
            f" Scale_Mode: {scalemode}\n Gating_Mode: {gating_mode}\n")

        if architecture == 'Resnet':
            model = GatedResNet18_Flex(
                num_classes=num_classes,
                kappa=k,
                stem=dataset,
                norm_type=norm,
                per_task_bn=per_task_bn,
                gate_skip=False,
                gate_head=False
            )
            model.set_class_slices(class_indices_per_task)
            model.configure_head_protection(freeze_old_cols=True,
                                            mask_non_current_logits=False)
        elif architecture == 'WideCNN':
            model = WideCNN(num_classes=num_classes)
        else:
            raise(ValueError(f'Architecture {architecture} is not valid'))

        if load_:
            model_name = f"{architecture}_{dataset}_T{n_tasks}"

            load_model_with_gates(model,
                                  f"checkpoints/{model_name}.pt", device="cuda")
            try:
                print(f'\nLoading Confidence statistics for model: {model_name}')
                mu, sigma, _ = load_mu_sigma(f"checkpoints/mu_sigma_hebbgate_in_{model_name}.pt", device="cuda")
                model.mu = mu
                model.sigma = sigma
                print('Confidence statistics loaded!\n')
            except:
                print(f"Confidence statistics (mu, sigma) for model: {model_name} not found!")
                print('Computing Confidence stats (mu, sigma) ...')
                mu, sigma = compute_conf_stats(model, tasks_data, n_tasks=num_tasks, kappa_targets=k_fractions,
                                               max_batches=10,
                                               device=device_eval)
                save_mu_sigma(f"checkpoints/mu_sigma_hebbgate_in_{model_name}.pt", mu, sigma,
                              extra={"num_tasks": num_tasks, "mode": "confidence"})
                print(f'Done - Saved at: checkpoints/mu_sigma_hebbgate_in_{model_name}.pt')
                model.mu = mu
                model.sigma = sigma

            print("Task-Incremental Setting")
            print("Evaluating -------------")
            acc_TIL = eval_all(model, tasks_data,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               kappa_targets=kappa_targets,
                               up2task=num_tasks)
            print("Per‑task TIL accuracies:", [f"{a:.1f}%" for a in acc_TIL])
            print("Average TIL accuracy:", sum(acc_TIL) / len(acc_TIL))
            print("\n---------------------------------------------\n")

            ###############################

            acc_CIL_znorm = eval_CIL_ptKappa_znorm(
                model, tasks_data,
                device=device_eval,
                kappa_targets=kappa_targets,
                n_tasks=n_tasks,
                dataset=dataset,
                criterion="confidence",
                norm_mode="z",
                max_batches=10
            )

            cil_znorm = acc_CIL_znorm
            cil_znorm_avg = sum(acc_CIL_znorm) / len(acc_CIL_znorm)
            print(f"    CIL-znorm  avg={cil_znorm_avg:.2f}  per-task={format_per_task(acc_CIL_znorm)}")

            #   -------------------------------------------------------------------  #

        current_time = datetime.now().strftime(
            "%H%M_%d%m%y")  # e.g. "1642_070325"



