

Official anonymous repository for:

> **HebbGate: Local Reward-Modulated Parameter Isolation for Exemplar-Free Class-Incremental Learning**  
> NeurIPS 2026 submission 35067

HebbGate is an exemplar-free, single-head class-incremental learning method that learns task-specific channel gates without backpropagating through the gate variables. The gates are updated using a local reward-modulated Hebbian rule that combines activation energy, a margin-based reward, and usage-aware scaling. A κ-decay schedule allows each task to explore a larger subnetwork before consolidating to a sparse task-specific mask.

Class-IL inference protocol. The reported Class-IL results use the batch-selection rule described in Appendix E.3 of the paper. Evaluation batches follow the standard dataset splits and are task-homogeneous. No explicit task identifier or class label is passed to the router. For each batch, all task-conditioned subnetworks are evaluated, their z-normalized confidence scores across batches is used to select the gates. 

Exemplar-free scope. Previous-task samples are not replayed to update the backbone or gates. Confidence calibration is label-free and gradient-free and does not modify model parameters. The confidence moments can be computed post hoc from task training loaders; after this pass, only the per-gate scalar moments are retained for evaluation.

## Repository status

During preparation of the response, we identified that several appendix ablations had been produced at different stages of development, when individual components were tested using different code versions or configurations. To remove these cross-version confounds and enable controlled comparisons, we reran the relevant experiments using one consolidated implementation and the same configuration protocol across methods. Consequently, the newly released standardized-run results may differ from the values in the submitted manuscript but carry the same insights. We report both sets separately: the manuscript values are the originally submitted results, while the released values are the new controlled reruns. We will replace the affected appendix tables with the standardized results in the revised manuscript.

This repository contains the code and released artifacts associated with submission 35067. The current release contains:

- the main training and evaluation entry points;
- raw main-result logs for CIFAR-100 and Tiny-ImageNet-200 under 10-task and 20-task streams;
- raw κ-decay and capacity-ablation logs;
- scripts for aggregating raw logs into paper-oriented CSV tables; and
- the confidence-statistics checkpoints used by the task-routing evaluation.

The large model-weight checkpoints are hosted separately because they exceed the practical GitHub file-size limit.

## Repository structure

```text
.
├── eval_main.py
├── main_loop.py
├── main_loop_longer_stream.py
├── Datasets.py
├── ResNet_comp_X.py
├── Helper_functions.py
├── capacity_schedules_updated.py
├── summarize_main_results.py
├── checkpoints/
│   └── mu_sigma_hebbgate_in_*.pt
├── main_logs/
│   ├── CIF100/Resnet/{T10,T20}/
│   └── imagenet/Resnet/{T10,T20}/
└── Ablations/
    ├── summarize_kappa_results.py
    └── kdecay/CIF100/Resnet/
        ├── dynamic/{cap01,cap0125}/
        └── static/{cap01,cap0125}/
```

In the code and log paths, `imagenet` denotes **Tiny-ImageNet-200**, not ImageNet-100.

## Model checkpoints

Download the model-weight checkpoints from:

**[HebbGate model weights](https://drive.google.com/drive/folders/1q1xM-UxIEKPqoHG-d0Pi0ZcDjJfxTTuF?usp=drive_link)**

Place the downloaded weight files in `checkpoints/`. For the current `eval_main.py` configuration, the expected layout is:

```text
checkpoints/
├── Resnet_CIF100_T10.pt
├── Resnet_CIF100_T20.pt
├── Resnet_imagenet_T10.pt
├── Resnet_imagenet_T20.pt
├── mu_sigma_hebbgate_in_Resnet_CIF100_T10.pt
├── mu_sigma_hebbgate_in_Resnet_CIF100_T20.pt
├── mu_sigma_hebbgate_in_Resnet_imagenet_T10.pt
└── mu_sigma_hebbgate_in_Resnet_imagenet_T20.pt
```

The `mu_sigma_hebbgate_in_*.pt` files contain the per-task confidence-score statistics used by z-normalized Class-IL routing and are already included in this repository.

## Environment

The code uses Python 3.10+ and the following direct third-party packages:

```text
torch
torchvision
numpy
tqdm
```

Install a CUDA-compatible PyTorch build for the available GPU, followed by the remaining dependencies. The current entry points assume CUDA and call `torch.cuda.set_device(...)`.

An exact, version-pinned environment is not yet included in this work-in-progress release.

## Datasets

### CIFAR-100

CIFAR-100 is downloaded automatically through `torchvision.datasets.CIFAR100`. Before running, set the `data_root` argument in the selected entry point to the desired local data directory.

### Tiny-ImageNet-200

Tiny-ImageNet-200 can be downloaded separately or obtained from:

**[Tiny-ImageNet-200 data](https://drive.google.com/drive/folders/1PTBswHa6uOzmERSBrGQwM86vcq73zmKE?usp=sharing)**

Set `root_dir` in the selected entry point to the extracted dataset directory. The loader expects an `ImageFolder`-compatible layout:

```text
tiny-imagenet-200/
├── train/
│   ├── <class-id>/
│   │   └── *.JPEG
│   └── ...
└── val/
    ├── <class-id>/
    │   └── *.JPEG
    └── ...
```

The standard Tiny-ImageNet validation split must therefore be reorganized from `val/images/` into per-class directories before use.

### ImageNet-100

ImageNet-100 is not distributed in this repository or in the linked data folder. It requires manual dataset preparation. 

## Evaluating the released checkpoints

`eval_main.py` evaluates the main HebbGate-IN checkpoints and reports:

- final Task-IL accuracy;
- per-task Task-IL accuracy; and
- z-normalized Class-IL accuracy under task-agnostic confidence-based routing.

Before running:

1. Download the model weights and place them in `checkpoints/`.
2. Set the CIFAR-100 and Tiny-ImageNet-200 paths in `eval_main.py`.
3. Select the stream length in the configuration block:

   ```python
   n_tasks = 10  # choices: 10 or 20
   ```

4. Select the datasets:

   ```python
   datasets = ["CIF100", "imagenet"]
   ```

5. Run:

   ```bash
   python eval_main.py
   ```

With both dataset identifiers enabled, the script evaluates CIFAR-100 and Tiny-ImageNet-200 sequentially for the selected stream length.

## Reproducing the training runs

### 10-task streams

`main_loop.py` contains the training procedure used for the 10-task experiments:

```bash
python main_loop.py
```

### 20-task streams

`main_loop_longer_stream.py` contains the corresponding procedure for the 20-task experiments and the longer-stream capacity schedule:

```bash
python main_loop_longer_stream.py
```

The experiment configuration, including datasets, seeds, learning rates, gate-update rate, reward mode, usage-scaling mode, κ targets, and data paths, is defined in the configuration block near the bottom of each script.

The scripts write training CSV files under `training_logs/`. To use the supplied main-results summarizer, organize completed runs as:

```text
main_logs/
└── <dataset>/
    └── <architecture>/
        └── <T10-or-T20>/
            └── <run>.csv
```

## Summarizing the main results

The repository includes raw logs for CIFAR-100 and Tiny-ImageNet-200 under both stream lengths. Generate the summary tables with:

```bash
python summarize_main_results.py main_logs
```

By default, the script writes the generated tables to:

```text
main_logs/main_results_summary/
├── main_results_runs.csv
├── main_results_by_directory.csv
├── main_results_paper_table.csv
├── main_results_paper_table_multiheader.csv
└── paper_tables/
```

The summary script extracts the final valid `til_avg` and `cil_znorm_avg` values from every run and reports sample mean and standard deviation across the available seeds. Use `--population-std` only if population standard deviation is specifically required.

## κ-decay and capacity ablations

Raw logs for the CIFAR-100 κ-decay and capacity settings are provided in:

```text
Ablations/kdecay/CIF100/Resnet/
├── dynamic/
│   ├── cap01/
│   └── cap0125/
└── static/
    ├── cap01/
    └── cap0125/
```

Generate their summaries with:

```bash
python Ablations/summarize_kappa_results.py Ablations/kdecay
```

The generated files are written to:

```text
Ablations/kdecay/kappa_ablation_results_summary/
```

The summarizer reports warnings for incomplete or invalid CSV runs and excludes those runs from the aggregate statistics.


