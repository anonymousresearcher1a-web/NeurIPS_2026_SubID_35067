#!/usr/bin/env python3
"""Summarise kappa-decay ablation-result CSV logs
Expected layout
---------------
Ablations/kdecay
├── CIF100/
│   └── Resnet/
│       ├── static/
│       │   ├── run_seed22.csv
│       │   └── run_seed23.csv
│       └── dynamic/
│           └── run_seed__.csv
└── imagenet/
    └── Resnet/


Outputs
-------
main_results_runs.csv
    One row per CSV file.
main_results_by_directory.csv
    One row per result directory with mean/std/min/max.
main_results_paper_table.csv
    Flat paper-oriented table.
main_results_paper_table_multiheader.csv
    Multi-row-header table when only one architecture is present.
paper_tables/<architecture>_paper_table.csv
paper_tables/<architecture>_paper_table_multiheader.csv
    Separate paper tables for each architecture.

Example
-------
python summarize_main_results.py ./logs
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TASK_COLUMN = "task_idx"
DEFAULT_TIL_COLUMN = "til_avg"
DEFAULT_CIL_COLUMN = "cil_znorm_avg"


@dataclass(frozen=True)
class Run:
    csv_path: Path
    relative_dir: Path
    dataset: str
    architecture: str
    step: str
    seed: str
    final_task_idx: int
    final_til: float
    final_cil: float


@dataclass(frozen=True)
class Group:
    relative_dir: Path
    dataset: str
    architecture: str
    step: str
    runs: tuple[Run, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise final TIL and CIL z-normalised results from CSV logs "
            "organised as dataset/architecture/step directories."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("./kdecay"),
        help="Root log directory. Default: ./logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input_dir>/main_results_summary",
    )
    parser.add_argument(
        "--task-column",
        default=DEFAULT_TASK_COLUMN,
        help=f"Task-index column. Default: {DEFAULT_TASK_COLUMN}",
    )
    parser.add_argument(
        "--til-column",
        default=DEFAULT_TIL_COLUMN,
        help=f"TIL column. Default: {DEFAULT_TIL_COLUMN}",
    )
    parser.add_argument(
        "--cil-column",
        default=DEFAULT_CIL_COLUMN,
        help=f"CIL z-normalised column. Default: {DEFAULT_CIL_COLUMN}",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal places in paper tables. Default: 2",
    )
    parser.add_argument(
        "--population-std",
        action="store_true",
        help="Use population std instead of sample std.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first invalid CSV instead of skipping it.",
    )
    return parser.parse_args()


def natural_key(text: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def parse_accuracy(value: Any) -> float:
    if value is None:
        raise ValueError("missing value")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        raise ValueError("missing value")
    if text.endswith("%"):
        text = text[:-1].strip()
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value!r}")
    return float(f"{number:.12g}")


def parse_task_idx(value: Any) -> int:
    number = float(str(value).strip())
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"invalid task index: {value!r}")
    return int(number)


def read_final_metrics(
    csv_path: Path,
    task_column: str,
    til_column: str,
    cil_column: str,
) -> tuple[int, float, float]:
    """Use max task index and the last valid metric value within that task."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        required = {task_column, til_column, cil_column}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        indexed_rows: list[tuple[int, dict[str, str]]] = []
        for row_number, row in enumerate(reader, start=2):
            raw_task = row.get(task_column)
            if raw_task is None or not str(raw_task).strip():
                continue
            try:
                task_idx = parse_task_idx(raw_task)
            except ValueError as exc:
                raise ValueError(f"row {row_number}: {exc}") from exc
            indexed_rows.append((task_idx, row))

    if not indexed_rows:
        raise ValueError("CSV contains no valid data rows")

    final_task_idx = max(task_idx for task_idx, _ in indexed_rows)
    final_rows = [row for task_idx, row in indexed_rows if task_idx == final_task_idx]

    def last_valid(column: str) -> float:
        for row in reversed(final_rows):
            try:
                return parse_accuracy(row.get(column))
            except (TypeError, ValueError):
                continue
        raise ValueError(
            f"no valid {column!r} value found for final "
            f"{task_column}={final_task_idx}"
        )

    return final_task_idx, last_valid(til_column), last_valid(cil_column)


def extract_seed(filename: str) -> str:
    stem = Path(filename).stem
    for pattern in (
        r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])s(\d+)(?:[_-]|$)",
    ):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return stem


def parse_context(root: Path, directory: Path) -> tuple[str, str, str, Path]:
    relative = directory.relative_to(root)
    parts = relative.parts
    if len(parts) >= 3:
        dataset, architecture, step = parts[-3], parts[-2], parts[-1]
    elif len(parts) == 2:
        dataset, architecture = parts
        step = ""
    elif len(parts) == 1:
        dataset = parts[0]
        architecture = ""
        step = ""
    else:
        dataset = architecture = step = ""
    return dataset, architecture, step, relative


def discover_csv_dirs(root: Path, output_dir: Path) -> list[Path]:
    directories: set[Path] = set()
    output_dir = output_dir.resolve()

    for csv_path in root.rglob("*.csv"):
        if not csv_path.is_file():
            continue
        try:
            csv_path.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass
        directories.add(csv_path.parent)

    return sorted(
        directories,
        key=lambda path: natural_key(path.relative_to(root).as_posix()),
    )


def load_runs(
    root: Path,
    output_dir: Path,
    task_column: str,
    til_column: str,
    cil_column: str,
    strict: bool,
) -> tuple[list[Run], list[str]]:
    runs: list[Run] = []
    warnings: list[str] = []

    for directory in discover_csv_dirs(root, output_dir):
        dataset, architecture, step, relative = parse_context(root, directory)

        for csv_path in sorted(directory.glob("*.csv"), key=lambda p: natural_key(p.name)):
            try:
                final_task_idx, final_til, final_cil = read_final_metrics(
                    csv_path,
                    task_column=task_column,
                    til_column=til_column,
                    cil_column=cil_column,
                )
            except (OSError, csv.Error, ValueError) as exc:
                message = f"Skipped {csv_path}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                warnings.append(message)
                continue

            runs.append(
                Run(
                    csv_path=csv_path,
                    relative_dir=relative,
                    dataset=dataset,
                    architecture=architecture,
                    step=step,
                    seed=extract_seed(csv_path.name),
                    final_task_idx=final_task_idx,
                    final_til=final_til,
                    final_cil=final_cil,
                )
            )

    return runs, warnings


def group_runs(runs: Sequence[Run]) -> list[Group]:
    grouped: dict[Path, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[run.relative_dir].append(run)

    result: list[Group] = []
    for relative_dir in sorted(grouped, key=lambda p: natural_key(p.as_posix())):
        members = sorted(
            grouped[relative_dir],
            key=lambda run: (natural_key(run.seed), natural_key(run.csv_path.name)),
        )
        first = members[0]
        result.append(
            Group(
                relative_dir=relative_dir,
                dataset=first.dataset,
                architecture=first.architecture,
                step=first.step,
                runs=tuple(members),
            )
        )
    return result


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def std(values: Sequence[float], population: bool) -> float | None:
    if len(values) < 2:
        return None
    return statistics.pstdev(values) if population else statistics.stdev(values)


def format_number(value: float | None, precision: int) -> str:
    return "" if value is None else f"{value:.{precision}f}"


def write_dict_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_rows(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def make_run_rows(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(
        runs,
        key=lambda r: (
            natural_key(r.dataset),
            natural_key(r.architecture),
            natural_key(r.step),
            natural_key(r.seed),
        ),
    ):
        rows.append(
            {
                "dataset": run.dataset,
                "architecture": run.architecture,
                "step": run.step,
                "directory": run.relative_dir.as_posix(),
                "seed": run.seed,
                "csv_file": run.csv_path.name,
                "csv_path": str(run.csv_path),
                "final_task_idx": run.final_task_idx,
                "final_til": run.final_til,
                "final_cil_znorm": run.final_cil,
            }
        )
    return rows


def make_group_rows(groups: Sequence[Group], population_std: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        til = [run.final_til for run in group.runs]
        cil = [run.final_cil for run in group.runs]
        rows.append(
            {
                "dataset": group.dataset,
                "architecture": group.architecture,
                "step": group.step,
                "directory": group.relative_dir.as_posix(),
                "n_runs": len(group.runs),
                "n_unique_seeds": len({run.seed for run in group.runs}),
                "seeds": ",".join(run.seed for run in group.runs),
                "files": ";".join(run.csv_path.name for run in group.runs),
                "til_mean": mean(til),
                "til_std": std(til, population_std),
                "til_min": min(til),
                "til_max": max(til),
                "cil_znorm_mean": mean(cil),
                "cil_znorm_std": std(cil, population_std),
                "cil_znorm_min": min(cil),
                "cil_znorm_max": max(cil),
            }
        )
    return rows


def safe_filename(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value
    ).strip("_")
    return cleaned or "all"


def ordered_pairs(groups: Sequence[Group]) -> list[tuple[str, str]]:
    return sorted(
        {(group.dataset, group.step) for group in groups},
        key=lambda item: (natural_key(item[0]), natural_key(item[1])),
    )


def group_lookup(groups: Sequence[Group]) -> dict[tuple[str, str, str], Group]:
    return {
        (group.architecture, group.dataset, group.step): group
        for group in groups
    }


def metric_values(group: Group | None, metric: str) -> list[float]:
    if group is None:
        return []
    if metric == "CIL":
        return [run.final_cil for run in group.runs]
    return [run.final_til for run in group.runs]


def make_architecture_flat_table(
    groups: Sequence[Group],
    architecture: str,
    precision: int,
    population_std: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    selected = [group for group in groups if group.architecture == architecture]
    pairs = ordered_pairs(selected)
    lookup = group_lookup(selected)

    fields = ["Metric"]
    for dataset, step in pairs:
        prefix = f"{dataset} {step}".strip()
        fields.extend([f"{prefix} mean", f"{prefix} std"])

    rows: list[dict[str, str]] = []
    for metric in ("CIL", "TIL"):
        row: dict[str, str] = {"Metric": metric}
        for dataset, step in pairs:
            values = metric_values(lookup.get((architecture, dataset, step)), metric)
            prefix = f"{dataset} {step}".strip()
            row[f"{prefix} mean"] = format_number(mean(values), precision) if values else ""
            row[f"{prefix} std"] = format_number(
                std(values, population_std), precision
            ) if values else ""
        rows.append(row)

    return fields, rows


def make_architecture_multiheader_table(
    groups: Sequence[Group],
    architecture: str,
    precision: int,
    population_std: bool,
) -> list[list[str]]:
    selected = [group for group in groups if group.architecture == architecture]
    pairs = ordered_pairs(selected)
    lookup = group_lookup(selected)

    dataset_header = ["Metric"]
    step_header = [""]
    statistic_header = [""]

    for dataset, step in pairs:
        dataset_header.extend([dataset, ""])
        step_header.extend([step, ""])
        statistic_header.extend(["mean", "std"])

    rows: list[list[str]] = [dataset_header, step_header, statistic_header]
    for metric in ("CIL", "TIL"):
        row = [metric]
        for dataset, step in pairs:
            values = metric_values(lookup.get((architecture, dataset, step)), metric)
            if values:
                row.extend(
                    [
                        format_number(mean(values), precision),
                        format_number(std(values, population_std), precision),
                    ]
                )
            else:
                row.extend(["", ""])
        rows.append(row)
    return rows


def make_combined_flat_table(
    groups: Sequence[Group],
    precision: int,
    population_std: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    contexts = sorted(
        {(g.architecture, g.dataset, g.step) for g in groups},
        key=lambda item: (
            natural_key(item[0]),
            natural_key(item[1]),
            natural_key(item[2]),
        ),
    )
    lookup = group_lookup(groups)

    fields = ["Metric"]
    for architecture, dataset, step in contexts:
        prefix = " ".join(part for part in (architecture, dataset, step) if part)
        fields.extend([f"{prefix} mean", f"{prefix} std"])

    rows: list[dict[str, str]] = []
    for metric in ("CIL", "TIL"):
        row: dict[str, str] = {"Metric": metric}
        for architecture, dataset, step in contexts:
            values = metric_values(lookup.get((architecture, dataset, step)), metric)
            prefix = " ".join(part for part in (architecture, dataset, step) if part)
            row[f"{prefix} mean"] = format_number(mean(values), precision) if values else ""
            row[f"{prefix} std"] = format_number(
                std(values, population_std), precision
            ) if values else ""
        rows.append(row)
    return fields, rows


def main() -> int:
    args = parse_args()

    if args.precision < 0 or args.precision > 15:
        print("ERROR: --precision must be between 0 and 15", file=sys.stderr)
        return 2

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (input_dir / "kappa_ablation_results_summary").resolve()
    )

    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    try:
        runs, warnings = load_runs(
            input_dir,
            output_dir,
            task_column=args.task_column,
            til_column=args.til_column,
            cil_column=args.cil_column,
            strict=args.strict,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if not runs:
        print("ERROR: no valid CSV files were found", file=sys.stderr)
        return 2

    groups = group_runs(runs)

    write_dict_csv(
        output_dir / "kappa_ablation_results_runs.csv",
        [
            "dataset",
            "architecture",
            "step",
            "directory",
            "seed",
            "csv_file",
            "csv_path",
            "final_task_idx",
            "final_til",
            "final_cil_znorm",
        ],
        make_run_rows(runs),
    )

    write_dict_csv(
        output_dir / "kappa_ablation_results_by_directory.csv",
        [
            "dataset",
            "architecture",
            "step",
            "directory",
            "n_runs",
            "n_unique_seeds",
            "seeds",
            "files",
            "til_mean",
            "til_std",
            "til_min",
            "til_max",
            "cil_znorm_mean",
            "cil_znorm_std",
            "cil_znorm_min",
            "cil_znorm_max",
        ],
        make_group_rows(groups, args.population_std),
    )

    combined_fields, combined_rows = make_combined_flat_table(
        groups,
        precision=args.precision,
        population_std=args.population_std,
    )
    write_dict_csv(
        output_dir / "kappa_ablation_results_paper_table.csv",
        combined_fields,
        combined_rows,
    )

    architectures = sorted({group.architecture for group in groups}, key=natural_key)
    tables_dir = output_dir / "paper_tables"

    for architecture in architectures:
        flat_fields, flat_rows = make_architecture_flat_table(
            groups,
            architecture=architecture,
            precision=args.precision,
            population_std=args.population_std,
        )
        name = safe_filename(architecture)
        write_dict_csv(
            tables_dir / f"{name}_paper_table.csv",
            flat_fields,
            flat_rows,
        )

        multiheader = make_architecture_multiheader_table(
            groups,
            architecture=architecture,
            precision=args.precision,
            population_std=args.population_std,
        )
        write_rows(
            tables_dir / f"{name}_paper_table_multiheader.csv",
            multiheader,
        )

    if len(architectures) == 1:
        write_rows(
            output_dir / "kappa_ablation_results_paper_table_multiheader.csv",
            make_architecture_multiheader_table(
                groups,
                architecture=architectures[0],
                precision=args.precision,
                population_std=args.population_std,
            ),
        )

    std_kind = "population" if args.population_std else "sample"
    print(f"Processed {len(runs)} CSV runs from {len(groups)} directories.")
    print(f"Standard deviation: {std_kind}.")
    print(f"Saved summaries to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
