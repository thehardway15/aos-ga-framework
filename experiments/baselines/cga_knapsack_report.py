import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aos_ga.core.engine import RunResult, run
from aos_ga.core.problem import Problem
from aos_ga.operators.binary import BitFlipMutation, UniformCrossover
from aos_ga.rng import run_generator
from aos_ga.variation.canonical import CanonicalPipeline

from ..configs import GENERATION_BUDGETS as GENERATION_BUDGETS
from ..configs import POPULATION_SIZES as POPULATION_SIZES
from ..datasets.knapsack import load_instance, load_optima
from ..datasets.seeds import load_repetition_seeds
from ..problems.knapsack import KnapsackProblem

INSTANCES = (
    "n20_uncorrelated",
    "n20_weakly",
    "n20_strongly",
    "n30_uncorrelated",
    "n30_weakly",
    "n30_strongly",
    "n50_uncorrelated",
    "n50_weakly",
    "n50_strongly",
)

_CSV_COLUMNS = [
    "instance_id",
    "correlation_type",
    "population_size",
    "generations",
    "seed",
    "best_objective",
    "optimum",
    "gap",
]

_AGG_CSV_COLUMNS = [
    "instance_id",
    "correlation_type",
    "population_size",
    "generations",
    "n_seeds",
    "mean_gap",
    "std_gap",
    "median_gap",
    "min_gap",
    "max_gap",
]


@dataclass(frozen=True)
class RunRecord:
    instance_id: str
    correlation_type: str
    population_size: int
    generations: int
    seed: int
    best_objective: float
    optimum: int
    gap: float


@dataclass(frozen=True)
class CellResult:
    instance_id: str
    correlation_type: str
    population_size: int
    generations: int
    records: tuple[RunRecord, ...]

    @property
    def gaps(self) -> tuple[float, ...]:
        return tuple(record.gap for record in self.records)

    @property
    def mean(self) -> float:
        return float(np.mean(self.gaps)) if self.gaps else 0.0

    @property
    def std(self) -> float:
        if not self.gaps:
            return 0.0
        if len(self.gaps) < 2:
            return 0.0
        return float(np.std(self.gaps, ddof=1))

    @property
    def median(self) -> float:
        return float(np.median(self.gaps)) if self.gaps else 0.0

    @property
    def minimum(self) -> float:
        return min(self.gaps) if self.gaps else 0.0

    @property
    def maximum(self) -> float:
        return max(self.gaps) if self.gaps else 0.0


def format_table(cells: list[CellResult]) -> str:
    header = (
        "Instance ID",
        "Correlation Type",
        "Population Size",
        "Generations",
        "Seeds",
        "Mean Gap",
        "Std Gap",
        "Median Gap",
        "Min Gap",
        "Max Gap",
    )
    lines = [" | ".join(header)]
    for cell in cells:
        lines.append(
            " | ".join(
                map(
                    str,
                    [
                        cell.instance_id,
                        cell.correlation_type,
                        cell.population_size,
                        cell.generations,
                        len(cell.records),
                        cell.mean,
                        cell.std,
                        cell.median,
                        cell.minimum,
                        cell.maximum,
                    ],
                )
            )
        )
    return "\n".join(lines)


def gap_to_optimum(best_objective: float, optimum: int) -> float:
    if optimum == 0:
        raise ValueError("Optimum cannot be zero when computing gap.")
    return (optimum - best_objective) / optimum


def write_aggregated_csv(path: Path, cells: list[CellResult]) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_AGG_CSV_COLUMNS)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "instance_id": cell.instance_id,
                    "correlation_type": cell.correlation_type,
                    "population_size": cell.population_size,
                    "generations": cell.generations,
                    "n_seeds": len(cell.records),
                    "mean_gap": cell.mean,
                    "std_gap": cell.std,
                    "median_gap": cell.median,
                    "min_gap": cell.minimum,
                    "max_gap": cell.maximum,
                }
            )


def write_csv(path: Path, cells: list[RunRecord]) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for record in cells:
            writer.writerow(
                {
                    "instance_id": record.instance_id,
                    "correlation_type": record.correlation_type,
                    "population_size": record.population_size,
                    "generations": record.generations,
                    "seed": record.seed,
                    "best_objective": record.best_objective,
                    "optimum": record.optimum,
                    "gap": record.gap,
                }
            )


def run_cga(
    problem: Problem[list[int]], seed: int, *, population_size: int, generations: int
) -> RunResult[list[int]]:
    pipeline = CanonicalPipeline(UniformCrossover(), 0.9, BitFlipMutation(), 1.0)

    return run(
        problem,
        pipeline,
        run_generator(seed),
        population_size=population_size,
        generations=generations,
    )


def evaluate_cell(
    instance_id: str, *, population_size: int, generations: int, seeds: list[int]
) -> CellResult:
    problem = KnapsackProblem(load_instance(instance_id))
    optimum = next(e for e in load_optima() if e.instance_id == instance_id).optimum

    records: list[RunRecord] = []
    for seed in seeds:
        result = run_cga(problem, seed, population_size=population_size, generations=generations)
        best_objective = result.best_objective
        gap = gap_to_optimum(best_objective, optimum)
        record = RunRecord(
            instance_id=instance_id,
            correlation_type=problem.correlation_type,
            population_size=population_size,
            generations=generations,
            seed=seed,
            best_objective=best_objective,
            optimum=optimum,
            gap=gap,
        )
        records.append(record)

    return CellResult(
        instance_id=instance_id,
        correlation_type=problem.correlation_type,
        population_size=population_size,
        generations=generations,
        records=tuple(records),
    )


def evaluate_sweep(seeds: list[int]) -> list[CellResult]:
    cells = []
    for instance_id, population_size, generations in itertools.product(
        INSTANCES, POPULATION_SIZES, GENERATION_BUDGETS
    ):
        cell = evaluate_cell(
            instance_id,
            population_size=population_size,
            generations=generations,
            seeds=seeds,
        )
        print(
            f"Evaluated cell: {instance_id}, pop={population_size}, "
            f"gen={generations}, mean gap={cell.mean:.4f}",
            file=sys.stderr,
        )
        cells.append(cell)
    return cells


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Run CGA on Knapsack instances and report results."
    )
    argument_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        required=False,
        help="Number of random seeds for reproducibility.",
    )
    argument_parser.add_argument(
        "--csv",
        type=str,
        default=None,
        required=False,
        help="Path to output CSV file for raw run records.",
    )
    argument_parser.add_argument(
        "--agg-csv",
        type=str,
        default=None,
        required=False,
        help="Path to output CSV file for aggregated per-cell statistics.",
    )

    args = argument_parser.parse_args()
    seeds = load_repetition_seeds()
    if args.seeds is not None:
        seeds = seeds[: args.seeds]

    cells = evaluate_sweep(seeds)
    print(format_table(cells))
    if args.csv:
        write_csv(Path(args.csv), [record for cell in cells for record in cell.records])
    if args.agg_csv:
        write_aggregated_csv(Path(args.agg_csv), cells)


if __name__ == "__main__":
    main()
