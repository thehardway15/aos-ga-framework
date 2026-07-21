"""Run the canonical GA on the continuous benchmarks and report the best fitness.

Sweeps the classic GA baseline (SBX ``p_c=0.9`` then polynomial mutation ``p_m=1.0``)
across the configuration grid ``FUNCTIONS x DIMENSIONS x POPULATION_SIZES x
GENERATION_BUDGETS`` over the study's repetition seeds, and reports the best fitness
found. Unlike the TSP and knapsack reports there is no gap-to-optimum metric: every
benchmark has optimum value 0, so with ``f(x) >= 0`` the raw ``best_objective`` (the
minimal ``f`` found) is already the absolute error to the optimum and is reported and
aggregated as-is. The stdout table and both CSV schemas carry the ``function`` name and
``dimension`` as the two configuration axes.
"""

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aos_ga.core.engine import RunResult, run
from aos_ga.operators.real import SBX, PolynomialMutation
from aos_ga.rng import run_generator
from aos_ga.variation.canonical import CanonicalPipeline

from ..configs import GENERATION_BUDGETS as GENERATION_BUDGETS
from ..configs import POPULATION_SIZES as POPULATION_SIZES
from ..datasets.seeds import load_repetition_seeds
from ..problems.continuous import (
    RASTRIGIN,
    ROSENBROCK,
    SPHERE,
    BenchmarkFunction,
    ContinuousProblem,
)

_CSV_COLUMNS = [
    "function",
    "dimension",
    "population_size",
    "generations",
    "seed",
    "best_objective",
]

_AGG_CSV_COLUMNS = [
    "function",
    "dimension",
    "population_size",
    "generations",
    "n_seeds",
    "mean_fitness",
    "std_fitness",
    "median_fitness",
    "min_fitness",
    "max_fitness",
]

FUNCTIONS = (SPHERE, RASTRIGIN, ROSENBROCK)
DIMENSIONS = (5, 10)


@dataclass(frozen=True)
class RunRecord:
    """One CGA run's outcome: its configuration, seed and best fitness found."""

    function: str
    dimension: int
    population_size: int
    generations: int
    seed: int
    best_objective: float


@dataclass(frozen=True)
class CellResult:
    """All runs for one grid cell, exposing their best-fitness aggregates.

    A cell is a fixed (function, dimension, population size, generation budget); the
    aggregates (mean, sample std, median, min, max) summarize ``best_objective`` over
    the cell's seeds and feed both the stdout table and the aggregated CSV.
    """

    function: str
    dimension: int
    population_size: int
    generations: int
    records: tuple[RunRecord, ...]

    @property
    def fitnesses(self) -> tuple[float, ...]:
        """The best-fitness value of each run, in record order."""
        return tuple(r.best_objective for r in self.records)

    @property
    def mean(self) -> float:
        """Mean best fitness over the cell's runs (0.0 for an empty cell)."""
        return float(np.mean(self.fitnesses)) if self.records else 0.0

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1) of best fitness; 0.0 for fewer than two runs."""
        return float(np.std(self.fitnesses, ddof=1)) if len(self.records) > 1 else 0.0

    @property
    def median(self) -> float:
        """Median best fitness over the cell's runs (0.0 for an empty cell)."""
        return float(np.median(self.fitnesses)) if self.records else 0.0

    @property
    def minimum(self) -> float:
        """Best (lowest) fitness over the cell's runs (0.0 for an empty cell)."""
        return float(np.min(self.fitnesses)) if self.records else 0.0

    @property
    def maximum(self) -> float:
        """Worst (highest) fitness over the cell's runs (0.0 for an empty cell)."""
        return float(np.max(self.fitnesses)) if self.records else 0.0


def format_table(cells: list[CellResult]) -> str:
    """Render a pipe-separated stdout table: a header plus one row per cell."""
    header = (
        "Function",
        "Dimension",
        "Population Size",
        "Generations",
        "Seeds",
        "Mean Fitness",
        "Std Fitness",
        "Median Fitness",
        "Min Fitness",
        "Max Fitness",
    )
    lines = [" | ".join(header)]
    for cell in cells:
        line = (
            cell.function,
            str(cell.dimension),
            str(cell.population_size),
            str(cell.generations),
            str(len(cell.records)),
            f"{cell.mean:.6f}",
            f"{cell.std:.6f}",
            f"{cell.median:.6f}",
            f"{cell.minimum:.6f}",
            f"{cell.maximum:.6f}",
        )
        lines.append(" | ".join(line))
    return "\n".join(lines)


def write_aggregated_csv(path: Path, cells: list[CellResult]) -> None:
    """Write one aggregated row per cell (the versioned per-cell statistics)."""
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_AGG_CSV_COLUMNS)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "function": cell.function,
                    "dimension": cell.dimension,
                    "population_size": cell.population_size,
                    "generations": cell.generations,
                    "n_seeds": len(cell.records),
                    "mean_fitness": cell.mean,
                    "std_fitness": cell.std,
                    "median_fitness": cell.median,
                    "min_fitness": cell.minimum,
                    "max_fitness": cell.maximum,
                }
            )


def write_csv(path: Path, records: list[RunRecord]) -> None:
    """Write the raw per-run rows (one line per record, best fitness kept verbatim)."""
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "function": record.function,
                    "dimension": record.dimension,
                    "population_size": record.population_size,
                    "generations": record.generations,
                    "seed": record.seed,
                    "best_objective": record.best_objective,
                }
            )


def run_cga(
    problem: ContinuousProblem, seed: int, *, population_size: int, generations: int
) -> RunResult[list[float]]:
    """One CGA run on ``problem`` from ``seed``: SBX ``p_c=0.9`` then polynomial ``p_m=1.0``.

    The polynomial step scales with the box width ``span = upper - lower`` of ``problem``.
    """
    pipeline = CanonicalPipeline(
        SBX(), 0.9, PolynomialMutation(span=problem.upper - problem.lower), 1.0
    )

    return run(
        problem,
        pipeline,
        run_generator(seed),
        population_size=population_size,
        generations=generations,
    )


def evaluate_cell(
    function: BenchmarkFunction,
    dimension: int,
    *,
    population_size: int,
    generations: int,
    seeds: list[int],
) -> CellResult:
    """Run the CGA once per seed on ``function`` at ``dimension`` and collect the records."""
    problem = ContinuousProblem(function, dimension)

    records: list[RunRecord] = []
    for seed in seeds:
        result = run_cga(problem, seed, population_size=population_size, generations=generations)
        record = RunRecord(
            function=function.name,
            dimension=dimension,
            population_size=population_size,
            generations=generations,
            seed=seed,
            best_objective=result.best_objective,
        )
        records.append(record)

    return CellResult(
        function=function.name,
        dimension=dimension,
        population_size=population_size,
        generations=generations,
        records=tuple(records),
    )


def evaluate_sweep(seeds: list[int]) -> list[CellResult]:
    """Evaluate every grid cell over ``seeds``, reporting per-cell progress on stderr."""
    cells = []
    for function, dimension, population_size, generations in itertools.product(
        FUNCTIONS, DIMENSIONS, POPULATION_SIZES, GENERATION_BUDGETS
    ):
        cell = evaluate_cell(
            function,
            dimension,
            population_size=population_size,
            generations=generations,
            seeds=seeds,
        )
        print(
            f"Evaluated cell: {function.name}, dim={dimension}, pop={population_size}, "
            f"gen={generations}, mean fitness={cell.mean:.4f}",
            file=sys.stderr,
        )
        cells.append(cell)
    return cells


def main() -> None:
    """Run the sweep, print the table and optionally write the raw and aggregated CSVs."""
    argument_parser = argparse.ArgumentParser(
        description="Run CGA on the continuous benchmark functions and report results."
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
