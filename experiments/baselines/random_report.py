"""Measure the random-selection baseline: each pool as the whole variation, per variant.

Sweeps each pool variant (full and reduced) of every representation as the whole
variation -- one :class:`~aos_ga.variation.random_operator.RandomOperatorStep` drawing
uniformly from that pool -- across the grid ``instances x POPULATION_SIZES x
GENERATION_BUDGETS`` over the study's repetition seeds, and takes the median (with the
mean, sample std, min and max) of each variant's final quality. It is the lower reference
point (Random selection, ``p_i = 1/K``) for the adaptive operator-selection strategies.

Three families (TSP, knapsack, continuous) share one schema through the
:class:`~experiments.configs.families.FamilyDescriptor`; ``problem`` names the family and
``instance_id`` is the problem's own name. Unlike the single-operator reference
(``fbo_report``), the pool is measured as one configuration, so a sweep cell's unit is a
pool variant, not an operator, and there is no oracle: the run emits two artifacts, the
raw per-run rows and the per-pool-variant statistics, both keyed by the pool variant.
"""

import argparse
import csv
import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aos_ga.core.engine import RunResult, run
from aos_ga.core.operator import Operator
from aos_ga.core.problem import Problem
from aos_ga.rng import run_generator
from aos_ga.variation.random_operator import RandomOperatorStep

from ..configs import GENERATION_BUDGETS as GENERATION_BUDGETS
from ..configs import POPULATION_SIZES as POPULATION_SIZES
from ..configs.families import FAMILIES as FAMILIES
from ..configs.families import FamilyDescriptor as FamilyDescriptor
from ..configs.pools import PoolVariant, build_pool
from ..datasets.seeds import load_repetition_seeds


@dataclass(frozen=True)
class RunRecord:
    """One pool variant's outcome on one seed: its configuration, seed and best quality."""

    problem: str
    instance_id: str
    population_size: int
    generations: int
    pool_variant: PoolVariant
    seed: int
    best_quality: float


@dataclass(frozen=True)
class PoolResult:
    """All runs of one pool variant in one grid cell, exposing their best-quality aggregates.

    A cell fixes (problem, instance, population size, generation budget); ``records`` holds
    one :class:`RunRecord` per seed. The aggregates summarize ``best_quality`` and feed the
    stdout table and the per-pool-variant CSV.
    """

    problem: str
    instance_id: str
    population_size: int
    generations: int
    pool_variant: PoolVariant
    records: tuple[RunRecord, ...]

    @property
    def qualities(self) -> tuple[float, ...]:
        """The ``best_quality`` of each run, in record order."""
        return tuple(record.best_quality for record in self.records)

    @property
    def median(self) -> float:
        """Median ``best_quality`` over the cell's runs (0.0 for an empty cell)."""
        return float(np.median(self.qualities)) if self.qualities else 0.0

    @property
    def mean(self) -> float:
        """Mean ``best_quality`` over the cell's runs (0.0 for an empty cell)."""
        return float(np.mean(self.qualities)) if self.qualities else 0.0

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1) of ``best_quality``; 0.0 for fewer than two runs."""
        if len(self.qualities) < 2:
            return 0.0

        return float(np.std(self.qualities, ddof=1))

    @property
    def minimum(self) -> float:
        """Lowest ``best_quality`` over the cell's runs (0.0 for an empty cell)."""
        return float(np.min(self.qualities)) if self.qualities else 0.0

    @property
    def maximum(self) -> float:
        """Highest ``best_quality`` over the cell's runs (0.0 for an empty cell)."""
        return float(np.max(self.qualities)) if self.qualities else 0.0


def run_random(
    problem: Problem[list[Any]],
    pool: Sequence[Operator[Any]],
    seed: int,
    *,
    population_size: int,
    generations: int,
) -> RunResult[list[Any]]:
    """One GA run on ``problem`` from ``seed`` with ``pool`` drawn uniformly as the variation.

    Wraps ``pool`` in a :class:`~aos_ga.variation.random_operator.RandomOperatorStep` and
    runs the engine with its defaults (tournament ``k=3``, one elite) from a fresh
    ``run_generator(seed)``. Returns the :class:`RunResult`; the caller reads ``best_quality``.
    """
    random_step = RandomOperatorStep(pool)

    return run(
        problem,
        random_step,
        run_generator(seed),
        population_size=population_size,
        generations=generations,
    )


def evaluate_pool_variant(
    problem_label: str,
    problem: Problem[list[Any]],
    pool: Sequence[Operator[Any]],
    pool_variant: PoolVariant,
    *,
    population_size: int,
    generations: int,
    seeds: Sequence[int],
) -> PoolResult:
    """Run ``pool`` once per seed on ``problem`` and collect its per-run records.

    ``problem_label`` names the family and ``instance_id`` comes from ``problem.name``.
    Each seed contributes one :class:`RunRecord` built from the run's ``best_quality``, from
    which the median and the other aggregates of the returned :class:`PoolResult` follow.
    """
    records: list[RunRecord] = []

    for seed in seeds:
        result = run_random(
            problem, pool, seed, population_size=population_size, generations=generations
        )
        records.append(
            RunRecord(
                problem=problem_label,
                instance_id=problem.name,
                population_size=population_size,
                generations=generations,
                pool_variant=pool_variant,
                seed=seed,
                best_quality=result.best_quality,
            )
        )

    return PoolResult(
        problem=problem_label,
        instance_id=problem.name,
        population_size=population_size,
        generations=generations,
        pool_variant=pool_variant,
        records=tuple(records),
    )


def evaluate_cell(
    family: FamilyDescriptor,
    spec: tuple[Any, ...],
    *,
    population_size: int,
    generations: int,
    seeds: Sequence[int],
) -> list[PoolResult]:
    """Measure both pool variants of ``family`` for one ``spec`` and one (N, G) configuration.

    Builds the problem and, for each pool variant, the corresponding operator pool (scaled
    by the spec's ``pool_bounds``), then runs the whole pool over ``seeds``. Returns one
    :class:`PoolResult` per variant, full before reduced.
    """
    problem = family.build_problem(spec)
    bounds = family.pool_bounds(spec)

    results: list[PoolResult] = []
    for pool_variant in PoolVariant:
        pool = build_pool(family.representation, pool_variant, real_bounds=bounds)
        result = evaluate_pool_variant(
            family.problem,
            problem,
            pool,
            pool_variant,
            population_size=population_size,
            generations=generations,
            seeds=seeds,
        )
        results.append(result)

    return results


def evaluate_sweep(
    seeds: Sequence[int], *, families: tuple[FamilyDescriptor, ...] = FAMILIES
) -> list[PoolResult]:
    """Evaluate every grid cell of ``families`` over ``seeds``, reporting progress on stderr.

    Iterates ``families x specs x POPULATION_SIZES x GENERATION_BUDGETS`` and, per cell, both
    pool variants, accumulating the per-variant results across all cells. The raw per-run
    records stay recoverable from each :class:`PoolResult`. Returns the flat list, two
    entries per cell (full before reduced).
    """
    pool_results: list[PoolResult] = []

    total_cells = sum(
        len(family.specs) * len(POPULATION_SIZES) * len(GENERATION_BUDGETS) for family in families
    )
    cell_index = 0
    print(f"Sweeping {total_cells} cells over {len(seeds)} seeds...", file=sys.stderr)

    for family in families:
        for spec in family.specs:
            for population_size, generations in itertools.product(
                POPULATION_SIZES, GENERATION_BUDGETS
            ):
                cell_results = evaluate_cell(
                    family,
                    spec,
                    population_size=population_size,
                    generations=generations,
                    seeds=seeds,
                )
                pool_results.extend(cell_results)

                cell_index += 1
                instance_id = cell_results[0].instance_id
                medians = " ".join(
                    f"{result.pool_variant.value}={result.median:.4f}" for result in cell_results
                )
                print(
                    f"[{cell_index}/{total_cells}] {family.problem}/{instance_id} "
                    f"N={population_size} G={generations} -> {medians}",
                    file=sys.stderr,
                )

    return pool_results


_RAW_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "pool_variant",
    "seed",
    "best_quality",
]

_BASELINE_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "pool_variant",
    "n_seeds",
    "median_quality",
    "mean_quality",
    "std_quality",
    "min_quality",
    "max_quality",
]


def format_table(results: list[PoolResult]) -> str:
    """Render a pipe-separated stdout table: a header plus one row per pool variant."""
    header = (
        "Problem",
        "Instance",
        "Population Size",
        "Generations",
        "Pool Variant",
        "Seeds",
        "Median",
        "Mean",
        "Std",
        "Min",
        "Max",
    )
    lines = [" | ".join(header)]
    for result in results:
        row = (
            result.problem,
            result.instance_id,
            str(result.population_size),
            str(result.generations),
            result.pool_variant.value,
            str(len(result.records)),
            f"{result.median:.6f}",
            f"{result.mean:.6f}",
            f"{result.std:.6f}",
            f"{result.minimum:.6f}",
            f"{result.maximum:.6f}",
        )
        lines.append(" | ".join(row))
    return "\n".join(lines)


def write_csv(path: Path, records: list[RunRecord]) -> None:
    """Write the raw per-run rows (one line per record, best_quality kept verbatim)."""
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_RAW_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "problem": record.problem,
                    "instance_id": record.instance_id,
                    "population_size": record.population_size,
                    "generations": record.generations,
                    "pool_variant": record.pool_variant.value,
                    "seed": record.seed,
                    "best_quality": record.best_quality,
                }
            )


def write_baseline_csv(path: Path, results: list[PoolResult]) -> None:
    """Write one aggregated row per pool variant (the versioned per-variant statistics)."""
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_BASELINE_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "problem": result.problem,
                    "instance_id": result.instance_id,
                    "population_size": result.population_size,
                    "generations": result.generations,
                    "pool_variant": result.pool_variant.value,
                    "n_seeds": len(result.records),
                    "median_quality": result.median,
                    "mean_quality": result.mean,
                    "std_quality": result.std,
                    "min_quality": result.minimum,
                    "max_quality": result.maximum,
                }
            )


def main() -> None:
    """Run the sweep, print the table and optionally write the raw and aggregated CSVs."""
    argument_parser = argparse.ArgumentParser(
        description="Measure the random-selection baseline and report the results."
    )
    argument_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        required=False,
        help="Number of repetition seeds to use (defaults to all).",
    )
    argument_parser.add_argument(
        "--family",
        type=str,
        default=None,
        required=False,
        choices=[family.problem for family in FAMILIES],
        help="Restrict the sweep to a single problem family.",
    )
    argument_parser.add_argument(
        "--csv",
        type=str,
        default=None,
        required=False,
        help="Path to output CSV file for raw per-run records.",
    )
    argument_parser.add_argument(
        "--baseline-csv",
        type=str,
        default=None,
        required=False,
        help="Path to output CSV file for per-pool-variant statistics.",
    )

    args = argument_parser.parse_args()
    seeds = load_repetition_seeds()
    if args.seeds is not None:
        seeds = seeds[: args.seeds]

    families = FAMILIES
    if args.family is not None:
        families = tuple(family for family in FAMILIES if family.problem == args.family)

    pool_results = evaluate_sweep(seeds, families=families)
    print(format_table(pool_results))

    if args.csv:
        records = [record for result in pool_results for record in result.records]
        write_csv(Path(args.csv), records)
        print(f"Wrote raw runs: {args.csv}", file=sys.stderr)
    if args.baseline_csv:
        write_baseline_csv(Path(args.baseline_csv), pool_results)
        print(f"Wrote per-variant stats: {args.baseline_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
