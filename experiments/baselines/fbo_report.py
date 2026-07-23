"""Measure each operator in isolation and derive the per-pool reference operator.

Sweeps every operator of each representation's full pool as the whole variation
(one :class:`~aos_ga.variation.single_operator.SingleOperatorStep` per operator) across
the grid ``instances x POPULATION_SIZES x GENERATION_BUDGETS`` over the study's
repetition seeds, and takes the median final quality of each operator. From those
medians it derives, per grid cell, the reference operator set -- the operators tying for
the best median -- for both the full and the reduced pool, the upper reference point for
the adaptive operator-selection strategies. Each operator runs on its own, so its median
is pool-independent: one full-pool measurement projects onto the reduced pool with no
extra runs.

Three families (TSP, knapsack, continuous) share one schema through a
:class:`FamilyDescriptor`; ``problem`` names the family and ``instance_id`` is the
problem's own name. The run emits three artifacts: the raw per-run rows and the
per-operator statistics (both pool-agnostic), and the reference rows (the only
pool-dependent one, tagged with the pool variant).
"""

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aos_ga.core.engine import RunResult, run
from aos_ga.core.operator import Operator
from aos_ga.core.problem import Problem
from aos_ga.core.representation import Representation
from aos_ga.rng import run_generator
from aos_ga.variation.single_operator import SingleOperatorStep

from ..configs import GENERATION_BUDGETS as GENERATION_BUDGETS
from ..configs import POPULATION_SIZES as POPULATION_SIZES
from ..configs.families import FAMILIES as FAMILIES
from ..configs.families import FamilyDescriptor as FamilyDescriptor
from ..configs.pools import PoolVariant, build_pool, pool_ids
from ..datasets.seeds import load_repetition_seeds
from .fbo_oracle import OperatorOracle, derive_oracle


@dataclass(frozen=True)
class RunRecord:
    """One operator's outcome on one seed: its configuration, seed and best quality."""

    problem: str
    instance_id: str
    population_size: int
    generations: int
    operator_id: str
    seed: int
    best_quality: float


@dataclass(frozen=True)
class OperatorResult:
    """All runs of one operator in one grid cell, exposing their best-quality aggregates.

    A cell fixes (problem, instance, population size, generation budget); ``records`` holds
    one :class:`RunRecord` per seed. The aggregates summarize ``best_quality`` and feed the
    stdout table, the per-operator CSV and -- through the median -- the reference derivation.
    """

    problem: str
    instance_id: str
    population_size: int
    generations: int
    operator_id: str
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
        if not self.qualities:
            return 0.0
        if len(self.qualities) < 2:
            return 0.0

        return float(np.std(self.qualities, ddof=1)) if self.qualities else 0.0

    @property
    def minimum(self) -> float:
        """Lowest ``best_quality`` over the cell's runs (0.0 for an empty cell)."""
        return float(np.min(self.qualities)) if self.qualities else 0.0

    @property
    def maximum(self) -> float:
        """Highest ``best_quality`` over the cell's runs (0.0 for an empty cell)."""
        return float(np.max(self.qualities)) if self.qualities else 0.0


@dataclass(frozen=True)
class OracleRow:
    """The reference operator set for one pool variant in one grid cell.

    Wraps an :class:`OperatorOracle` (the operators tying for the best median and that
    shared median) with the cell key and the pool variant it was derived for. One row of
    the reference CSV.
    """

    problem: str
    instance_id: str
    population_size: int
    generations: int
    pool_variant: PoolVariant
    oracle: OperatorOracle


def run_single_operator(
    problem: Problem[list[Any]],
    operator: Operator[Any],
    seed: int,
    *,
    population_size: int,
    generations: int,
) -> RunResult[list[Any]]:
    """One GA run on ``problem`` from ``seed`` with ``operator`` as the whole variation.

    Wraps ``operator`` in a :class:`~aos_ga.variation.single_operator.SingleOperatorStep`
    and runs the engine with its defaults (tournament ``k=3``, one elite) from a fresh
    ``run_generator(seed)``. Returns the :class:`RunResult`; the caller reads ``best_quality``.
    """
    single_operator = SingleOperatorStep(operator)

    return run(
        problem,
        single_operator,
        run_generator(seed),
        population_size=population_size,
        generations=generations,
    )


def evaluate_operator(
    problem_label: str,
    problem: Problem[list[Any]],
    operator: Operator[Any],
    *,
    population_size: int,
    generations: int,
    seeds: list[int],
) -> OperatorResult:
    """Run ``operator`` once per seed on ``problem`` and collect its per-run records.

    ``problem_label`` names the family and ``instance_id`` comes from ``problem.name``.
    Returns an :class:`OperatorResult` of one :class:`RunRecord` per seed, from which the
    median and the other aggregates follow.
    """
    records: list[RunRecord] = []

    for seed in seeds:
        result = run_single_operator(
            problem, operator, seed, population_size=population_size, generations=generations
        )

        records.append(
            RunRecord(
                problem=problem_label,
                instance_id=problem.name,
                population_size=population_size,
                generations=generations,
                operator_id=operator.operator_id,
                seed=seed,
                best_quality=result.best_quality,
            )
        )

    return OperatorResult(
        problem=problem_label,
        instance_id=problem.name,
        population_size=population_size,
        generations=generations,
        operator_id=operator.operator_id,
        records=tuple(records),
    )


def derive_oracle_rows(
    operator_results: list[OperatorResult], representation: Representation
) -> list[OracleRow]:
    """Derive the full- and reduced-pool reference rows from the cell's operator results.

    Builds the ``{operator_id: median}`` table and applies :func:`derive_oracle` to the
    full and the reduced membership of ``representation``, returning the two
    :class:`OracleRow` rows (full first, then reduced) keyed by the shared cell config.
    Assumes ``operator_results`` covers the full pool.
    """
    oracle_rows: list[OracleRow] = []
    medians = {r.operator_id: float(r.median) for r in operator_results}

    for pool_variant in PoolVariant:
        oracle = derive_oracle(medians, pool_ids(representation, pool_variant))

        oracle_rows.append(
            OracleRow(
                problem=operator_results[0].problem,
                instance_id=operator_results[0].instance_id,
                population_size=operator_results[0].population_size,
                generations=operator_results[0].generations,
                pool_variant=pool_variant,
                oracle=oracle,
            )
        )

    return oracle_rows


def evaluate_cell(
    family: FamilyDescriptor,
    spec: tuple[Any, ...],
    *,
    population_size: int,
    generations: int,
    seeds: list[int],
) -> tuple[list[OperatorResult], list[OracleRow]]:
    """Measure the full pool of ``family`` for one ``spec`` and one (N, G) configuration.

    Builds the problem and its full operator pool (scaled by the spec's ``pool_bounds``),
    runs every operator over ``seeds``, then derives the reference rows for both pool
    variants. Returns the per-operator results and the two reference rows.
    """
    problem = family.build_problem(spec)
    bounds = family.pool_bounds(spec)
    operator_pool = build_pool(family.representation, PoolVariant.FULL, real_bounds=bounds)

    operator_results: list[OperatorResult] = []
    for operator in operator_pool:
        operator_result = evaluate_operator(
            family.problem,
            problem,
            operator,
            population_size=population_size,
            generations=generations,
            seeds=seeds,
        )
        operator_results.append(operator_result)
    oracle_rows = derive_oracle_rows(operator_results, family.representation)

    return operator_results, oracle_rows


def evaluate_sweep(
    seeds: list[int], *, families: tuple[FamilyDescriptor, ...] = FAMILIES
) -> tuple[list[OperatorResult], list[OracleRow]]:
    """Evaluate every grid cell of ``families`` over ``seeds``, reporting progress on stderr.

    Iterates ``families x specs x POPULATION_SIZES x GENERATION_BUDGETS``, accumulating the
    per-operator results and reference rows across all cells. The raw per-run records stay
    recoverable from each :class:`OperatorResult`. Returns the two flat lists.
    """
    operator_results: list[OperatorResult] = []
    oracle_rows: list[OracleRow] = []

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
                new_operator_results, new_oracle_rows = evaluate_cell(
                    family,
                    spec,
                    population_size=population_size,
                    generations=generations,
                    seeds=seeds,
                )

                operator_results.extend(new_operator_results)
                oracle_rows.extend(new_oracle_rows)

                cell_index += 1
                instance_id = new_operator_results[0].instance_id
                o_star = ";".join(new_oracle_rows[0].oracle.o_star)
                print(
                    f"[{cell_index}/{total_cells}] {family.problem}/{instance_id} "
                    f"N={population_size} G={generations} -> o*(full)={o_star}",
                    file=sys.stderr,
                )

    return operator_results, oracle_rows


_RAW_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "operator_id",
    "seed",
    "best_quality",
]

_OPERATORS_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "operator_id",
    "n_seeds",
    "median_quality",
    "mean_quality",
    "std_quality",
    "min_quality",
    "max_quality",
]

_ORACLE_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "pool_variant",
    "o_star",
    "o_star_count",
    "o_star_median",
]


def format_table(results: list[OperatorResult]) -> str:
    """Render a pipe-separated stdout table: a header plus one row per operator."""
    header = (
        "Problem",
        "Instance",
        "Population Size",
        "Generations",
        "Operator",
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
            result.operator_id,
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
                    "operator_id": record.operator_id,
                    "seed": record.seed,
                    "best_quality": record.best_quality,
                }
            )


def write_operators_csv(path: Path, results: list[OperatorResult]) -> None:
    """Write one aggregated row per operator (the versioned per-operator statistics)."""
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_OPERATORS_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "problem": result.problem,
                    "instance_id": result.instance_id,
                    "population_size": result.population_size,
                    "generations": result.generations,
                    "operator_id": result.operator_id,
                    "n_seeds": len(result.records),
                    "median_quality": result.median,
                    "mean_quality": result.mean,
                    "std_quality": result.std,
                    "min_quality": result.minimum,
                    "max_quality": result.maximum,
                }
            )


def write_oracle_csv(path: Path, rows: list[OracleRow]) -> None:
    """Write the reference-set rows: the reference operators as a ";"-joined list.

    The only pool-dependent artifact. Each row carries its pool variant and, from its
    ``OperatorOracle``, the reference operator set, its size and the shared median.
    """
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_ORACLE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "problem": row.problem,
                    "instance_id": row.instance_id,
                    "population_size": row.population_size,
                    "generations": row.generations,
                    "pool_variant": row.pool_variant.value,
                    "o_star": ";".join(row.oracle.o_star),
                    "o_star_count": row.oracle.o_star_count,
                    "o_star_median": row.oracle.o_star_median,
                }
            )


def main() -> None:
    """Run the sweep, print the table and optionally write the raw and aggregated CSVs."""
    argument_parser = argparse.ArgumentParser(
        description="Measure each operator in isolation and report the results."
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
        "--operators-csv",
        type=str,
        default=None,
        required=False,
        help="Path to output CSV file for per-operator statistics.",
    )
    argument_parser.add_argument(
        "--oracle-csv",
        type=str,
        default=None,
        required=False,
        help="Path to output CSV file for the per-pool reference operators.",
    )

    args = argument_parser.parse_args()
    seeds = load_repetition_seeds()
    if args.seeds is not None:
        seeds = seeds[: args.seeds]

    families = FAMILIES
    if args.family is not None:
        families = tuple(family for family in FAMILIES if family.problem == args.family)

    operator_results, oracle_rows = evaluate_sweep(seeds, families=families)
    print(format_table(operator_results))

    if args.csv:
        records = [record for result in operator_results for record in result.records]
        write_csv(Path(args.csv), records)
        print(f"Wrote raw runs: {args.csv}", file=sys.stderr)
    if args.operators_csv:
        write_operators_csv(Path(args.operators_csv), operator_results)
        print(f"Wrote per-operator stats: {args.operators_csv}", file=sys.stderr)
    if args.oracle_csv:
        write_oracle_csv(Path(args.oracle_csv), oracle_rows)
        print(f"Wrote reference operators: {args.oracle_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
