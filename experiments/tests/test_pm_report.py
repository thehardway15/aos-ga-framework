"""Contract tests for the Probability Matching sweep script's pure functions.

These pin the composable, side-effect-free core of
:mod:`experiments.strategies.pm_report`: the per-run and per-pool-variant result
records with their aggregates (median, mean, sample std, min, max), the stdout table
formatter and the two CSV writers. The schema deliberately matches the random-selection
baseline row for row, so the adaptive results and their floor can be joined on
``(problem, instance_id, population_size, generations, pool_variant)`` without any
reshaping in the analysis phase.

Unlike the baselines, this sweep runs a *stateful* variation step: the strategy's
quality estimates are run state. One section below therefore does launch a small GA --
the cheapest possible one -- to pin that each run gets a fresh strategy. Leaking that
state between runs would not raise anything; it would quietly make later repetitions
start from what the earlier ones learned, inflating the results in a way no aggregate
could reveal.

The module under test is not implemented yet: this file is the executable
specification. Expected public names: ``FAMILIES``, ``POPULATION_SIZES``,
``GENERATION_BUDGETS``, ``RunRecord``, ``PoolResult``, ``run_pm``,
``evaluate_pool_variant``, ``evaluate_cell``, ``evaluate_sweep``, ``format_table``,
``write_csv``, ``write_aggregate_csv``, ``main``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aos_ga.core.representation import Representation
from experiments.configs.pools import PoolVariant, build_pool
from experiments.datasets.tsplib import load_instance
from experiments.problems.tsp import TSPProblem
from experiments.strategies.pm_report import (
    PoolResult,
    RunRecord,
    evaluate_pool_variant,
    format_table,
    run_pm,
    write_aggregate_csv,
    write_csv,
)

# Exact header the raw per-run CSV must expose, in order (identical to the Random
# baseline's, so the two artifacts stack).
_RAW_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "pool_variant",
    "seed",
    "best_quality",
]

# Exact header the aggregated CSV must expose, in order.
_AGGREGATE_CSV_COLUMNS = [
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

# A deliberately tiny configuration for the two tests that do run the GA: enough
# reproduction events for the strategy to accumulate state, cheap enough to run inline.
_TINY_INSTANCE = "eil22"
_TINY_POPULATION = 10
_TINY_GENERATIONS = 3


def _record(
    *,
    problem: str = "tsp",
    instance_id: str = "eil22",
    population_size: int = 50,
    generations: int = 50,
    pool_variant: PoolVariant = PoolVariant.FULL,
    seed: int = 0,
    best_quality: float = -426.0,
) -> RunRecord:
    """A hand-built per-run record; fields default to a plausible TSP full-pool run.

    ``best_quality`` defaults negative because TSP is a minimization problem whose
    unified quality is ``-tour_length``.
    """
    return RunRecord(
        problem=problem,
        instance_id=instance_id,
        population_size=population_size,
        generations=generations,
        pool_variant=pool_variant,
        seed=seed,
        best_quality=best_quality,
    )


def _pool_result(
    pool_variant: PoolVariant,
    qualities: list[float],
    *,
    problem: str = "tsp",
    instance_id: str = "eil22",
    population_size: int = 50,
    generations: int = 50,
) -> PoolResult:
    """A pool variant's cell result aggregating one record per quality in ``qualities``."""
    records = tuple(
        _record(
            problem=problem,
            instance_id=instance_id,
            population_size=population_size,
            generations=generations,
            pool_variant=pool_variant,
            seed=index,
            best_quality=quality,
        )
        for index, quality in enumerate(qualities)
    )
    return PoolResult(
        problem=problem,
        instance_id=instance_id,
        population_size=population_size,
        generations=generations,
        pool_variant=pool_variant,
        records=records,
    )


@pytest.fixture(scope="module")
def tiny_problem() -> TSPProblem:
    """The smallest TSP instance, loaded once for the two runs below."""
    return TSPProblem(load_instance(_TINY_INSTANCE))


# --- PoolResult aggregates ---------------------------------------------------------


def test_pool_result_exposes_qualities_in_record_order() -> None:
    result = _pool_result(PoolVariant.FULL, [0.1, 0.2, 0.3])
    assert result.qualities == (0.1, 0.2, 0.3)


def test_pool_result_reports_central_tendency() -> None:
    result = _pool_result(PoolVariant.FULL, [0.1, 0.2, 0.3])
    assert result.mean == pytest.approx(0.2)
    assert result.median == pytest.approx(0.2)


def test_pool_result_reports_the_spread() -> None:
    result = _pool_result(PoolVariant.FULL, [0.1, 0.2, 0.3])
    assert result.minimum == pytest.approx(0.1)
    assert result.maximum == pytest.approx(0.3)
    # Sample standard deviation (ddof=1), the convention for error bars over the 30
    # repetitions: var = (0.01 + 0 + 0.01) / (3 - 1) = 0.01, std = 0.1.
    assert result.std == pytest.approx(0.1)


def test_std_of_a_single_run_is_zero() -> None:
    assert _pool_result(PoolVariant.FULL, [0.42]).std == 0.0


def test_pool_result_carries_its_configuration() -> None:
    result = _pool_result(
        PoolVariant.REDUCED, [0.3, 0.4], problem="continuous", instance_id="sphere_d5"
    )
    assert result.problem == "continuous"
    assert result.instance_id == "sphere_d5"
    assert result.pool_variant is PoolVariant.REDUCED
    assert len(result.records) == 2


# --- format_table ------------------------------------------------------------------


def test_table_has_a_header_and_one_row_per_pool_variant() -> None:
    results = [_pool_result(PoolVariant.FULL, [0.1, 0.2]), _pool_result(PoolVariant.REDUCED, [0.3])]
    lines = format_table(results).splitlines()
    assert len(lines) >= len(results) + 1


def test_table_names_every_pool_variant() -> None:
    results = [_pool_result(PoolVariant.FULL, [0.1]), _pool_result(PoolVariant.REDUCED, [0.3])]
    table = format_table(results)
    for result in results:
        assert result.pool_variant.value in table


def test_table_surfaces_the_instance_id() -> None:
    table = format_table([_pool_result(PoolVariant.FULL, [0.3], instance_id="berlin52")])
    assert "berlin52" in table


# --- write_csv (raw per-run) -------------------------------------------------------


def test_raw_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "pm.csv"
    write_csv(path, [_record()])
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == _RAW_CSV_COLUMNS


def test_raw_csv_writes_one_row_per_record(tmp_path: Path) -> None:
    records = [_record(seed=1, best_quality=-500.0), _record(seed=2, best_quality=-450.0)]
    path = tmp_path / "pm.csv"
    write_csv(path, records)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["seed"] for row in rows] == ["1", "2"]


def test_raw_csv_preserves_raw_row_fields(tmp_path: Path) -> None:
    # The published rows feed re-aggregation without re-running the GA, so best_quality
    # stays a raw value and the family, instance and pool variant ride along verbatim.
    path = tmp_path / "pm.csv"
    record = _record(
        problem="continuous",
        instance_id="sphere_d5",
        pool_variant=PoolVariant.REDUCED,
        best_quality=-3.75,
    )
    write_csv(path, [record])
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["best_quality"]) == pytest.approx(-3.75)
    assert row["problem"] == "continuous"
    assert row["instance_id"] == "sphere_d5"
    assert row["pool_variant"] == "reduced"


# --- write_aggregate_csv (per-pool-variant aggregates) -----------------------------


def test_aggregate_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "aos_pm.csv"
    write_aggregate_csv(path, [_pool_result(PoolVariant.FULL, [0.1, 0.2, 0.3])])
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == _AGGREGATE_CSV_COLUMNS


def test_aggregate_csv_writes_one_row_per_pool_variant(tmp_path: Path) -> None:
    results = [
        _pool_result(PoolVariant.FULL, [0.1, 0.2]),
        _pool_result(PoolVariant.REDUCED, [0.3, 0.4]),
    ]
    path = tmp_path / "aos_pm.csv"
    write_aggregate_csv(path, results)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["pool_variant"] for row in rows] == ["full", "reduced"]


def test_aggregate_csv_row_matches_the_aggregates(tmp_path: Path) -> None:
    result = _pool_result(PoolVariant.FULL, [10.0, 20.0, 30.0])
    path = tmp_path / "aos_pm.csv"
    write_aggregate_csv(path, [result])
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["n_seeds"]) == 3
    assert row["pool_variant"] == "full"
    assert float(row["median_quality"]) == pytest.approx(result.median)
    assert float(row["mean_quality"]) == pytest.approx(result.mean)
    assert float(row["std_quality"]) == pytest.approx(result.std)
    assert float(row["min_quality"]) == pytest.approx(result.minimum)
    assert float(row["max_quality"]) == pytest.approx(result.maximum)


def test_the_two_artifacts_share_the_baseline_key_columns() -> None:
    # The adaptive results are only comparable with the Random floor and the fixed-best
    # ceiling if the configuration key is spelled identically in every artifact.
    key = ["problem", "instance_id", "population_size", "generations", "pool_variant"]
    assert _RAW_CSV_COLUMNS[: len(key)] == key
    assert _AGGREGATE_CSV_COLUMNS[: len(key)] == key


# --- strategy state is per run, not per sweep --------------------------------------


def test_repeating_a_seed_reproduces_the_run(tiny_problem: TSPProblem) -> None:
    # The decisive test for this sweep: the same seed twice in a row must give the same
    # result. A strategy shared across runs would carry its estimates into the second
    # run and silently change it -- no error, just better-looking numbers.
    pool = build_pool(Representation.PERMUTATION, PoolVariant.FULL)
    result = evaluate_pool_variant(
        "tsp",
        tiny_problem,
        pool,
        PoolVariant.FULL,
        population_size=_TINY_POPULATION,
        generations=_TINY_GENERATIONS,
        seeds=[7, 7],
    )
    first, second = result.qualities
    assert first == second


def test_run_pm_is_deterministic_from_its_seed(tiny_problem: TSPProblem) -> None:
    # Same guarantee one level down, without the aggregation: a single run is a pure
    # function of (problem, pool, seed, N, G).
    pool = build_pool(Representation.PERMUTATION, PoolVariant.FULL)

    def once() -> float:
        return run_pm(
            tiny_problem,
            pool,
            5,
            population_size=_TINY_POPULATION,
            generations=_TINY_GENERATIONS,
        ).best_quality

    assert once() == once()


# --- families and public API surface -----------------------------------------------


def test_sweep_constants_match_the_methodology() -> None:
    import experiments.strategies.pm_report as report

    assert report.POPULATION_SIZES == (20, 50)
    assert report.GENERATION_BUDGETS == (20, 30, 50)


def test_families_cover_the_three_problem_classes() -> None:
    import experiments.strategies.pm_report as report

    by_name = {family.problem: family for family in report.FAMILIES}
    assert set(by_name) == {"tsp", "knapsack", "continuous"}
    assert by_name["tsp"].representation is Representation.PERMUTATION
    assert by_name["knapsack"].representation is Representation.BINARY
    assert by_name["continuous"].representation is Representation.REAL


def test_exposes_the_sweep_and_entry_point() -> None:
    import experiments.strategies.pm_report as report

    for name in (
        "run_pm",
        "evaluate_pool_variant",
        "evaluate_cell",
        "evaluate_sweep",
        "main",
    ):
        assert hasattr(report, name)
