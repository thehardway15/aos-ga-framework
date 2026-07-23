"""Contract tests for the random-selection baseline reporting script's pure functions.

These pin the composable, side-effect-free core of
:mod:`experiments.baselines.random_report`: the per-run and per-pool-variant result
records with their aggregates (median, mean, sample std, min, max), the stdout table
formatter and the two CSV writers. They never launch the GA sweep -- that is the
script's expensive, offline job, and the determinism of ``run`` and
``RandomOperatorStep`` from a fixed seed is already covered by
``tests/test_variation_random_operator.py``. Records are built by hand here, so the
aggregates and the serialization are pinned independently of any real run.

The metric is the unified quality ``best_quality`` (more is better; for a minimization
problem ``g = -f``), so no gap-to-optimum is involved. Every family (TSP, knapsack,
continuous) shares one schema: ``problem`` names the family and ``instance_id`` is the
problem's own ``name`` (``"eil22"``, ``"n20_strongly"``, ``"sphere_d5"``), folding the
continuous function and dimension into a single id.

Unlike the single-operator reference (``fbo_report``), Random measures the whole pool as
one configuration, so the unit of a sweep cell is a **pool variant** (``full`` /
``reduced``), not an operator, and there is **no oracle**: the run emits two artifacts,
the raw per-run rows and the per-pool-variant statistics, both keyed by the pool variant.

The module under test is not implemented yet: this file is the executable specification.
Expected public names: ``FAMILIES``, ``POPULATION_SIZES``, ``GENERATION_BUDGETS``,
``RunRecord``, ``PoolResult``, ``run_random``, ``evaluate_pool_variant``,
``evaluate_cell``, ``evaluate_sweep``, ``format_table``, ``write_csv``,
``write_baseline_csv``, ``main``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aos_ga.core.representation import Representation
from experiments.baselines.random_report import (
    PoolResult,
    RunRecord,
    format_table,
    write_baseline_csv,
    write_csv,
)
from experiments.configs.pools import PoolVariant

# Exact header the raw per-run CSV must expose, in order (keyed by pool variant, no operator).
_RAW_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "pool_variant",
    "seed",
    "best_quality",
]

# Exact header the per-pool-variant baseline CSV must expose, in order.
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
            seed=i,
            best_quality=q,
        )
        for i, q in enumerate(qualities)
    )
    return PoolResult(
        problem=problem,
        instance_id=instance_id,
        population_size=population_size,
        generations=generations,
        pool_variant=pool_variant,
        records=records,
    )


# --- PoolResult aggregates -----------------------------------------------------


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
    # A single seed (e.g. a --seeds 1 quick run) has no sample spread; the metric must
    # degrade to 0.0 rather than raising.
    assert _pool_result(PoolVariant.FULL, [0.42]).std == 0.0


def test_pool_result_carries_its_configuration() -> None:
    result = _pool_result(
        PoolVariant.REDUCED, [0.3, 0.4], problem="continuous", instance_id="sphere_d5"
    )
    assert result.problem == "continuous"
    assert result.instance_id == "sphere_d5"
    assert result.pool_variant is PoolVariant.REDUCED
    assert result.population_size == 50
    assert result.generations == 50
    assert len(result.records) == 2


# --- format_table --------------------------------------------------------------


def test_table_has_a_header_and_one_row_per_pool_variant() -> None:
    results = [_pool_result(PoolVariant.FULL, [0.1, 0.2]), _pool_result(PoolVariant.REDUCED, [0.3])]
    lines = format_table(results).splitlines()
    assert len(lines) >= len(results) + 1  # at least a header plus one row per pool variant


def test_table_names_every_pool_variant() -> None:
    results = [_pool_result(PoolVariant.FULL, [0.1]), _pool_result(PoolVariant.REDUCED, [0.3])]
    table = format_table(results)
    for result in results:
        assert result.pool_variant.value in table


def test_table_surfaces_the_instance_id() -> None:
    table = format_table([_pool_result(PoolVariant.FULL, [0.3], instance_id="berlin52")])
    assert "berlin52" in table


# --- write_csv (raw per-run) ---------------------------------------------------


def test_raw_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "random.csv"
    write_csv(path, [_record()])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _RAW_CSV_COLUMNS


def test_raw_csv_writes_one_row_per_record(tmp_path: Path) -> None:
    records = [_record(seed=1, best_quality=-500.0), _record(seed=2, best_quality=-450.0)]
    path = tmp_path / "random.csv"
    write_csv(path, records)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["seed"] for row in rows] == ["1", "2"]


def test_raw_csv_preserves_raw_row_fields(tmp_path: Path) -> None:
    # The published rows feed re-aggregation without re-running the GA, so best_quality
    # stays a raw value here and the family, instance and pool variant ride along verbatim.
    path = tmp_path / "random.csv"
    record = _record(
        problem="continuous",
        instance_id="sphere_d5",
        pool_variant=PoolVariant.REDUCED,
        best_quality=-3.75,
    )
    write_csv(path, [record])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert float(row["best_quality"]) == pytest.approx(-3.75)
    assert row["problem"] == "continuous"
    assert row["instance_id"] == "sphere_d5"
    assert row["pool_variant"] == "reduced"


# --- write_baseline_csv (per-pool-variant aggregates) --------------------------


def test_baseline_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "random_baseline.csv"
    write_baseline_csv(path, [_pool_result(PoolVariant.FULL, [0.1, 0.2, 0.3])])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _BASELINE_CSV_COLUMNS


def test_baseline_csv_writes_one_row_per_pool_variant(tmp_path: Path) -> None:
    results = [
        _pool_result(PoolVariant.FULL, [0.1, 0.2]),
        _pool_result(PoolVariant.REDUCED, [0.3, 0.4]),
    ]
    path = tmp_path / "random_baseline.csv"
    write_baseline_csv(path, results)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # The pool variant is serialized as its value, full before reduced.
    assert [row["pool_variant"] for row in rows] == ["full", "reduced"]


def test_baseline_csv_row_matches_the_aggregates(tmp_path: Path) -> None:
    result = _pool_result(PoolVariant.FULL, [10.0, 20.0, 30.0])
    path = tmp_path / "random_baseline.csv"
    write_baseline_csv(path, [result])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert int(row["n_seeds"]) == 3
    assert row["pool_variant"] == "full"
    assert float(row["median_quality"]) == pytest.approx(result.median)
    assert float(row["mean_quality"]) == pytest.approx(result.mean)
    assert float(row["std_quality"]) == pytest.approx(result.std)
    assert float(row["min_quality"]) == pytest.approx(result.minimum)
    assert float(row["max_quality"]) == pytest.approx(result.maximum)


# --- families and public API surface -------------------------------------------


def test_sweep_constants_match_the_methodology() -> None:
    import experiments.baselines.random_report as report

    assert report.POPULATION_SIZES == (20, 50)
    assert report.GENERATION_BUDGETS == (20, 30, 50)


def test_families_cover_the_three_problem_classes() -> None:
    import experiments.baselines.random_report as report

    by_name = {family.problem: family for family in report.FAMILIES}
    assert set(by_name) == {"tsp", "knapsack", "continuous"}
    assert by_name["tsp"].representation is Representation.PERMUTATION
    assert by_name["knapsack"].representation is Representation.BINARY
    assert by_name["continuous"].representation is Representation.REAL


def test_exposes_the_sweep_and_entry_point() -> None:
    import experiments.baselines.random_report as report

    for name in (
        "run_random",
        "evaluate_pool_variant",
        "evaluate_cell",
        "evaluate_sweep",
        "main",
    ):
        assert hasattr(report, name)
