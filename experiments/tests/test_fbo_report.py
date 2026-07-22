"""Contract tests for the single-operator reference reporting script's pure functions.

These pin the composable, side-effect-free core of
:mod:`experiments.baselines.fbo_report`: the per-run and per-operator result records
with their aggregates (median, mean, sample std, min, max), the derivation of each
pool's reference operator set from the per-operator medians, the stdout table
formatter and the three CSV writers. They never launch the GA sweep -- that is the
script's expensive, offline job, and the determinism of ``run`` and
``SingleOperatorStep`` from a fixed seed is already covered elsewhere. Records are
built by hand here, so the aggregates, the reference-set derivation and the
serialization are pinned independently of any real run.

The metric is the unified quality ``best_quality`` (more is better; for a minimization
problem ``g = -f``), so no gap-to-optimum is involved. Every family (TSP, knapsack,
continuous) shares one schema: ``problem`` names the family and ``instance_id`` is the
problem's own ``name`` (``"eil22"``, ``"n20_strongly"``, ``"sphere_d5"``), folding the
continuous function and dimension into a single id.

The reference operator set (``o*``) for a pool is the set of operators tying for the
maximum median; because each operator is measured on its own, a single full-pool
measurement projects onto both the full and the reduced pool via
:func:`experiments.baselines.fbo_oracle.derive_oracle`, so the reduced-pool reference
may differ from the full-pool one when the global maximizer sits outside the reduced
pool -- the case pinned below.

The module under test is not implemented yet: this file is the executable
specification. Expected public names: ``FAMILIES``, ``FamilyDescriptor``,
``POPULATION_SIZES``, ``GENERATION_BUDGETS``, ``RunRecord``, ``OperatorResult``,
``OracleRow``, ``derive_oracle_rows``, ``evaluate_operator``, ``evaluate_cell``,
``evaluate_sweep``, ``run_single_operator``, ``format_table``, ``write_csv``,
``write_operators_csv``, ``write_oracle_csv``, ``main``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aos_ga.core.representation import Representation
from experiments.baselines.fbo_report import (
    OperatorResult,
    RunRecord,
    derive_oracle_rows,
    format_table,
    write_csv,
    write_operators_csv,
    write_oracle_csv,
)
from experiments.configs.pools import PoolVariant
from experiments.problems.continuous import ROSENBROCK, SPHERE

# Exact header the raw per-run CSV must expose, in order (pool-agnostic).
_RAW_CSV_COLUMNS = [
    "problem",
    "instance_id",
    "population_size",
    "generations",
    "operator_id",
    "seed",
    "best_quality",
]

# Exact header the per-operator CSV must expose, in order (pool-agnostic).
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

# Exact header the reference-set CSV must expose, in order (the only pool-dependent file).
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


def _record(
    *,
    problem: str = "tsp",
    instance_id: str = "eil22",
    population_size: int = 50,
    generations: int = 50,
    operator_id: str = "ox",
    seed: int = 0,
    best_quality: float = -426.0,
) -> RunRecord:
    """A hand-built per-run record; fields default to a plausible TSP ``ox`` run.

    ``best_quality`` defaults negative because TSP is a minimization problem whose
    unified quality is ``-tour_length``.
    """
    return RunRecord(
        problem=problem,
        instance_id=instance_id,
        population_size=population_size,
        generations=generations,
        operator_id=operator_id,
        seed=seed,
        best_quality=best_quality,
    )


def _op_result(
    operator_id: str,
    qualities: list[float],
    *,
    problem: str = "tsp",
    instance_id: str = "eil22",
    population_size: int = 50,
    generations: int = 50,
) -> OperatorResult:
    """An operator's cell result aggregating one record per quality in ``qualities``."""
    records = tuple(
        _record(
            problem=problem,
            instance_id=instance_id,
            population_size=population_size,
            generations=generations,
            operator_id=operator_id,
            seed=i,
            best_quality=q,
        )
        for i, q in enumerate(qualities)
    )
    return OperatorResult(
        problem=problem,
        instance_id=instance_id,
        population_size=population_size,
        generations=generations,
        operator_id=operator_id,
        records=records,
    )


def _full_permutation_results() -> list[OperatorResult]:
    """The full permutation pool measured so the global maximizer sits outside REDUCED.

    Medians are set (one record each, so the median is that value) so that the full
    pool's best is ``swap`` (median 9), which is not in the reduced pool
    ``(ox, cx, inversion)`` whose best is the tie ``ox``/``cx`` at median 5. This makes
    the reduced-pool reference genuinely differ from the full-pool one.
    """
    medians = {
        "ox": 5.0,
        "pmx": 6.0,
        "cx": 5.0,
        "swap": 9.0,
        "inversion": 4.0,
        "insert": 2.0,
    }
    return [_op_result(op_id, [value]) for op_id, value in medians.items()]


# --- OperatorResult aggregates -------------------------------------------------


def test_operator_result_exposes_qualities_in_record_order() -> None:
    result = _op_result("ox", [0.1, 0.2, 0.3])
    assert result.qualities == (0.1, 0.2, 0.3)


def test_operator_result_reports_central_tendency() -> None:
    result = _op_result("ox", [0.1, 0.2, 0.3])
    assert result.mean == pytest.approx(0.2)
    assert result.median == pytest.approx(0.2)


def test_operator_result_reports_the_spread() -> None:
    result = _op_result("ox", [0.1, 0.2, 0.3])
    assert result.minimum == pytest.approx(0.1)
    assert result.maximum == pytest.approx(0.3)
    # Sample standard deviation (ddof=1), the convention for error bars over the 30
    # repetitions: var = (0.01 + 0 + 0.01) / (3 - 1) = 0.01, std = 0.1.
    assert result.std == pytest.approx(0.1)


def test_std_of_a_single_run_is_zero() -> None:
    # A single seed (e.g. a --seeds 1 quick run) has no sample spread; the metric must
    # degrade to 0.0 rather than raising.
    assert _op_result("ox", [0.42]).std == 0.0


def test_operator_result_carries_its_configuration() -> None:
    result = _op_result("cx", [0.3, 0.4], problem="continuous", instance_id="sphere_d5")
    assert result.problem == "continuous"
    assert result.instance_id == "sphere_d5"
    assert result.operator_id == "cx"
    assert result.population_size == 50
    assert result.generations == 50
    assert len(result.records) == 2


# --- derive_oracle_rows (the pool projection, D9) ------------------------------


def test_derive_oracle_rows_returns_full_then_reduced() -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    assert [row.pool_variant for row in rows] == [PoolVariant.FULL, PoolVariant.REDUCED]


def test_full_oracle_is_the_global_maximizer_set() -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    full = next(row for row in rows if row.pool_variant is PoolVariant.FULL)
    # ``swap`` is the sole global maximizer (median 9) even though it is outside REDUCED.
    assert full.oracle.o_star == ("swap",)
    assert full.oracle.o_star_count == 1
    assert full.oracle.o_star_median == pytest.approx(9.0)


def test_reduced_oracle_projects_onto_the_reduced_pool() -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    reduced = next(row for row in rows if row.pool_variant is PoolVariant.REDUCED)
    # Restricted to (ox, cx, inversion), the best median is the tie ox/cx at 5, kept in
    # membership order -- the reduced reference is a strictly lower ceiling than the full.
    assert reduced.oracle.o_star == ("ox", "cx")
    assert reduced.oracle.o_star_count == 2
    assert reduced.oracle.o_star_median == pytest.approx(5.0)


def test_oracle_median_is_a_measured_operator_median() -> None:
    # Single source of truth: the reference median equals the winning operator's own
    # per-operator median, so the two aggregated CSVs never disagree.
    results = _full_permutation_results()
    rows = derive_oracle_rows(results, Representation.PERMUTATION)
    full = next(row for row in rows if row.pool_variant is PoolVariant.FULL)
    winner = next(r for r in results if r.operator_id == "swap")
    assert full.oracle.o_star_median == winner.median


# --- format_table --------------------------------------------------------------


def test_table_has_a_header_and_one_row_per_operator() -> None:
    results = [_op_result("ox", [0.1, 0.2]), _op_result("cx", [0.3, 0.4])]
    lines = format_table(results).splitlines()
    assert len(lines) >= len(results) + 1  # at least a header plus one row per operator


def test_table_names_every_operator() -> None:
    results = [_op_result("ox", [0.1]), _op_result("swap", [0.3])]
    table = format_table(results)
    for result in results:
        assert result.operator_id in table


def test_table_surfaces_the_instance_id() -> None:
    table = format_table([_op_result("ox", [0.3], instance_id="berlin52")])
    assert "berlin52" in table


# --- write_csv (raw per-run) ---------------------------------------------------


def test_raw_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "fbo.csv"
    write_csv(path, [_record()])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _RAW_CSV_COLUMNS


def test_raw_csv_writes_one_row_per_record(tmp_path: Path) -> None:
    records = [_record(seed=1, best_quality=-500.0), _record(seed=2, best_quality=-450.0)]
    path = tmp_path / "fbo.csv"
    write_csv(path, records)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["seed"] for row in rows] == ["1", "2"]


def test_raw_csv_preserves_raw_row_fields(tmp_path: Path) -> None:
    # The published rows feed re-aggregation without re-running the GA, so best_quality
    # stays a raw value here and the family, instance and operator ride along verbatim.
    path = tmp_path / "fbo.csv"
    record = _record(
        problem="continuous", instance_id="sphere_d5", operator_id="sbx", best_quality=-3.75
    )
    write_csv(path, [record])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert float(row["best_quality"]) == pytest.approx(-3.75)
    assert row["problem"] == "continuous"
    assert row["instance_id"] == "sphere_d5"
    assert row["operator_id"] == "sbx"


# --- write_operators_csv (per-operator aggregates) -----------------------------


def test_operators_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "fbo_operators.csv"
    write_operators_csv(path, [_op_result("ox", [0.1, 0.2, 0.3])])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _OPERATORS_CSV_COLUMNS


def test_operators_csv_writes_one_row_per_operator(tmp_path: Path) -> None:
    results = [_op_result("ox", [0.1, 0.2]), _op_result("cx", [0.3, 0.4])]
    path = tmp_path / "fbo_operators.csv"
    write_operators_csv(path, results)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["operator_id"] for row in rows] == ["ox", "cx"]


def test_operators_csv_row_matches_the_aggregates(tmp_path: Path) -> None:
    result = _op_result("ox", [10.0, 20.0, 30.0])
    path = tmp_path / "fbo_operators.csv"
    write_operators_csv(path, [result])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert int(row["n_seeds"]) == 3
    assert row["operator_id"] == "ox"
    assert float(row["median_quality"]) == pytest.approx(result.median)
    assert float(row["mean_quality"]) == pytest.approx(result.mean)
    assert float(row["std_quality"]) == pytest.approx(result.std)
    assert float(row["min_quality"]) == pytest.approx(result.minimum)
    assert float(row["max_quality"]) == pytest.approx(result.maximum)


# --- write_oracle_csv (reference set, pool-dependent) --------------------------


def test_oracle_csv_has_the_expected_header(tmp_path: Path) -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    path = tmp_path / "fbo_oracle.csv"
    write_oracle_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _ORACLE_CSV_COLUMNS


def test_oracle_csv_writes_full_and_reduced_rows(tmp_path: Path) -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    path = tmp_path / "fbo_oracle.csv"
    write_oracle_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert [row["pool_variant"] for row in written] == ["full", "reduced"]


def test_oracle_csv_serializes_o_star_as_semicolon_join(tmp_path: Path) -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    path = tmp_path / "fbo_oracle.csv"
    write_oracle_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as f:
        written = {row["pool_variant"]: row for row in csv.DictReader(f)}
    # A singleton is the bare id; a tie is a ";"-joined list in membership order.
    assert written["full"]["o_star"] == "swap"
    assert written["reduced"]["o_star"] == "ox;cx"


def test_oracle_csv_row_matches_the_derivation(tmp_path: Path) -> None:
    rows = derive_oracle_rows(_full_permutation_results(), Representation.PERMUTATION)
    path = tmp_path / "fbo_oracle.csv"
    write_oracle_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as f:
        written = {row["pool_variant"]: row for row in csv.DictReader(f)}
    assert int(written["full"]["o_star_count"]) == 1
    assert float(written["full"]["o_star_median"]) == pytest.approx(9.0)
    assert int(written["reduced"]["o_star_count"]) == 2
    assert float(written["reduced"]["o_star_median"]) == pytest.approx(5.0)


# --- family descriptors and public API surface ---------------------------------


def test_sweep_constants_match_the_methodology() -> None:
    import experiments.baselines.fbo_report as report

    assert report.POPULATION_SIZES == (20, 50)
    assert report.GENERATION_BUDGETS == (20, 30, 50)


def test_families_cover_the_three_problem_classes() -> None:
    import experiments.baselines.fbo_report as report

    by_name = {family.problem: family for family in report.FAMILIES}
    assert set(by_name) == {"tsp", "knapsack", "continuous"}
    assert by_name["tsp"].representation is Representation.PERMUTATION
    assert by_name["knapsack"].representation is Representation.BINARY
    assert by_name["continuous"].representation is Representation.REAL
    for family in report.FAMILIES:
        assert family.specs  # every family carries at least one instance spec


def test_continuous_family_bounds_are_per_function() -> None:
    # The key correction to the contract: the pool bounds are a function of the instance,
    # not of the family, because Rosenbrock's domain differs from Sphere's -- and the
    # gaussian/polynomial scaling depends on the domain width.
    import experiments.baselines.fbo_report as report

    continuous = next(family for family in report.FAMILIES if family.problem == "continuous")
    assert continuous.pool_bounds((SPHERE, 5)) == (SPHERE.lower, SPHERE.upper)
    assert continuous.pool_bounds((ROSENBROCK, 5)) == (ROSENBROCK.lower, ROSENBROCK.upper)
    assert continuous.pool_bounds((SPHERE, 5)) != continuous.pool_bounds((ROSENBROCK, 5))


def test_continuous_family_builds_a_named_problem() -> None:
    # instance_id folds the function and dimension into the problem's own name.
    import experiments.baselines.fbo_report as report

    continuous = next(family for family in report.FAMILIES if family.problem == "continuous")
    problem = continuous.build_problem((SPHERE, 5))
    assert problem.name == "sphere_d5"
    assert problem.representation is Representation.REAL


def test_exposes_the_sweep_and_entry_point() -> None:
    import experiments.baselines.fbo_report as report

    for name in (
        "run_single_operator",
        "evaluate_operator",
        "evaluate_cell",
        "evaluate_sweep",
        "main",
    ):
        assert hasattr(report, name)
