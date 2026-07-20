"""Contract tests for the CGA-on-continuous reporting script's pure functions.

These pin the composable, side-effect-free core of
:mod:`experiments.baselines.cga_continuous_report`: the per-run and per-cell result
records with their aggregates (median, mean, sample std, min, max), the stdout table
formatter and the CSV writers. They never launch the GA sweep -- that is the script's
expensive, offline job, and the determinism of ``run`` from a fixed seed is already
covered by the CGA-on-continuous smoke test. Records are built by hand here, so the
aggregates and serialization are pinned independently of any real run.

Unlike the TSP and knapsack reports there is no gap-to-optimum metric: every benchmark
function has optimum value 0, so a relative gap ``(optimum - best) / optimum`` divides
by zero. Since ``f(x) >= 0`` everywhere and the optimum is 0, the raw ``best_objective``
(the minimal ``f`` found) is already the absolute error to the optimum, so the best
fitness is reported directly and aggregated as-is. Each record carries the ``function``
name and the ``dimension`` as first-class fields (the two axes of the configuration
grid), so both are part of both CSV schemas rather than being recovered from an id.

The module under test is not implemented yet: this file is the executable
specification. Expected public names: ``FUNCTIONS``, ``DIMENSIONS``,
``POPULATION_SIZES``, ``GENERATION_BUDGETS``, ``RunRecord``, ``CellResult``,
``evaluate_cell``, ``evaluate_sweep``, ``format_table``, ``write_csv``,
``write_aggregated_csv``, ``run_cga``, ``main``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.baselines.cga_continuous_report import (
    CellResult,
    RunRecord,
    format_table,
    write_aggregated_csv,
    write_csv,
)

# Exact header the published CSV must expose, in order (raw per-run rows).
_CSV_COLUMNS = [
    "function",
    "dimension",
    "population_size",
    "generations",
    "seed",
    "best_objective",
]

# Exact header the aggregated CSV must expose, in order (one row per cell).
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


def _record(
    *,
    function: str = "rastrigin",
    dimension: int = 10,
    population_size: int = 50,
    generations: int = 50,
    seed: int = 0,
    best_objective: float = 12.5,
) -> RunRecord:
    """A hand-built per-run record; fields default to a plausible rastrigin d=10 run."""
    return RunRecord(
        function=function,
        dimension=dimension,
        population_size=population_size,
        generations=generations,
        seed=seed,
        best_objective=best_objective,
    )


def _cell(function: str, fitnesses: list[float], *, dimension: int = 10) -> CellResult:
    """A cell aggregating one record per best fitness in ``fitnesses``."""
    records = tuple(
        _record(function=function, dimension=dimension, seed=i, best_objective=f)
        for i, f in enumerate(fitnesses)
    )
    return CellResult(
        function=function,
        dimension=dimension,
        population_size=50,
        generations=50,
        records=records,
    )


# --- CellResult aggregates -----------------------------------------------------


def test_cell_exposes_its_fitnesses_in_record_order() -> None:
    cell = _cell("rastrigin", [0.1, 0.2, 0.3])
    assert cell.fitnesses == (0.1, 0.2, 0.3)


def test_cell_reports_central_tendency() -> None:
    cell = _cell("rastrigin", [0.1, 0.2, 0.3])
    assert cell.mean == pytest.approx(0.2)
    assert cell.median == pytest.approx(0.2)


def test_cell_reports_the_spread() -> None:
    cell = _cell("rastrigin", [0.1, 0.2, 0.3])
    assert cell.minimum == pytest.approx(0.1)
    assert cell.maximum == pytest.approx(0.3)
    # Sample standard deviation (ddof=1), the convention for error bars over the
    # 30 repetitions: var = (0.01 + 0 + 0.01) / (3 - 1) = 0.01, std = 0.1.
    assert cell.std == pytest.approx(0.1)


def test_std_of_a_single_run_is_zero() -> None:
    # A single seed (e.g. a --seeds 1 quick run) has no sample spread; the metric
    # must degrade to 0.0 rather than raising.
    assert _cell("rastrigin", [0.42]).std == 0.0


def test_cell_carries_its_configuration() -> None:
    cell = _cell("sphere", [0.3, 0.4], dimension=5)
    assert cell.function == "sphere"
    assert cell.dimension == 5
    assert cell.population_size == 50
    assert cell.generations == 50
    assert len(cell.records) == 2


# --- format_table --------------------------------------------------------------


def test_table_has_a_header_and_one_row_per_cell() -> None:
    cells = [_cell("sphere", [0.1, 0.2]), _cell("rastrigin", [0.3, 0.4])]
    lines = format_table(cells).splitlines()
    assert len(lines) >= len(cells) + 1  # at least a header plus one row per cell


def test_table_names_every_function() -> None:
    cells = [_cell("sphere", [0.1]), _cell("rosenbrock", [0.3])]
    table = format_table(cells)
    for cell in cells:
        assert cell.function in table


def test_table_surfaces_the_dimension() -> None:
    # Dimension is one of the two grid axes, so the stdout view must expose it. The
    # cell is built so "10" appears only as the dimension (not inside 50 or a stat).
    table = format_table([_cell("rastrigin", [0.3], dimension=10)])
    assert "10" in table


# --- write_csv -----------------------------------------------------------------


def test_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write_csv(path, [_record()])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _CSV_COLUMNS


def test_csv_writes_one_row_per_record(tmp_path: Path) -> None:
    records = [_record(seed=1, best_objective=0.5), _record(seed=2, best_objective=0.25)]
    path = tmp_path / "out.csv"
    write_csv(path, records)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["seed"] for row in rows] == ["1", "2"]


def test_csv_preserves_raw_row_fields(tmp_path: Path) -> None:
    # The published rows feed plots and thesis tables, so best_objective stays a raw
    # value here and the function name and dimension ride along verbatim.
    path = tmp_path / "out.csv"
    write_csv(path, [_record(function="sphere", dimension=5, best_objective=3.75)])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert float(row["best_objective"]) == pytest.approx(3.75)
    assert row["function"] == "sphere"
    assert int(row["dimension"]) == 5


# --- write_aggregated_csv ------------------------------------------------------


def test_aggregated_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "agg.csv"
    write_aggregated_csv(path, [_cell("rastrigin", [0.1, 0.2, 0.3])])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _AGG_CSV_COLUMNS


def test_aggregated_csv_writes_one_row_per_cell(tmp_path: Path) -> None:
    cells = [_cell("sphere", [0.1, 0.2]), _cell("rastrigin", [0.3, 0.4])]
    path = tmp_path / "agg.csv"
    write_aggregated_csv(path, cells)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["function"] for row in rows] == ["sphere", "rastrigin"]


def test_aggregated_csv_row_matches_the_cell_aggregates(tmp_path: Path) -> None:
    cell = _cell("rastrigin", [0.1, 0.2, 0.3])
    path = tmp_path / "agg.csv"
    write_aggregated_csv(path, [cell])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert int(row["n_seeds"]) == 3
    assert row["function"] == "rastrigin"
    assert int(row["dimension"]) == 10
    assert float(row["mean_fitness"]) == pytest.approx(cell.mean)
    assert float(row["std_fitness"]) == pytest.approx(cell.std)
    assert float(row["median_fitness"]) == pytest.approx(cell.median)
    assert float(row["min_fitness"]) == pytest.approx(cell.minimum)
    assert float(row["max_fitness"]) == pytest.approx(cell.maximum)


# --- public API surface --------------------------------------------------------


def test_sweep_constants_match_the_methodology() -> None:
    import experiments.baselines.cga_continuous_report as report

    # FUNCTIONS holds the three benchmark specs in this order; pin them by name so the
    # grid rows read sphere/rastrigin/rosenbrock regardless of object identity.
    assert tuple(function.name for function in report.FUNCTIONS) == (
        "sphere",
        "rastrigin",
        "rosenbrock",
    )
    assert report.DIMENSIONS == (5, 10)
    assert report.POPULATION_SIZES == (20, 50)
    assert report.GENERATION_BUDGETS == (20, 30, 50)


def test_exposes_the_sweep_and_entry_point() -> None:
    import experiments.baselines.cga_continuous_report as report

    for name in ("run_cga", "evaluate_cell", "evaluate_sweep", "main"):
        assert hasattr(report, name)
