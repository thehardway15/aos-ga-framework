"""Contract tests for the CGA-on-TSP reporting script's pure functions.

These pin the composable, side-effect-free core of
:mod:`experiments.baselines.cga_tsp_report`: the gap-to-optimum metric, the
per-run and per-cell result records with their aggregates (median, mean, sample
std, min, max), the stdout table formatter and the raw per-run CSV writer. They
never launch the GA sweep -- that is the script's expensive, offline job, and the
determinism of ``run`` from a fixed seed is already covered by the CGA smoke test.
Records are built by hand here, so the aggregates and serialization are pinned
independently of any real run.

The module under test is not implemented yet: this file is the executable
specification. Expected public names: ``INSTANCES``, ``POPULATION_SIZES``,
``GENERATION_BUDGETS``, ``gap_to_optimum``, ``RunRecord``, ``CellResult``,
``evaluate_cell``, ``evaluate_sweep``, ``format_table``, ``write_csv``,
``run_cga``, ``main``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.baselines.cga_tsp_report import (
    CellResult,
    RunRecord,
    format_table,
    gap_to_optimum,
    write_aggregated_csv,
    write_csv,
)

# Exact header the published CSV must expose, in order (raw per-run rows).
_CSV_COLUMNS = [
    "instance_id",
    "population_size",
    "generations",
    "seed",
    "best_objective",
    "optimum",
    "gap",
]

# Exact header the aggregated CSV must expose, in order (one row per cell).
_AGG_CSV_COLUMNS = [
    "instance_id",
    "population_size",
    "generations",
    "n_seeds",
    "mean_gap",
    "std_gap",
    "median_gap",
    "min_gap",
    "max_gap",
]


def _record(
    *,
    instance_id: str = "eil51",
    population_size: int = 50,
    generations: int = 50,
    seed: int = 0,
    best_objective: float = 639.0,
    optimum: int = 426,
    gap: float = 0.5,
) -> RunRecord:
    """A hand-built per-run record; fields default to a plausible eil51 run."""
    return RunRecord(
        instance_id=instance_id,
        population_size=population_size,
        generations=generations,
        seed=seed,
        best_objective=best_objective,
        optimum=optimum,
        gap=gap,
    )


def _cell(instance_id: str, gaps: list[float]) -> CellResult:
    """A cell aggregating one record per gap in ``gaps``."""
    records = tuple(_record(instance_id=instance_id, seed=i, gap=g) for i, g in enumerate(gaps))
    return CellResult(instance_id=instance_id, population_size=50, generations=50, records=records)


# --- gap_to_optimum ------------------------------------------------------------


def test_gap_is_zero_when_objective_equals_optimum() -> None:
    assert gap_to_optimum(426.0, 426) == 0.0


def test_gap_is_the_relative_excess_over_optimum() -> None:
    assert gap_to_optimum(639.0, 426) == pytest.approx(0.5)
    assert gap_to_optimum(852.0, 426) == pytest.approx(1.0)


def test_gap_returns_a_float() -> None:
    assert isinstance(gap_to_optimum(500.0, 426), float)


# --- CellResult aggregates -----------------------------------------------------


def test_cell_exposes_its_gaps_in_record_order() -> None:
    cell = _cell("eil51", [0.1, 0.2, 0.3])
    assert cell.gaps == (0.1, 0.2, 0.3)


def test_cell_reports_central_tendency() -> None:
    cell = _cell("eil51", [0.1, 0.2, 0.3])
    assert cell.mean == pytest.approx(0.2)
    assert cell.median == pytest.approx(0.2)


def test_cell_reports_the_spread() -> None:
    cell = _cell("eil51", [0.1, 0.2, 0.3])
    assert cell.minimum == pytest.approx(0.1)
    assert cell.maximum == pytest.approx(0.3)
    # Sample standard deviation (ddof=1), the convention for error bars over the
    # 30 repetitions: var = (0.01 + 0 + 0.01) / (3 - 1) = 0.01, std = 0.1.
    assert cell.std == pytest.approx(0.1)


def test_std_of_a_single_run_is_zero() -> None:
    # A single seed (e.g. a --seeds 1 quick run) has no sample spread; the metric
    # must degrade to 0.0 rather than raising.
    assert _cell("eil51", [0.42]).std == 0.0


def test_cell_carries_its_configuration() -> None:
    cell = _cell("berlin52", [0.3, 0.4])
    assert cell.instance_id == "berlin52"
    assert cell.population_size == 50
    assert cell.generations == 50
    assert len(cell.records) == 2


# --- format_table --------------------------------------------------------------


def test_table_has_a_header_and_one_row_per_cell() -> None:
    cells = [_cell("eil22", [0.1, 0.2]), _cell("eil51", [0.3, 0.4])]
    lines = format_table(cells).splitlines()
    assert len(lines) >= len(cells) + 1  # at least a header plus one row per cell


def test_table_names_every_instance() -> None:
    cells = [_cell("eil22", [0.1]), _cell("eil51", [0.2]), _cell("berlin52", [0.3])]
    table = format_table(cells)
    for cell in cells:
        assert cell.instance_id in table


# --- write_csv -----------------------------------------------------------------


def test_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write_csv(path, [_record()])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _CSV_COLUMNS


def test_csv_writes_one_row_per_record(tmp_path: Path) -> None:
    records = [_record(seed=1, gap=0.5), _record(seed=2, gap=0.25)]
    path = tmp_path / "out.csv"
    write_csv(path, records)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["seed"] for row in rows] == ["1", "2"]


def test_csv_preserves_raw_gap_as_a_fraction(tmp_path: Path) -> None:
    # The published rows feed plots, so gap stays a raw fraction here (percent
    # formatting, if any, belongs only to the stdout table).
    path = tmp_path / "out.csv"
    write_csv(path, [_record(best_objective=639.0, optimum=426, gap=0.5)])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert float(row["gap"]) == pytest.approx(0.5)
    assert float(row["best_objective"]) == pytest.approx(639.0)
    assert int(row["optimum"]) == 426


# --- write_aggregated_csv ------------------------------------------------------


def test_aggregated_csv_has_the_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "agg.csv"
    write_aggregated_csv(path, [_cell("eil51", [0.1, 0.2, 0.3])])
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == _AGG_CSV_COLUMNS


def test_aggregated_csv_writes_one_row_per_cell(tmp_path: Path) -> None:
    cells = [_cell("eil22", [0.1, 0.2]), _cell("eil51", [0.3, 0.4])]
    path = tmp_path / "agg.csv"
    write_aggregated_csv(path, cells)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["instance_id"] for row in rows] == ["eil22", "eil51"]


def test_aggregated_csv_row_matches_the_cell_aggregates(tmp_path: Path) -> None:
    cell = _cell("eil51", [0.1, 0.2, 0.3])
    path = tmp_path / "agg.csv"
    write_aggregated_csv(path, [cell])
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert int(row["n_seeds"]) == 3
    assert float(row["mean_gap"]) == pytest.approx(cell.mean)
    assert float(row["std_gap"]) == pytest.approx(cell.std)
    assert float(row["median_gap"]) == pytest.approx(cell.median)
    assert float(row["min_gap"]) == pytest.approx(cell.minimum)
    assert float(row["max_gap"]) == pytest.approx(cell.maximum)


# --- public API surface --------------------------------------------------------


def test_sweep_constants_match_the_methodology() -> None:
    import experiments.baselines.cga_tsp_report as report

    assert report.INSTANCES == ("eil22", "eil51", "berlin52")
    assert report.POPULATION_SIZES == (20, 50)
    assert report.GENERATION_BUDGETS == (20, 30, 50)


def test_exposes_the_sweep_and_entry_point() -> None:
    import experiments.baselines.cga_tsp_report as report

    for name in ("run_cga", "evaluate_cell", "evaluate_sweep", "main"):
        assert hasattr(report, name)
