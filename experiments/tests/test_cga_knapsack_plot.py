"""Contract tests for the CGA-on-knapsack gap plot.

These pin the pure, matplotlib-backed helpers of
:mod:`experiments.baselines.cga_knapsack_plot`: reading the aggregated CSV,
building a figure with one panel per instance (a 3x3 grid over the three sizes and
three correlation types), and writing it to disk. The whole module is marked
``heavy`` because it needs the ``analysis`` optional dependency (matplotlib), so it
runs only in the CI job that installs those extras.

The aggregated CSV carries a ``correlation_type`` column the plot does not itself
need; ``load_aggregated`` must read the columns it uses and ignore the rest, so the
fixture below includes that column to mirror the real report output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.baselines.cga_knapsack_plot import (
    AggRow,
    build_figure,
    load_aggregated,
    save_figure,
)
from experiments.baselines.cga_knapsack_report import (
    GENERATION_BUDGETS,
    INSTANCES,
    POPULATION_SIZES,
)

pytestmark = pytest.mark.heavy


def _rows() -> list[AggRow]:
    """A full grid of aggregated rows with placeholder statistics."""
    return [
        AggRow(
            instance_id=instance_id,
            population_size=population_size,
            generations=generations,
            mean_gap=0.5,
            std_gap=0.1,
        )
        for instance_id in INSTANCES
        for population_size in POPULATION_SIZES
        for generations in GENERATION_BUDGETS
    ]


def test_load_aggregated_reads_typed_rows(tmp_path: Path) -> None:
    path = tmp_path / "agg.csv"
    path.write_text(
        "instance_id,correlation_type,population_size,generations,"
        "n_seeds,mean_gap,std_gap,median_gap,min_gap,max_gap\n"
        "n50_strongly,strongly,50,50,30,0.16,0.03,0.15,0.10,0.24\n",
        encoding="utf-8",
    )
    rows = load_aggregated(path)
    assert len(rows) == 1
    assert rows[0].instance_id == "n50_strongly"
    assert rows[0].population_size == 50
    assert rows[0].generations == 50
    assert rows[0].mean_gap == pytest.approx(0.16)
    assert rows[0].std_gap == pytest.approx(0.03)


def test_build_figure_has_one_panel_per_instance() -> None:
    fig = build_figure(_rows())
    assert len(fig.axes) == len(INSTANCES)


def test_save_figure_creates_a_nonempty_file(tmp_path: Path) -> None:
    out = tmp_path / "fig.png"
    save_figure(build_figure(_rows()), out)
    assert out.exists()
    assert out.stat().st_size > 0
