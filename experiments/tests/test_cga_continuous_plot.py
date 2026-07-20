"""Contract tests for the CGA-on-continuous best-fitness plot.

These pin the pure, matplotlib-backed helpers of
:mod:`experiments.baselines.cga_continuous_plot`: reading the aggregated CSV,
building a figure with one panel per configuration (a 3x2 grid over the three
functions and two dimensions), and writing it to disk. The whole module is marked
``heavy`` because it needs the ``analysis`` optional dependency (matplotlib), so it
runs only in the CI job that installs those extras.

The optimum is 0 for every benchmark, so there is no gap to plot: the panels show the
best fitness itself on a log y-axis (the values span orders of magnitude). The spread
over the repetition seeds is drawn as a median line with a min-max band -- both bounds
are strictly positive, so they render on a log axis without the negative-lower-bound
artifact a symmetric mean +/- std would hit. The aggregated CSV also carries
``mean_fitness``, ``std_fitness`` and ``n_seeds`` columns the plot does not need;
``load_aggregated`` must read the columns it uses and ignore the rest, so the fixture
below includes them to mirror the real report output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.baselines.cga_continuous_plot import (
    AggRow,
    build_figure,
    load_aggregated,
    save_figure,
)
from experiments.baselines.cga_continuous_report import (
    DIMENSIONS,
    FUNCTIONS,
    GENERATION_BUDGETS,
    POPULATION_SIZES,
)

pytestmark = pytest.mark.heavy


def _rows() -> list[AggRow]:
    """A full grid of aggregated rows with placeholder statistics (min < median < max)."""
    return [
        AggRow(
            function=function.name,
            dimension=dimension,
            population_size=population_size,
            generations=generations,
            median_fitness=0.5,
            min_fitness=0.1,
            max_fitness=0.9,
        )
        for function in FUNCTIONS
        for dimension in DIMENSIONS
        for population_size in POPULATION_SIZES
        for generations in GENERATION_BUDGETS
    ]


def test_load_aggregated_reads_typed_rows(tmp_path: Path) -> None:
    path = tmp_path / "agg.csv"
    path.write_text(
        "function,dimension,population_size,generations,"
        "n_seeds,mean_fitness,std_fitness,median_fitness,min_fitness,max_fitness\n"
        "rastrigin,10,50,50,30,0.20,0.05,0.16,0.10,0.24\n",
        encoding="utf-8",
    )
    rows = load_aggregated(path)
    assert len(rows) == 1
    assert rows[0].function == "rastrigin"
    assert rows[0].dimension == 10
    assert rows[0].population_size == 50
    assert rows[0].generations == 50
    assert rows[0].median_fitness == pytest.approx(0.16)
    assert rows[0].min_fitness == pytest.approx(0.10)
    assert rows[0].max_fitness == pytest.approx(0.24)


def test_build_figure_has_one_panel_per_configuration() -> None:
    fig = build_figure(_rows())
    assert len(fig.axes) == len(FUNCTIONS) * len(DIMENSIONS)


def test_save_figure_creates_a_nonempty_file(tmp_path: Path) -> None:
    out = tmp_path / "fig.png"
    save_figure(build_figure(_rows()), out)
    assert out.exists()
    assert out.stat().st_size > 0
