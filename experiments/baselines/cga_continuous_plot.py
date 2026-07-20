"""Plot the CGA-on-continuous best-fitness baseline.

Reads the aggregated CSV written by ``cga_continuous_report --agg-csv`` and draws a
3x2 grid of panels: one per (function, dimension) configuration, rows following the
``FUNCTIONS`` order (sphere/rastrigin/rosenbrock) and columns the ``DIMENSIONS`` order
(d=5, d=10). Each panel plots the best fitness against the generation budget ``G``,
one line per population size ``N``.

Because every benchmark has optimum value 0, there is no gap to plot: the panels show
the best fitness itself, on a log y-axis since the values span orders of magnitude
(especially Rastrigin). The spread over the repetition seeds is drawn as the median
line with a min-max band. Both the median and the band bounds are strictly positive,
so they render on a log axis without the negative-lower-bound artifact a symmetric
mean +/- std would hit; the band is the honest depiction of the right-skewed fitness
distribution the log scale is there to reveal.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .cga_continuous_report import (
    DIMENSIONS,
    FUNCTIONS,
    GENERATION_BUDGETS,
    POPULATION_SIZES,
)

_DEFAULT_AGG_CSV = Path("results/aggregated/cga_continuous.csv")
_DEFAULT_FIGURE = Path("results/figures/cga_continuous_fitness.png")


@dataclass(frozen=True)
class AggRow:
    """One aggregated cell read back from the CSV (only the fields the plot needs).

    The plot draws a median line with a min-max band, so it reads ``median_fitness``,
    ``min_fitness`` and ``max_fitness`` and ignores the ``mean_fitness``, ``std_fitness``
    and ``n_seeds`` columns the report also writes.
    """

    function: str
    dimension: int
    population_size: int
    generations: int
    median_fitness: float
    min_fitness: float
    max_fitness: float


def load_aggregated(path: Path) -> list[AggRow]:
    """Read the aggregated-CSV rows written by ``cga_continuous_report --agg-csv``.

    Keeps only the columns the plot uses (the configuration axes plus the median and
    min/max fitness) and ignores the mean, std and seed-count columns.
    """
    with path.open(newline="", encoding="utf-8") as f:
        return [
            AggRow(
                function=row["function"],
                dimension=int(row["dimension"]),
                population_size=int(row["population_size"]),
                generations=int(row["generations"]),
                median_fitness=float(row["median_fitness"]),
                min_fitness=float(row["min_fitness"]),
                max_fitness=float(row["max_fitness"]),
            )
            for row in csv.DictReader(f)
        ]


def build_figure(rows: list[AggRow]) -> Figure:
    """Draw a 3x2 grid of panels (rows are functions, columns are dimensions).

    One panel per (function, dimension): median best fitness vs budget on a log y-axis,
    a line per population size with a min-max band in the line's colour. The grid layout
    follows the ``FUNCTIONS`` x ``DIMENSIONS`` order, so rows read sphere/rastrigin/
    rosenbrock and columns read d=5/d=10.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharey=False)
    for ax, (function, dimension) in zip(
        axes.flat, itertools.product(FUNCTIONS, DIMENSIONS), strict=True
    ):
        for population_size in POPULATION_SIZES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row.function == function.name
                    and row.dimension == dimension
                    and row.population_size == population_size
                ),
                key=lambda row: row.generations,
            )
            budgets = [row.generations for row in selected]
            (line,) = ax.plot(
                budgets,
                [row.median_fitness for row in selected],
                marker="o",
                label=f"N={population_size}",
            )
            ax.fill_between(
                budgets,
                [row.min_fitness for row in selected],
                [row.max_fitness for row in selected],
                alpha=0.2,
                color=line.get_color(),
            )
        ax.set_yscale("log")
        ax.set_title(f"{function.name} d={dimension}")
        ax.set_xlabel("Generation budget G")
        ax.set_ylabel("Best fitness (log scale)")
        ax.set_xticks(list(GENERATION_BUDGETS))
        ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    """Write ``fig`` to ``path`` (creating the parent directory), at 150 dpi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the CGA-on-continuous fitness baseline.")
    parser.add_argument(
        "--agg-csv",
        type=Path,
        default=_DEFAULT_AGG_CSV,
        help="Path to the aggregated CSV produced by cga_continuous_report --agg-csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_FIGURE,
        help="Path to write the figure to.",
    )
    args = parser.parse_args()
    save_figure(build_figure(load_aggregated(args.agg_csv)), args.out)


if __name__ == "__main__":
    main()
