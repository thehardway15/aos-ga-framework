from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .cga_knapsack_report import GENERATION_BUDGETS, INSTANCES, POPULATION_SIZES

_DEFAULT_AGG_CSV = Path("results/aggregated/cga_knapsack.csv")
_DEFAULT_FIGURE = Path("results/figures/cga_knapsack_gap.png")


@dataclass(frozen=True)
class AggRow:
    """One aggregated cell read back from the CSV (only the fields the plot needs)."""

    instance_id: str
    population_size: int
    generations: int
    mean_gap: float
    std_gap: float


def load_aggregated(path: Path) -> list[AggRow]:
    """Read the aggregated-CSV rows written by ``cga_knapsack_report --agg-csv``."""
    with path.open(newline="", encoding="utf-8") as f:
        return [
            AggRow(
                instance_id=row["instance_id"],
                population_size=int(row["population_size"]),
                generations=int(row["generations"]),
                mean_gap=float(row["mean_gap"]),
                std_gap=float(row["std_gap"]),
            )
            for row in csv.DictReader(f)
        ]


def build_figure(rows: list[AggRow]) -> Figure:
    """Draw a 3x3 grid of panels (rows are sizes, columns are correlation types).

    One panel per instance: mean gap vs budget, a line per population, std bars. The
    grid layout follows the size-major ``INSTANCES`` order, so rows read n=20/30/50 and
    columns read uncorrelated/weakly/strongly.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=False)
    for ax, instance_id in zip(axes.flat, INSTANCES, strict=True):
        for population_size in POPULATION_SIZES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row.instance_id == instance_id and row.population_size == population_size
                ),
                key=lambda row: row.generations,
            )
            ax.errorbar(
                [row.generations for row in selected],
                [row.mean_gap for row in selected],
                yerr=[row.std_gap for row in selected],
                marker="o",
                capsize=3,
                label=f"N={population_size}",
            )
        ax.set_title(instance_id)
        ax.set_xlabel("Generation budget G")
        ax.set_ylabel("Mean gap to optimum")
        ax.set_xticks(list(GENERATION_BUDGETS))
        ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    """Write ``fig`` to ``path`` (creating the parent directory), at 150 dpi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the CGA-on-Knapsack gap baseline.")
    parser.add_argument(
        "--agg-csv",
        type=Path,
        default=_DEFAULT_AGG_CSV,
        help="Path to the aggregated CSV produced by cga_knapsack_report --agg-csv.",
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
