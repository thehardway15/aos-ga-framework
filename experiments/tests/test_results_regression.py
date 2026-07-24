"""Regression guard: the versioned result tables still match the code that produced them.

The aggregated CSVs under ``results/aggregated/`` are committed artifacts that the
thesis quotes, but nothing else in the suite ties them to the code. Every other test
here checks a contract in isolation, so a change to an operator, the pool scaling, the
engine or the seed derivation stays green while silently invalidating numbers already
written down. That has happened once: rescaling the polynomial mutation by the domain
width moved every continuous result, and the drift was found by hand, later, rather
than by a failing test.

The guard is deliberately narrow. It re-runs **one** grid cell per artifact -- the
cheapest instance at the smallest budget, over the full set of repetition seeds -- and
compares every aggregate the artifact stores, not just the median. That is enough:
these harnesses share the engine, the seeds and the aggregation, so a change with the
power to move one cell moves the whole sweep. One cell costs a few seconds; the full
sweep costs hours and belongs in a deliberate re-run, not in the suite.

A failure here is not necessarily a defect. It means the code and the committed
results have parted ways, and exactly one of two things must follow: revert the change,
or re-run the affected sweep and commit the new artifact together with the reason.
What must not happen is the two drifting apart unnoticed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from experiments.baselines import (
    cga_continuous_report,
    cga_knapsack_report,
    cga_tsp_report,
    fbo_report,
    random_report,
)
from experiments.configs.families import FAMILIES
from experiments.datasets.seeds import load_repetition_seeds
from experiments.problems.continuous import SPHERE
from experiments.strategies import pm_report

# The cheapest corner of the grid: the smallest population and the shortest budget.
# The cell is a witness, not a sample -- see the module docstring.
POPULATION_SIZE = 20
GENERATIONS = 20

# Absolute tolerance for comparing a stored decimal string against a recomputed float.
# The values are written with ``repr``-grade precision, so agreement is expected to be
# near-exact; the tolerance absorbs decimal round-tripping, nothing more.
TOLERANCE = 1e-9

_AGGREGATED = Path(__file__).resolve().parents[2] / "results" / "aggregated"

_FAMILIES = {family.problem: family for family in FAMILIES}


def _load_rows(name: str, **key: str) -> list[dict[str, str]]:
    """Return the rows of ``results/aggregated/<name>`` matching every column in ``key``."""
    with (_AGGREGATED / name).open(newline="") as csvfile:
        rows = list(csv.DictReader(csvfile))

    return [row for row in rows if all(row[column] == value for column, value in key.items())]


def _load_row(name: str, **key: str) -> dict[str, str]:
    """Return the single row of ``results/aggregated/<name>`` identified by ``key``."""
    rows = _load_rows(name, **key)
    assert len(rows) == 1, f"expected exactly one row in {name} for {key}, got {len(rows)}"

    return rows[0]


def _assert_matches(row: dict[str, str], recomputed: Any, columns: dict[str, str]) -> None:
    """Check each stored ``column`` against the attribute of ``recomputed`` it came from."""
    for column, attribute in columns.items():
        stored = float(row[column])
        actual = float(getattr(recomputed, attribute))
        assert actual == pytest.approx(stored, abs=TOLERANCE), (
            f"{column}: artifact has {stored!r}, code now produces {actual!r}"
        )


@pytest.fixture(scope="module")
def seeds() -> list[int]:
    """The study's repetition seeds -- the same set every level of the sweep uses."""
    return load_repetition_seeds()


def test_cga_tsp_cell_still_matches_the_versioned_aggregate(seeds: list[int]) -> None:
    row = _load_row(
        "cga_tsp.csv",
        instance_id="eil22",
        population_size=str(POPULATION_SIZE),
        generations=str(GENERATIONS),
    )
    cell = cga_tsp_report.evaluate_cell(
        "eil22", population_size=POPULATION_SIZE, generations=GENERATIONS, seeds=seeds
    )

    assert int(row["n_seeds"]) == len(cell.records)
    _assert_matches(
        row,
        cell,
        {
            "mean_gap": "mean",
            "std_gap": "std",
            "median_gap": "median",
            "min_gap": "minimum",
            "max_gap": "maximum",
        },
    )


def test_cga_knapsack_cell_still_matches_the_versioned_aggregate(seeds: list[int]) -> None:
    row = _load_row(
        "cga_knapsack.csv",
        instance_id="n20_uncorrelated",
        population_size=str(POPULATION_SIZE),
        generations=str(GENERATIONS),
    )
    cell = cga_knapsack_report.evaluate_cell(
        "n20_uncorrelated",
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seeds=seeds,
    )

    assert int(row["n_seeds"]) == len(cell.records)
    _assert_matches(
        row,
        cell,
        {
            "mean_gap": "mean",
            "std_gap": "std",
            "median_gap": "median",
            "min_gap": "minimum",
            "max_gap": "maximum",
        },
    )


def test_cga_continuous_cell_still_matches_the_versioned_aggregate(seeds: list[int]) -> None:
    row = _load_row(
        "cga_continuous.csv",
        function="sphere",
        dimension="5",
        population_size=str(POPULATION_SIZE),
        generations=str(GENERATIONS),
    )
    cell = cga_continuous_report.evaluate_cell(
        SPHERE, 5, population_size=POPULATION_SIZE, generations=GENERATIONS, seeds=seeds
    )

    assert int(row["n_seeds"]) == len(cell.records)
    _assert_matches(
        row,
        cell,
        {
            "mean_fitness": "mean",
            "std_fitness": "std",
            "median_fitness": "median",
            "min_fitness": "minimum",
            "max_fitness": "maximum",
        },
    )


def test_fbo_cell_still_matches_the_versioned_aggregates(seeds: list[int]) -> None:
    # The one cell that covers a whole pool: every operator of the full permutation pool
    # runs solo, so a change to any single operator surfaces here.
    operator_results, oracle_rows = fbo_report.evaluate_cell(
        _FAMILIES["tsp"],
        "eil22",
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seeds=seeds,
    )

    for operator_result in operator_results:
        row = _load_row(
            "fbo_operators.csv",
            problem="tsp",
            instance_id="eil22",
            population_size=str(POPULATION_SIZE),
            generations=str(GENERATIONS),
            operator_id=operator_result.operator_id,
        )
        assert int(row["n_seeds"]) == len(operator_result.records)
        _assert_matches(
            row,
            operator_result,
            {
                "median_quality": "median",
                "mean_quality": "mean",
                "std_quality": "std",
                "min_quality": "minimum",
                "max_quality": "maximum",
            },
        )

    # The reference operator is derived, not measured, so it is checked separately: a
    # change could move the medians without moving the argmax, or the reverse.
    for oracle_row in oracle_rows:
        row = _load_row(
            "fbo_oracle.csv",
            problem="tsp",
            instance_id="eil22",
            population_size=str(POPULATION_SIZE),
            generations=str(GENERATIONS),
            pool_variant=oracle_row.pool_variant.value,
        )
        assert row["o_star"] == ";".join(oracle_row.oracle.o_star)
        assert int(row["o_star_count"]) == oracle_row.oracle.o_star_count
        assert float(row["o_star_median"]) == pytest.approx(
            oracle_row.oracle.o_star_median, abs=TOLERANCE
        )


@pytest.mark.parametrize(
    ("artifact", "module"),
    [("random_baseline.csv", random_report), ("aos_pm.csv", pm_report)],
    ids=["random", "probability-matching"],
)
def test_pooled_cell_still_matches_the_versioned_aggregate(
    artifact: str, module: Any, seeds: list[int]
) -> None:
    # Both pool variants at once: these two harnesses share a schema by design, so they
    # share a check as well.
    results = module.evaluate_cell(
        _FAMILIES["tsp"],
        "eil22",
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seeds=seeds,
    )

    for result in results:
        row = _load_row(
            artifact,
            problem="tsp",
            instance_id="eil22",
            population_size=str(POPULATION_SIZE),
            generations=str(GENERATIONS),
            pool_variant=result.pool_variant.value,
        )
        assert int(row["n_seeds"]) == len(result.records)
        _assert_matches(
            row,
            result,
            {
                "median_quality": "median",
                "mean_quality": "mean",
                "std_quality": "std",
                "min_quality": "minimum",
                "max_quality": "maximum",
            },
        )
