"""Integrity checks for the frozen TSPLIB dataset in ``data/tsplib/``.

These run against the real, versioned artifacts produced by the builder: every
instance loads with a matching checksum, and evaluating each recorded optimal tour
reproduces the manifest's optimal length — including the known TSPLIB optima
eil51 -> 426 and berlin52 -> 7542, plus the Held-Karp value eil22 -> 278.
"""

from __future__ import annotations

import pytest

from experiments.datasets.tsplib import load_instance, load_manifest, load_optimal_tour
from experiments.problems.tsp import TSPProblem

_EXPECTED_OPTIMA = {"eil22": 278, "eil51": 426, "berlin52": 7542}


def test_manifest_lists_all_three_instances() -> None:
    assert {entry.instance_id for entry in load_manifest()} == set(_EXPECTED_OPTIMA)


@pytest.mark.parametrize("instance_id", sorted(_EXPECTED_OPTIMA))
def test_instance_loads_with_verified_checksum(instance_id: str) -> None:
    # verify=True by default: a checksum mismatch would raise here.
    instance = load_instance(instance_id)
    assert instance.instance_id == instance_id
    assert instance.dimension == len(instance.coordinates)
    assert instance.edge_weight_type == "EUC_2D"


@pytest.mark.parametrize(("instance_id", "expected"), sorted(_EXPECTED_OPTIMA.items()))
def test_optimal_tour_reproduces_recorded_optimum(instance_id: str, expected: int) -> None:
    problem = TSPProblem(load_instance(instance_id))
    assert problem.evaluate(load_optimal_tour(instance_id)) == expected

    entry = next(e for e in load_manifest() if e.instance_id == instance_id)
    assert entry.optimal_length == expected
