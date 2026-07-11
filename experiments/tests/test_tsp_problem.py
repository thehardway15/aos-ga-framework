"""Contract tests for the TSP test problem.

These pin the public API of :class:`experiments.problems.tsp.TSPProblem`: its
metadata, the integer ``EUC_2D`` distance matrix (``nint`` rounding, symmetric,
zero diagonal), the closed-cycle ``evaluate``, the seeded ``initialize``, and the
inherited quality ``g``. They run on small synthetic ``TSPInstance`` objects with
hand-computable tour lengths; validation against real TSPLIB optima (eil51 -> 426,
berlin52 -> 7542) lives in a separate integrity test built after the dataset
builder.

The module these tests import does not exist yet: this file is the executable
specification. Expected public name: ``TSPProblem``.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence

import numpy as np

from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation
from experiments.datasets.tsplib import TSPInstance
from experiments.problems.tsp import TSPProblem

# A 3-4-5 right triangle: edges d(0,1)=3, d(1,2)=5, d(0,2)=4.
_TRIANGLE: list[tuple[float, float]] = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]


def _problem(coords: Sequence[tuple[float, float]], name: str = "demo") -> TSPProblem:
    instance = TSPInstance(
        instance_id=name,
        dimension=len(coords),
        edge_weight_type="EUC_2D",
        coordinates=tuple(coords),
    )
    return TSPProblem(instance)


# --- metadata ------------------------------------------------------------------


def test_exposes_metadata() -> None:
    problem = _problem(_TRIANGLE, name="tri")
    assert problem.name == "tri"
    assert problem.direction is Direction.MINIMIZE
    assert problem.representation is Representation.PERMUTATION
    assert problem.dimension == 3


def test_is_a_problem_instance() -> None:
    assert isinstance(_problem([(0.0, 0.0), (1.0, 0.0)]), Problem)


# --- distance matrix -----------------------------------------------------------


def test_distance_matrix_has_known_edges() -> None:
    problem = _problem(_TRIANGLE)
    assert problem.distances[0, 1] == 3
    assert problem.distances[1, 2] == 5
    assert problem.distances[0, 2] == 4


def test_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    problem = _problem(_TRIANGLE)
    assert bool((problem.distances == problem.distances.T).all())
    assert all(problem.distances[i, i] == 0 for i in range(problem.dimension))


def test_distance_uses_nearest_integer_rounding() -> None:
    # sqrt(8) = 2.83 -> 3 (not floor 2); sqrt(5) = 2.24 -> 2 (not ceil 3).
    problem = _problem([(0.0, 0.0), (2.0, 2.0), (1.0, 2.0)])
    assert problem.distances[0, 1] == 3
    assert problem.distances[0, 2] == 2


# --- evaluate ------------------------------------------------------------------


def test_evaluate_sums_the_closed_cycle() -> None:
    # 0->1->2->0 = 3 + 5 + 4, i.e. the return edge 2->0 is included.
    assert _problem(_TRIANGLE).evaluate([0, 1, 2]) == 12


def test_evaluate_returns_a_float() -> None:
    assert isinstance(_problem(_TRIANGLE).evaluate([0, 1, 2]), float)


def test_evaluate_is_rotation_invariant() -> None:
    problem = _problem(_TRIANGLE)
    assert problem.evaluate([0, 1, 2]) == problem.evaluate([1, 2, 0])


# --- quality g(x) --------------------------------------------------------------


def test_g_is_negated_length_for_minimization() -> None:
    problem = _problem(_TRIANGLE)
    assert problem.g([0, 1, 2]) == -problem.evaluate([0, 1, 2])


def test_shorter_tour_has_higher_quality() -> None:
    # Unit square (side 10): the perimeter (40) beats a diagonal-crossing tour (48).
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    problem = _problem(square)
    assert problem.evaluate([0, 1, 2, 3]) < problem.evaluate([0, 2, 1, 3])
    assert problem.g([0, 1, 2, 3]) > problem.g([0, 2, 1, 3])


# --- initialize ----------------------------------------------------------------


def test_initialize_produces_a_permutation() -> None:
    tour = _problem(_TRIANGLE).initialize(np.random.default_rng(0))
    assert sorted(tour) == [0, 1, 2]


def test_initialize_returns_a_list_of_int() -> None:
    tour = _problem(_TRIANGLE).initialize(np.random.default_rng(0))
    assert isinstance(tour, list)
    assert all(isinstance(city, int) for city in tour)


def test_initialize_is_deterministic_for_the_same_seed() -> None:
    problem = _problem([(float(i), 0.0) for i in range(6)])
    assert problem.initialize(np.random.default_rng(42)) == problem.initialize(
        np.random.default_rng(42)
    )


def test_initialize_differs_for_different_seeds() -> None:
    problem = _problem([(float(i), 0.0) for i in range(8)])
    assert problem.initialize(np.random.default_rng(1)) != problem.initialize(
        np.random.default_rng(2)
    )


def test_initialize_uses_only_the_injected_generator() -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    problem = _problem([(float(i), 0.0) for i in range(8)])
    before = pickle.dumps(np.random.get_state())
    problem.initialize(np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before
