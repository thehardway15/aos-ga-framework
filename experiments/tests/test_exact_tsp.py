"""Contract tests for the exact Held-Karp TSP solver.

These pin the public API of :mod:`experiments.datasets.exact_tsp`: a single
function ``held_karp`` returning the exact optimal tour length and one optimal
tour for a small symmetric distance matrix. The solver is deterministic and is
used once, offline, to fix the reference optimum of the ``eil22`` instance, which
has no published optimum in TSPLIB's symmetric-TSP section.

The module these tests import does not exist yet: this file is the executable
specification of the solver's contract. Expected public name: ``held_karp``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from experiments.datasets.exact_tsp import held_karp


def _tour_length(distance: NDArray[np.int64], tour: list[int]) -> int:
    """Closed-cycle length of ``tour`` under ``distance`` (wraps last to first)."""
    n = len(tour)
    return int(sum(distance[tour[i], tour[(i + 1) % n]] for i in range(n)))


def _brute_force_optimum(distance: NDArray[np.int64]) -> int:
    """Optimal closed-cycle length by exhaustive search, fixing the start at city 0."""
    n = int(distance.shape[0])
    return min(_tour_length(distance, [0, *perm]) for perm in itertools.permutations(range(1, n)))


def _random_symmetric_matrix(n: int, seed: int) -> NDArray[np.int64]:
    """A random symmetric integer distance matrix with a zero diagonal."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.integers(1, 100, size=(n, n)), k=1)
    return (upper + upper.T).astype(np.int64)


# A unit square scaled to side 10; sides cost 10, diagonals nint(sqrt(200)) = 14.
_SQUARE = np.array(
    [
        [0, 10, 14, 10],
        [10, 0, 10, 14],
        [14, 10, 0, 10],
        [10, 14, 10, 0],
    ],
    dtype=np.int64,
)

# A 3-city instance: every tour is the same single cycle 0-1-2-0.
_TRIANGLE = np.array(
    [
        [0, 3, 4],
        [3, 0, 5],
        [4, 5, 0],
    ],
    dtype=np.int64,
)


# --- return shape ---------------------------------------------------------------


def test_held_karp_returns_length_and_tour() -> None:
    length, tour = held_karp(_SQUARE)
    assert isinstance(length, int)
    assert isinstance(tour, list)
    assert all(isinstance(city, int) for city in tour)


def test_tour_is_a_permutation_of_all_cities_starting_at_zero() -> None:
    _, tour = held_karp(_SQUARE)
    assert tour[0] == 0
    assert sorted(tour) == list(range(_SQUARE.shape[0]))


def test_reported_length_matches_the_returned_tour() -> None:
    length, tour = held_karp(_SQUARE)
    assert _tour_length(_SQUARE, tour) == length


# --- known small optima ---------------------------------------------------------


def test_square_prefers_the_perimeter_over_the_diagonals() -> None:
    # Going around the sides (4 x 10) beats any tour using a diagonal (>= 48).
    length, _ = held_karp(_SQUARE)
    assert length == 40


def test_triangle_returns_the_single_cycle_length() -> None:
    length, tour = held_karp(_TRIANGLE)
    assert length == 12
    assert sorted(tour) == [0, 1, 2]


def test_two_cities_tour_traverses_the_edge_twice() -> None:
    distance = np.array([[0, 7], [7, 0]], dtype=np.int64)
    length, tour = held_karp(distance)
    assert length == 14
    assert sorted(tour) == [0, 1]


# --- agreement with exhaustive search ------------------------------------------


@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_matches_brute_force_on_small_instances(n: int) -> None:
    distance = _random_symmetric_matrix(n, seed=n)
    length, tour = held_karp(distance)
    assert length == _brute_force_optimum(distance)
    assert _tour_length(distance, tour) == length
    assert sorted(tour) == list(range(n))


def test_is_deterministic_across_calls() -> None:
    distance = _random_symmetric_matrix(7, seed=123)
    assert held_karp(distance) == held_karp(distance)
