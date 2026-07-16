"""Contract tests for the exact 0/1 knapsack dynamic-programming solver.

These pin the public API of :mod:`experiments.datasets.exact_knapsack`: a single
function ``knapsack_dp`` returning the exact optimal value and one optimal 0/1
selection vector for a small integer instance. The solver is deterministic and is
used once, offline, to fix the reference optima of the nine Pisinger knapsack
instances, which -- unlike the TSP instances -- ship without a published optimum.

The module these tests import does not exist yet: this file is the executable
specification of the solver's contract. Expected public name: ``knapsack_dp``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from experiments.datasets.exact_knapsack import knapsack_dp


def _total(quantities: Sequence[int], selection: Sequence[int]) -> int:
    """Dot product of an integer vector with a 0/1 selection mask."""
    return sum(q * s for q, s in zip(quantities, selection, strict=True))


def _brute_force_optimum(values: Sequence[int], weights: Sequence[int], capacity: int) -> int:
    """Optimal 0/1 knapsack value by exhaustive search over all 2**n subsets."""
    best = 0
    for bits in itertools.product((0, 1), repeat=len(values)):
        if _total(weights, bits) <= capacity:
            best = max(best, _total(values, bits))
    return best


def _random_instance(n: int, seed: int) -> tuple[list[int], list[int], int]:
    """A random integer instance with a 50%-of-total-weight capacity."""
    rng = np.random.default_rng(seed)
    values = [int(v) for v in rng.integers(1, 100, size=n)]
    weights = [int(w) for w in rng.integers(1, 100, size=n)]
    capacity = sum(weights) // 2
    return values, weights, capacity


# The classic 3-item instance: {item1, item2} weigh exactly 50 and are worth 220.
_CLASSIC_VALUES = [60, 100, 120]
_CLASSIC_WEIGHTS = [10, 20, 30]
_CLASSIC_CAPACITY = 50


# --- return shape --------------------------------------------------------------


def test_returns_value_and_selection() -> None:
    value, selection = knapsack_dp(_CLASSIC_VALUES, _CLASSIC_WEIGHTS, _CLASSIC_CAPACITY)
    assert isinstance(value, int)
    assert isinstance(selection, list)
    assert all(isinstance(bit, int) for bit in selection)


def test_selection_is_a_bitstring_of_length_n() -> None:
    _, selection = knapsack_dp(_CLASSIC_VALUES, _CLASSIC_WEIGHTS, _CLASSIC_CAPACITY)
    assert len(selection) == len(_CLASSIC_VALUES)
    assert set(selection) <= {0, 1}


# --- self-consistency ----------------------------------------------------------


def test_selection_is_feasible() -> None:
    _, selection = knapsack_dp(_CLASSIC_VALUES, _CLASSIC_WEIGHTS, _CLASSIC_CAPACITY)
    assert _total(_CLASSIC_WEIGHTS, selection) <= _CLASSIC_CAPACITY


def test_reported_value_matches_the_returned_selection() -> None:
    value, selection = knapsack_dp(_CLASSIC_VALUES, _CLASSIC_WEIGHTS, _CLASSIC_CAPACITY)
    assert _total(_CLASSIC_VALUES, selection) == value


# --- known small optimum -------------------------------------------------------


def test_classic_instance_optimum() -> None:
    # The optimal subset {item1, item2} is unique, so any correct solver returns it.
    value, selection = knapsack_dp(_CLASSIC_VALUES, _CLASSIC_WEIGHTS, _CLASSIC_CAPACITY)
    assert value == 220
    assert selection == [0, 1, 1]


# --- boundary cases ------------------------------------------------------------


def test_zero_capacity_selects_nothing() -> None:
    value, selection = knapsack_dp([1, 2, 3], [3, 4, 5], 0)
    assert value == 0
    assert selection == [0, 0, 0]


def test_every_item_heavier_than_capacity_selects_nothing() -> None:
    value, selection = knapsack_dp([5, 6], [10, 20], 5)
    assert value == 0
    assert selection == [0, 0]


def test_capacity_fits_all_items_selects_everything() -> None:
    value, selection = knapsack_dp([3, 4], [1, 2], 100)
    assert value == 7
    assert selection == [1, 1]


def test_boundary_capacity_equal_to_total_weight_is_feasible() -> None:
    # capacity == sum(weights): the full set is exactly feasible and optimal.
    value, selection = knapsack_dp([3, 4], [1, 2], 3)
    assert value == 7
    assert selection == [1, 1]


def test_empty_instance_returns_zero_and_empty_selection() -> None:
    value, selection = knapsack_dp([], [], 10)
    assert value == 0
    assert selection == []


# --- agreement with exhaustive search ------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 8, 10, 12])
def test_matches_brute_force_on_small_instances(n: int) -> None:
    values, weights, capacity = _random_instance(n, seed=n)
    value, selection = knapsack_dp(values, weights, capacity)
    assert value == _brute_force_optimum(values, weights, capacity)
    # The returned selection realises that value and respects the capacity.
    assert _total(values, selection) == value
    assert _total(weights, selection) <= capacity
    assert len(selection) == n


def test_is_deterministic_across_calls() -> None:
    values, weights, capacity = _random_instance(10, seed=123)
    assert knapsack_dp(values, weights, capacity) == knapsack_dp(values, weights, capacity)
