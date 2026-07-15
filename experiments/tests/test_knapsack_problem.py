"""Contract tests for the 0/1 knapsack test problem.

These pin the public API of :class:`experiments.problems.knapsack.KnapsackProblem`:
its metadata, the big-M penalty rule ``rho = sum(values) + 1``, the value-plus-penalty
``evaluate`` (a maximization objective), the seeded ``initialize`` and the inherited
quality ``g``. They run on small synthetic ``KnapsackInstance`` objects with
hand-computable fitness -- a 4-item instance whose ``50%`` capacity is ``W=5`` -- so
every asserted number is checkable by hand and the whole feasible/infeasible split is
enumerable. Validation against real dataset instances and exact DP optima lives in the
later exact-solver and reporting steps, not here.

The module these tests import does not exist yet: this file is the executable
specification. Expected public name: ``KnapsackProblem``.

Key facts pinned:
- direction MAXIMIZE, representation BINARY, genome ``list[int]`` of 0/1 bits;
- ``f(x) = sum(v_i x_i) - rho * max(0, sum(w_i x_i) - W)`` with ``rho = sum(v) + 1``;
- big-M guarantee: every infeasible solution scores strictly below the empty
  knapsack (``f < 0``), by the same rule across all correlation types;
- capacity is a feasible boundary (``sum(w_i x_i) == W`` incurs no penalty);
- legalization is by penalty, so ``repair`` is the inherited identity.
"""

from __future__ import annotations

import itertools
import pickle
from collections.abc import Sequence

import numpy as np

from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation
from experiments.datasets.knapsack import KnapsackInstance
from experiments.problems.knapsack import KnapsackProblem

# A 4-item instance with tiny weights so the "50% knapsack" capacity is W = 5:
# sum(weights) = 10 -> floor(0.5 * 10) = 5. sum(values) = 100 -> rho = 101.
_VALUES: list[int] = [10, 20, 30, 40]
_WEIGHTS: list[int] = [1, 2, 3, 4]
_CAPACITY = 5
_PENALTY = 101


def _instance(
    values: Sequence[int],
    weights: Sequence[int],
    *,
    correlation_type: str = "uncorrelated",
    name: str = "kp_demo",
) -> KnapsackInstance:
    """Build a synthetic instance with the 50% capacity rule ``W = floor(0.5 * sum(w))``."""
    return KnapsackInstance(
        instance_id=name,
        n=len(values),
        R=1000,
        correlation_type=correlation_type,
        values=tuple(values),
        weights=tuple(weights),
        capacity=sum(weights) // 2,
        seed=0,
    )


def _problem(
    values: Sequence[int] = _VALUES,
    weights: Sequence[int] = _WEIGHTS,
    *,
    correlation_type: str = "uncorrelated",
) -> KnapsackProblem:
    return KnapsackProblem(_instance(values, weights, correlation_type=correlation_type))


def _uniform_problem(n: int) -> KnapsackProblem:
    """An ``n``-bit instance whose values/weights are irrelevant to ``initialize``."""
    return _problem([1] * n, [1] * n)


# --- metadata ------------------------------------------------------------------


def test_exposes_metadata() -> None:
    problem = _problem(correlation_type="strongly")
    assert problem.name == "kp_demo"
    assert problem.direction is Direction.MAXIMIZE
    assert problem.representation is Representation.BINARY
    assert problem.dimension == 4
    assert problem.correlation_type == "strongly"


def test_is_a_problem_instance() -> None:
    assert isinstance(_problem(), Problem)


# --- big-M penalty rule --------------------------------------------------------


def test_penalty_is_sum_of_values_plus_one() -> None:
    # rho = sum(v) + 1 = 101 for this instance, computed per instance from the data.
    assert _problem().penalty == _PENALTY


# --- evaluate: value accounting ------------------------------------------------


def test_evaluate_empty_knapsack_is_zero() -> None:
    assert _problem().evaluate([0, 0, 0, 0]) == 0.0


def test_evaluate_feasible_sums_selected_values() -> None:
    # No penalty while total weight <= W: fitness is just the sum of chosen values.
    problem = _problem()
    assert problem.evaluate([1, 1, 0, 0]) == 30  # v=10+20, w=1+2=3 <= 5
    assert problem.evaluate([0, 0, 0, 1]) == 40  # v=40,    w=4    <= 5
    assert problem.evaluate([1, 0, 1, 0]) == 40  # v=10+30, w=1+3=4 <= 5


def test_evaluate_at_capacity_boundary_counts_as_feasible() -> None:
    # sum(w) == W is feasible (overflow = max(0, 0) = 0), so no penalty applies.
    assert _problem().evaluate([1, 0, 0, 1]) == 50  # v=10+40, w=1+4=5 == W


def test_evaluate_returns_a_float() -> None:
    assert isinstance(_problem().evaluate([1, 1, 0, 0]), float)


# --- evaluate: penalty accounting ----------------------------------------------


def test_evaluate_infeasible_subtracts_big_m_penalty() -> None:
    problem = _problem()
    # w=1+2+3=6 > 5 -> overflow 1: f = 60 - 101 * 1 = -41.
    assert problem.evaluate([1, 1, 1, 0]) == 60 - _PENALTY * 1
    # w=10 > 5 -> overflow 5: f = 100 - 101 * 5 = -405.
    assert problem.evaluate([1, 1, 1, 1]) == 100 - _PENALTY * 5


def test_every_infeasible_solution_scores_below_the_empty_knapsack() -> None:
    # The big-M acceptance criterion, checked exhaustively on the 2^4 lattice.
    problem = _problem()
    empty = problem.evaluate([0, 0, 0, 0])
    for bits in itertools.product((0, 1), repeat=problem.dimension):
        x = list(bits)
        weight = sum(w for w, b in zip(_WEIGHTS, x, strict=True) if b)
        if weight > problem.capacity:
            assert problem.evaluate(x) < empty, x


def test_penalty_rule_is_uniform_across_correlation_types() -> None:
    # Same rule rho = sum(v) + 1 for every correlation type -> the big-M guarantee
    # holds identically for uncorrelated, weakly and strongly correlated instances
    # (different rho values, one uniform rule; comparable across correlations).
    instances = [
        ("uncorrelated", [8, 3, 11, 5], [2, 7, 4, 9]),
        ("weakly", [12, 14, 9, 20], [10, 12, 8, 18]),
        ("strongly", [21, 31, 41, 11], [20, 30, 40, 10]),
    ]
    for correlation_type, values, weights in instances:
        problem = _problem(values, weights, correlation_type=correlation_type)
        assert problem.penalty == sum(values) + 1
        empty = problem.evaluate([0] * problem.dimension)
        for bits in itertools.product((0, 1), repeat=problem.dimension):
            x = list(bits)
            weight = sum(w for w, b in zip(weights, x, strict=True) if b)
            if weight > problem.capacity:
                assert problem.evaluate(x) < empty, (correlation_type, x)


# --- quality g(x) --------------------------------------------------------------


def test_g_equals_objective_for_maximization() -> None:
    # Maximization: g = +f, both for feasible and infeasible genomes.
    problem = _problem()
    for x in ([1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]):
        assert problem.g(x) == problem.evaluate(x)


def test_higher_value_feasible_has_higher_quality() -> None:
    problem = _problem()
    assert problem.evaluate([0, 0, 0, 1]) > problem.evaluate([1, 1, 0, 0])  # 40 > 30
    assert problem.g([0, 0, 0, 1]) > problem.g([1, 1, 0, 0])


# --- initialize ----------------------------------------------------------------


def test_initialize_produces_a_bitstring() -> None:
    genome = _uniform_problem(8).initialize(np.random.default_rng(0))
    assert len(genome) == 8
    assert all(bit in (0, 1) for bit in genome)


def test_initialize_returns_a_list_of_int() -> None:
    genome = _uniform_problem(8).initialize(np.random.default_rng(0))
    assert isinstance(genome, list)
    assert all(isinstance(bit, int) for bit in genome)


def test_initialize_is_deterministic_for_the_same_seed() -> None:
    problem = _uniform_problem(16)
    assert problem.initialize(np.random.default_rng(42)) == problem.initialize(
        np.random.default_rng(42)
    )


def test_initialize_differs_for_different_seeds() -> None:
    # A 32-bit space makes a collision between two fixed seeds effectively impossible,
    # independent of how the Bernoulli(0.5) draw is implemented.
    problem = _uniform_problem(32)
    assert problem.initialize(np.random.default_rng(1)) != problem.initialize(
        np.random.default_rng(2)
    )


def test_initialize_uses_only_the_injected_generator() -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    problem = _uniform_problem(16)
    before = pickle.dumps(np.random.get_state())
    problem.initialize(np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


# --- repair --------------------------------------------------------------------


def test_repair_is_identity() -> None:
    # Legalization is by penalty, not repair: repair leaves even an infeasible genome
    # unchanged (the inherited identity from Problem).
    problem = _problem()
    x = [1, 1, 1, 0]  # infeasible, yet repair must not alter it
    assert problem.repair(x) == x
