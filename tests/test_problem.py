"""Contract tests for the framework's problem abstraction and quality function.

These pin the public API of :mod:`aos_ga.core.problem`: the optimization
``Direction`` (with its ``sign``), the ``Representation`` families, the unified
quality function ``g(x)`` that decides the sign of the objective once, and the
abstract ``Problem`` base every concrete test problem implements.

The names these tests import are not implemented yet: this file is the
executable specification of the contract. Expected public names: ``Direction``,
``Representation``, ``quality``, ``Problem``.

The two problems defined below are minimal test doubles -- a maximization one
(binary genome, higher is better) and a minimization one (real genome, lower is
better, with a bounded domain) -- used only to exercise the abstract contract.
The real test problems live in the ``experiments`` layer.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from aos_ga.core.problem import Direction, Problem, quality
from aos_ga.core.representation import Representation


class _MaxProblem(Problem[list[int]]):
    """A knapsack-like maximization double: fitness is the sum of a binary genome."""

    name = "max-sum"
    direction = Direction.MAXIMIZE
    representation = Representation.BINARY

    def __init__(self, size: int) -> None:
        self.size = size

    def evaluate(self, individual: list[int]) -> float:
        return float(sum(individual))

    def initialize(self, rng: np.random.Generator) -> list[int]:
        return [int(bit) for bit in rng.integers(0, 2, size=self.size)]


class _MinProblem(Problem[list[float]]):
    """A sphere-like minimization double: fitness is the sum of squares, clipped."""

    name = "min-sphere"
    direction = Direction.MINIMIZE
    representation = Representation.REAL

    def __init__(self, dim: int, low: float = -1.0, high: float = 1.0) -> None:
        self.dim = dim
        self.low = low
        self.high = high

    def evaluate(self, individual: list[float]) -> float:
        return float(sum(value * value for value in individual))

    def initialize(self, rng: np.random.Generator) -> list[float]:
        return [float(value) for value in rng.uniform(self.low, self.high, size=self.dim)]

    def repair(self, individual: list[float]) -> list[float]:
        return [min(self.high, max(self.low, value)) for value in individual]


# --- quality (the g(x) transform) ----------------------------------------------


def test_quality_is_identity_for_maximization() -> None:
    assert quality(3.5, Direction.MAXIMIZE) == 3.5


def test_quality_negates_for_minimization() -> None:
    assert quality(3.5, Direction.MINIMIZE) == -3.5


def test_quality_increases_with_objective_for_maximization() -> None:
    assert quality(2.0, Direction.MAXIMIZE) > quality(1.0, Direction.MAXIMIZE)


def test_quality_decreases_with_objective_for_minimization() -> None:
    # Lower is better, so a larger objective must map to a smaller quality.
    assert quality(2.0, Direction.MINIMIZE) < quality(1.0, Direction.MINIMIZE)


# --- Direction -----------------------------------------------------------------


def test_direction_sign_is_the_f_to_g_multiplier() -> None:
    assert Direction.MAXIMIZE.sign == 1
    assert Direction.MINIMIZE.sign == -1


def test_direction_values_are_human_readable() -> None:
    # String values keep the direction serializable for configs and logs.
    assert Direction.MAXIMIZE.value == "maximize"
    assert Direction.MINIMIZE.value == "minimize"


# --- Representation ------------------------------------------------------------


def test_representation_covers_the_three_genome_families() -> None:
    assert {member.value for member in Representation} == {"permutation", "binary", "real"}


# --- Problem: abstractness -----------------------------------------------------


def test_problem_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Problem()  # type: ignore[abstract]


def test_subclass_missing_required_methods_is_abstract() -> None:
    class _Incomplete(Problem[list[int]]):
        name = "incomplete"
        direction = Direction.MAXIMIZE
        representation = Representation.BINARY

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


# --- Problem: metadata ---------------------------------------------------------


def test_problem_exposes_its_metadata() -> None:
    problem = _MaxProblem(size=3)
    assert isinstance(problem.name, str)
    assert problem.name
    assert problem.direction is Direction.MAXIMIZE
    assert problem.representation is Representation.BINARY


# --- Problem: g(x) ties direction to quality -----------------------------------


def test_g_matches_the_quality_transform() -> None:
    problem = _MaxProblem(size=4)
    individual = [1, 1, 0, 1]
    assert problem.g(individual) == quality(problem.evaluate(individual), problem.direction)


def test_g_increases_with_quality_for_maximization() -> None:
    problem = _MaxProblem(size=4)
    better = [1, 1, 1, 0]
    worse = [1, 0, 0, 0]
    assert problem.evaluate(better) > problem.evaluate(worse)
    assert problem.g(better) > problem.g(worse)


def test_g_increases_with_quality_for_minimization() -> None:
    problem = _MinProblem(dim=2)
    better = [0.0, 0.0]
    worse = [0.5, 0.5]
    # The better solution has the lower objective yet the higher quality: g(x)
    # inverts the sign so "more is better" holds across both directions.
    assert problem.evaluate(better) < problem.evaluate(worse)
    assert problem.g(better) > problem.g(worse)


# --- Problem: evaluate is total ------------------------------------------------


def test_evaluate_returns_a_finite_float() -> None:
    problem = _MinProblem(dim=4)
    value = problem.evaluate(problem.initialize(np.random.default_rng(0)))
    assert isinstance(value, float)
    assert math.isfinite(value)


# --- Problem: repair (legalization) --------------------------------------------


def test_repair_defaults_to_identity() -> None:
    problem = _MaxProblem(size=4)
    individual = [1, 0, 1, 1]
    assert problem.repair(individual) == individual


def test_repair_legalizes_an_out_of_bounds_genome() -> None:
    problem = _MinProblem(dim=3, low=-1.0, high=1.0)
    assert problem.repair([2.0, -3.0, 0.5]) == [1.0, -1.0, 0.5]


def test_repair_is_idempotent() -> None:
    problem = _MinProblem(dim=3, low=-1.0, high=1.0)
    once = problem.repair([2.0, -3.0, 0.5])
    assert problem.repair(once) == once


# --- Problem: initialize is seeded and legal -----------------------------------


def test_initialize_is_deterministic_for_the_same_generator_seed() -> None:
    problem = _MinProblem(dim=5)
    first = problem.initialize(np.random.default_rng(123))
    second = problem.initialize(np.random.default_rng(123))
    assert first == second


def test_initialize_differs_for_different_generator_seeds() -> None:
    problem = _MinProblem(dim=5)
    assert problem.initialize(np.random.default_rng(1)) != problem.initialize(
        np.random.default_rng(2)
    )


def test_initialize_produces_a_legal_individual() -> None:
    problem = _MinProblem(dim=6, low=-1.0, high=1.0)
    individual = problem.initialize(np.random.default_rng(0))
    assert len(individual) == problem.dim
    assert all(problem.low <= value <= problem.high for value in individual)


def test_initialize_uses_only_the_injected_generator() -> None:
    # Drawing from the injected Generator must not touch NumPy's global state,
    # keeping initialization reproducible from the run seed alone.
    problem = _MaxProblem(size=8)
    before = pickle.dumps(np.random.get_state())
    problem.initialize(np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before
