"""Contract tests for the continuous benchmark test problem (real representation).

These pin the public API of :mod:`experiments.problems.continuous`: the
``BenchmarkFunction`` spec that carries a function's analytic form, domain and
optimum coordinate, and the :class:`ContinuousProblem` that wraps one function at a
fixed dimension. Fitness is the raw function value (a minimization objective), the
seeded ``initialize`` samples uniformly over the domain, and -- unlike the TSP and
knapsack problems, whose ``repair`` is the inherited identity -- ``repair`` here
box-clips every coordinate back into ``[lower, upper]`` (the per-coordinate clipping
rule). Every asserted number is hand-computable: integer
vectors make the transcendental terms collapse (``cos(2*pi*k) == 1``), and the three
functions are pinned against an independent reference formula on sampled points.

The module these tests import does not exist yet: this file is the executable
specification. Expected public names: ``BenchmarkFunction``, ``ContinuousProblem``,
``SPHERE``, ``RASTRIGIN``, ``ROSENBROCK``.

Key facts pinned:
- direction MINIMIZE, representation REAL, genome ``list[float]``;
- three benchmarks -- Sphere/Rastrigin on ``[-5.12, 5.12]`` with optimum at 0,
  Rosenbrock on ``[-2.048, 2.048]`` with optimum at 1 -- all with ``f(x) >= 0`` and
  ``f(optimum) == 0`` exactly;
- ``name == f"{function.name}_d{dimension}"``; ``__init__`` rejects ``dimension < 1``;
- ``initialize`` draws only from the injected generator, in-domain and deterministic;
- ``repair`` box-clips into the domain, returning a new list without mutating input.
"""

from __future__ import annotations

import dataclasses
import math
import pickle
from collections.abc import Sequence

import numpy as np
import pytest

from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation
from experiments.problems.continuous import (
    RASTRIGIN,
    ROSENBROCK,
    SPHERE,
    BenchmarkFunction,
    ContinuousProblem,
)


def _problem(function: BenchmarkFunction, dimension: int) -> ContinuousProblem:
    return ContinuousProblem(function, dimension)


# Independent reference implementations of the three formulas,
# used to pin evaluate against transcription errors on sampled points.


def _ref_sphere(x: Sequence[float]) -> float:
    return float(sum(xi**2 for xi in x))


def _ref_rastrigin(x: Sequence[float]) -> float:
    return 10.0 * len(x) + sum(xi**2 - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x)


def _ref_rosenbrock(x: Sequence[float]) -> float:
    return float(
        sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))
    )


# --- BenchmarkFunction spec ----------------------------------------------------


def test_sphere_spec_fields() -> None:
    assert SPHERE.name == "sphere"
    assert SPHERE.lower == -5.12
    assert SPHERE.upper == 5.12
    assert SPHERE.optimum_coordinate == 0.0


def test_rastrigin_spec_fields() -> None:
    assert RASTRIGIN.name == "rastrigin"
    assert RASTRIGIN.lower == -5.12
    assert RASTRIGIN.upper == 5.12
    assert RASTRIGIN.optimum_coordinate == 0.0


def test_rosenbrock_spec_fields() -> None:
    assert ROSENBROCK.name == "rosenbrock"
    assert ROSENBROCK.lower == -2.048
    assert ROSENBROCK.upper == 2.048
    assert ROSENBROCK.optimum_coordinate == 1.0


def test_benchmark_function_is_frozen() -> None:
    # The spec is an immutable value object: rebinding a field must raise.
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPHERE.lower = 0.0  # type: ignore[misc]


# --- evaluate: hand-computable anchors -----------------------------------------


def test_sphere_formula() -> None:
    # Pure sum of squares -- exact in IEEE, no transcendental terms.
    assert SPHERE.evaluate([3.0, 4.0]) == 25.0
    assert SPHERE.evaluate([1.0, 1.0, 1.0, 1.0, 1.0]) == 5.0
    assert SPHERE.evaluate([0.0, 0.0, 0.0]) == 0.0


def test_rastrigin_formula() -> None:
    # cos(2*pi*k) == 1 for integer k, so integer vectors give exact-looking values;
    # asserted via approx to stay robust to the cosine's last-bit rounding.
    assert RASTRIGIN.evaluate([1.0, 2.0]) == pytest.approx(5.0)  # 20 + (1-10) + (4-10)
    assert RASTRIGIN.evaluate([1.0, 2.0, 3.0]) == pytest.approx(14.0)  # 30 - 9 - 6 - 1
    assert RASTRIGIN.evaluate([0.5]) == pytest.approx(20.25)  # cos(pi) == -1


def test_rosenbrock_formula() -> None:
    # Polynomial in x -- exact in IEEE.
    assert ROSENBROCK.evaluate([2.0, 3.0]) == 101.0  # 100*(3-4)^2 + (1-2)^2
    assert ROSENBROCK.evaluate([0.0, 0.0]) == 1.0  # 100*0 + (1-0)^2
    assert ROSENBROCK.evaluate([1.0, 1.0]) == 0.0  # optimum for d = 2
    assert ROSENBROCK.evaluate([1.0, 1.0, 1.0]) == 0.0  # both terms vanish


def test_evaluate_matches_reference_formula() -> None:
    # Agreement on sampled in-domain points pins the exact formula -- notably the
    # sign of the cosine term and Rosenbrock's (d - 1) term count.
    references = {
        "sphere": _ref_sphere,
        "rastrigin": _ref_rastrigin,
        "rosenbrock": _ref_rosenbrock,
    }
    for function in (SPHERE, RASTRIGIN, ROSENBROCK):
        ref = references[function.name]
        rng = np.random.default_rng(7)
        for _ in range(100):
            x = list(rng.uniform(function.lower, function.upper, size=6))
            assert function.evaluate(x) == pytest.approx(ref(x))


# --- metadata ------------------------------------------------------------------


def test_exposes_metadata() -> None:
    problem = _problem(RASTRIGIN, 10)
    assert problem.name == "rastrigin_d10"
    assert problem.direction is Direction.MINIMIZE
    assert problem.representation is Representation.REAL
    assert problem.dimension == 10
    assert problem.lower == RASTRIGIN.lower
    assert problem.upper == RASTRIGIN.upper


def test_name_encodes_function_and_dimension() -> None:
    assert _problem(SPHERE, 5).name == "sphere_d5"
    assert _problem(RASTRIGIN, 10).name == "rastrigin_d10"
    assert _problem(ROSENBROCK, 5).name == "rosenbrock_d5"


def test_is_a_problem_instance() -> None:
    assert isinstance(_problem(SPHERE, 5), Problem)


# --- __init__ validation -------------------------------------------------------


def test_init_rejects_nonpositive_dimension() -> None:
    for bad in (0, -1, -10):
        with pytest.raises(ValueError):
            ContinuousProblem(SPHERE, bad)


def test_init_accepts_dimension_one() -> None:
    # d = 1 is the boundary of the valid range; the experiment uses d in {5, 10}.
    assert ContinuousProblem(SPHERE, 1).dimension == 1


# --- evaluate: ContinuousProblem -----------------------------------------------


def test_continuous_problem_evaluate_delegates_to_function() -> None:
    assert _problem(SPHERE, 2).evaluate([3.0, 4.0]) == 25.0
    assert _problem(ROSENBROCK, 2).evaluate([2.0, 3.0]) == 101.0
    assert _problem(RASTRIGIN, 2).evaluate([1.0, 2.0]) == pytest.approx(5.0)


def test_evaluate_returns_a_float() -> None:
    assert isinstance(_problem(SPHERE, 2).evaluate([1.0, 2.0]), float)


def test_evaluate_is_nonnegative_on_the_domain() -> None:
    # f(x) >= 0 for all three functions; the optimum value 0 is the floor.
    for function in (SPHERE, RASTRIGIN, ROSENBROCK):
        problem = ContinuousProblem(function, 5)
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = list(rng.uniform(function.lower, function.upper, size=5))
            assert problem.evaluate(x) >= 0.0


def test_evaluate_is_finite_outside_the_domain() -> None:
    # evaluate is pure arithmetic and never raises, even off the clipped domain.
    value = _problem(RASTRIGIN, 3).evaluate([100.0, -100.0, 50.0])
    assert math.isfinite(value)


# --- optimum property ----------------------------------------------------------


def test_optimum_is_the_optimum_coordinate_repeated() -> None:
    assert _problem(SPHERE, 5).optimum == [0.0] * 5
    assert _problem(RASTRIGIN, 3).optimum == [0.0] * 3
    assert _problem(ROSENBROCK, 4).optimum == [1.0] * 4


def test_optimum_has_dimension_length_of_floats() -> None:
    opt = _problem(ROSENBROCK, 7).optimum
    assert isinstance(opt, list)
    assert len(opt) == 7
    assert all(isinstance(v, float) for v in opt)


def test_evaluate_at_optimum_is_exactly_zero() -> None:
    # By construction the global optimum value is 0, and it is exact in IEEE:
    # 0^2 == 0, cos(0) == 1 (so 10d - 10d == 0), (1 - 1)^2 == 0.
    for function in (SPHERE, RASTRIGIN, ROSENBROCK):
        for d in (5, 10):
            problem = ContinuousProblem(function, d)
            assert problem.evaluate(problem.optimum) == 0.0


# --- quality g(x) --------------------------------------------------------------


def test_g_equals_negative_objective_for_minimization() -> None:
    problem = _problem(SPHERE, 2)
    for x in ([3.0, 4.0], [1.0, 1.0], [0.0, 0.0]):
        assert problem.g(x) == -problem.evaluate(x)


def test_lower_objective_has_higher_quality() -> None:
    problem = _problem(SPHERE, 2)
    assert problem.evaluate([1.0, 1.0]) < problem.evaluate([3.0, 4.0])  # 2 < 25
    assert problem.g([1.0, 1.0]) > problem.g([3.0, 4.0])


def test_g_at_optimum_is_zero() -> None:
    problem = _problem(SPHERE, 5)
    assert problem.g(problem.optimum) == 0.0  # g = -0.0, which equals 0.0


# --- initialize ----------------------------------------------------------------


def test_initialize_has_dimension_length() -> None:
    genome = _problem(SPHERE, 8).initialize(np.random.default_rng(0))
    assert len(genome) == 8


def test_initialize_returns_a_list_of_float() -> None:
    genome = _problem(SPHERE, 8).initialize(np.random.default_rng(0))
    assert isinstance(genome, list)
    assert all(isinstance(v, float) for v in genome)


def test_initialize_samples_within_the_domain() -> None:
    # A wide draw makes the range check meaningful; each function uses its own domain.
    for function in (SPHERE, ROSENBROCK):
        genome = ContinuousProblem(function, 100).initialize(np.random.default_rng(1))
        assert all(function.lower <= v <= function.upper for v in genome)


def test_initialize_is_deterministic_for_the_same_seed() -> None:
    problem = _problem(RASTRIGIN, 16)
    assert problem.initialize(np.random.default_rng(42)) == problem.initialize(
        np.random.default_rng(42)
    )


def test_initialize_differs_for_different_seeds() -> None:
    problem = _problem(SPHERE, 16)
    assert problem.initialize(np.random.default_rng(1)) != problem.initialize(
        np.random.default_rng(2)
    )


def test_initialize_uses_only_the_injected_generator() -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    problem = _problem(SPHERE, 16)
    before = pickle.dumps(np.random.get_state())
    problem.initialize(np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


# --- repair: box-clip (the first problem whose repair is not the identity) ------


def test_repair_leaves_in_domain_genome_value_equal() -> None:
    problem = _problem(SPHERE, 5)  # domain [-5.12, 5.12]
    x = [0.0, 1.5, -3.2, 5.12, -5.12]
    assert problem.repair(x) == x


def test_repair_clips_each_coordinate_to_the_nearest_bound() -> None:
    problem = _problem(SPHERE, 5)
    clipped = problem.repair([10.0, -10.0, 0.0, 5.12, -5.12])
    assert clipped == [5.12, -5.12, 0.0, 5.12, -5.12]


def test_repair_keeps_boundary_values() -> None:
    problem = _problem(SPHERE, 2)
    assert problem.repair([-5.12, 5.12]) == [-5.12, 5.12]


def test_repair_uses_the_function_domain() -> None:
    # Rosenbrock's tighter domain clips where Sphere's would not.
    problem = _problem(ROSENBROCK, 2)  # domain [-2.048, 2.048]
    assert problem.repair([5.0, -5.0]) == [2.048, -2.048]


def test_repair_returns_a_new_list_without_mutating_input() -> None:
    problem = _problem(SPHERE, 3)
    x = [10.0, -10.0, 0.0]
    result = problem.repair(x)
    assert result is not x
    assert x == [10.0, -10.0, 0.0]  # input left untouched (no aliasing)


def test_repair_is_idempotent() -> None:
    problem = _problem(SPHERE, 3)
    once = problem.repair([10.0, -10.0, 0.0])
    assert problem.repair(once) == once


def test_repair_preserves_length() -> None:
    problem = _problem(SPHERE, 4)
    assert len(problem.repair([10.0, -10.0, 1.0, 2.0])) == 4


def test_repair_returns_a_list_of_float() -> None:
    problem = _problem(SPHERE, 3)
    result = problem.repair([10.0, -10.0, 0.0])
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)
