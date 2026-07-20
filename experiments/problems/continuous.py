"""Continuous benchmark test problems (real-valued representation).

Wraps one of three classic benchmark functions -- Sphere, Rastrigin, Rosenbrock --
as a :class:`~aos_ga.core.problem.Problem`: a genome is a real vector
(``list[float]``) and its fitness is the raw function value, a minimization
objective. Each function is described by a :class:`BenchmarkFunction` spec that
carries its analytic form, box domain and optimum coordinate. Unlike the TSP and
knapsack problems, whose ``repair`` is the inherited identity, ``repair`` here
box-clips every coordinate back into the domain, so the variation operators need not
know the bounds themselves.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation


@dataclass(frozen=True)
class BenchmarkFunction:
    """An analytic benchmark function: its formula, box domain and optimum.

    ``evaluate`` holds the function ``f(x)`` itself, so each of the three module-level
    constants carries its own formula. The domain ``[lower, upper]`` is identical on
    every axis; the global optimum sits at ``optimum_coordinate`` on each axis (0 for
    Sphere and Rastrigin, 1 for Rosenbrock) and, by construction, has value 0.
    """

    name: str
    evaluate: Callable[[Sequence[float]], float]
    lower: float
    upper: float
    optimum_coordinate: float


def _sphere(x: Sequence[float]) -> float:
    """Sphere ``f(x) = sum(x_i^2)``: unimodal, non-negative, minimum 0 at the origin."""
    return sum(xi**2 for xi in x)


def _rastrigin(x: Sequence[float]) -> float:
    """Rastrigin ``f(x) = 10d + sum(x_i^2 - 10 cos(2 pi x_i))``: multimodal, min 0 at 0.

    Every per-axis term is bounded below by -10 (reached only at 0), so with the +10d
    offset the value is always non-negative.
    """
    return float(10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x))


def _rosenbrock(x: Sequence[float]) -> float:
    """Rosenbrock ``f(x) = sum_i[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]``: min 0 at 1.

    A sum of squares, hence non-negative; the narrow curved valley makes the descent
    direction hard to follow.
    """
    return sum(100 * (x[i + 1] - xi**2) ** 2 + (xi - 1) ** 2 for i, xi in enumerate(x[:-1]))


RASTRIGIN = BenchmarkFunction("rastrigin", _rastrigin, -5.12, 5.12, 0.0)
ROSENBROCK = BenchmarkFunction("rosenbrock", _rosenbrock, -2.048, 2.048, 1.0)
SPHERE = BenchmarkFunction("sphere", _sphere, -5.12, 5.12, 0.0)


class ContinuousProblem(Problem[list[float]]):
    """A benchmark function at a fixed dimension as a minimization problem.

    Pairs a :class:`BenchmarkFunction` with a dimension ``d``: the genome is a real
    vector of length ``d`` and fitness is the function value (lower is better). The
    domain bounds are read from the function and kept feasible by ``repair``.
    """

    direction = Direction.MINIMIZE
    representation = Representation.REAL

    def __init__(self, benchmark_function: BenchmarkFunction, dimension: int):
        """Store the function and dimension; name the problem ``f"{name}_d{dimension}"``.

        Raises ``ValueError`` if ``dimension < 1``.
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")

        self.name = f"{benchmark_function.name}_d{dimension}"
        self.dimension = dimension
        self.function = benchmark_function
        self.lower = benchmark_function.lower
        self.upper = benchmark_function.upper
        self.optimum_coordinate = benchmark_function.optimum_coordinate

    @property
    def optimum(self) -> list[float]:
        """The global optimum: ``optimum_coordinate`` repeated on every axis (value 0)."""
        return [self.optimum_coordinate] * self.dimension

    def repair(self, solution: list[float]) -> list[float]:
        """Box-clip each coordinate into ``[lower, upper]``, returning a new list.

        The first problem whose ``repair`` is not the identity: variation may push a
        coordinate out of the domain, and clipping to the nearest bound keeps every
        solution feasible without the operators knowing the bounds.
        """
        return [float(x) for x in np.clip(solution, self.lower, self.upper)]

    def evaluate(self, x: list[float]) -> float:
        """Raw function value ``f(x)``, the minimization objective; never raises."""
        return self.function.evaluate(x)

    def initialize(self, rng: Generator) -> list[float]:
        """Sample each coordinate uniformly over ``[lower, upper]`` using only ``rng``."""
        return [float(v) for v in rng.uniform(self.lower, self.upper, size=self.dimension)]
