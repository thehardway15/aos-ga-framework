"""Problem families for the experiment sweeps: instances and how to build each one.

A :class:`FamilyDescriptor` bundles one problem family's instances with the callables
that turn an instance handle into a :class:`~aos_ga.core.problem.Problem` and into the
real-valued pool's domain bounds. The three families (TSP, knapsack, continuous) share
this one schema, so every sweep -- the single-operator reference, the random-selection
baseline and later the adaptive strategies -- iterates the same ``FAMILIES`` tuple
instead of re-deriving the instance list. ``problem`` names the family and an instance's
``instance_id`` is the problem's own ``name``; the continuous family folds a
``(function, dimension)`` pair into that single id (``"sphere_d5"``).

Pool bounds are per instance, not per family: Rosenbrock's domain differs from Sphere's,
and the gaussian/polynomial scaling in :func:`experiments.configs.pools.build_pool`
depends on the domain width, so the descriptor exposes ``pool_bounds`` as a callable on
the instance handle. The discrete families return ``None`` bounds.

This is experiment configuration alongside :mod:`experiments.configs.pools` and the grid
constants in :mod:`experiments.configs`. Import the names from this submodule directly;
never re-export them from ``experiments.configs.__init__``, which would close an import
cycle (``configs`` is already a dependency of ``experiments.datasets``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aos_ga.core.problem import Problem
from aos_ga.core.representation import Representation

from ..datasets.knapsack import load_instance as load_knapsack_instance
from ..datasets.tsplib import load_instance as load_tsp_instance
from ..problems.continuous import (
    RASTRIGIN,
    ROSENBROCK,
    SPHERE,
    BenchmarkFunction,
    ContinuousProblem,
)
from ..problems.knapsack import KnapsackProblem
from ..problems.tsp import TSPProblem

ContinuousSpec = tuple[BenchmarkFunction, int]


@dataclass(frozen=True)
class FamilyDescriptor:
    """One problem family: its instances and how to build each problem and its pool bounds.

    ``specs`` are opaque per-family instance handles (an id string, or a
    ``(function, dimension)`` pair); ``build_problem`` turns one into a problem and
    ``pool_bounds`` gives the real-valued pool's domain bounds (``None`` for the discrete
    families). The handle type is erased to ``Any`` here so the three families share one
    tuple; each builder below keeps its own precise type.
    """

    problem: str
    representation: Representation
    specs: tuple[Any, ...]
    build_problem: Callable[[Any], Problem[Any]]
    pool_bounds: Callable[[Any], tuple[float, float] | None]


def _build_tsp(spec: str) -> Problem[list[int]]:
    """Build the TSP problem for instance id ``spec``."""
    return TSPProblem(load_tsp_instance(spec))


def _build_knapsack(spec: str) -> Problem[list[int]]:
    """Build the knapsack problem for instance id ``spec``."""
    return KnapsackProblem(load_knapsack_instance(spec))


def _build_continuous(spec: ContinuousSpec) -> Problem[list[float]]:
    """Build the continuous problem for the ``(function, dimension)`` ``spec``."""
    benchmark_function, dimension = spec
    return ContinuousProblem(benchmark_function, dimension)


def _continuous_bounds(spec: ContinuousSpec) -> tuple[float, float]:
    """Return the box domain ``(lower, upper)`` of the spec's benchmark function."""
    function, _ = spec
    return (function.lower, function.upper)


_TSP_INSTANCES = ("eil22", "eil51", "berlin52")
_KNAPSACK_INSTANCES = (
    "n20_uncorrelated",
    "n20_weakly",
    "n20_strongly",
    "n30_uncorrelated",
    "n30_weakly",
    "n30_strongly",
    "n50_uncorrelated",
    "n50_weakly",
    "n50_strongly",
)
_CONTINUOUS_SPECS: tuple[ContinuousSpec, ...] = tuple(
    (function, dimension) for function in (SPHERE, RASTRIGIN, ROSENBROCK) for dimension in (5, 10)
)


FAMILIES: tuple[FamilyDescriptor, ...] = (
    FamilyDescriptor("tsp", Representation.PERMUTATION, _TSP_INSTANCES, _build_tsp, lambda _: None),
    FamilyDescriptor(
        "knapsack", Representation.BINARY, _KNAPSACK_INSTANCES, _build_knapsack, lambda _: None
    ),
    FamilyDescriptor(
        "continuous", Representation.REAL, _CONTINUOUS_SPECS, _build_continuous, _continuous_bounds
    ),
)
