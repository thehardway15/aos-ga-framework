"""End-to-end smoke test for the canonical GA slice on the continuous benchmarks.

This composes the already-implemented pieces -- the generational engine ``run``,
the ``CanonicalPipeline`` (SBX ``p_c=0.9`` then polynomial mutation ``p_m=1.0``),
the ``ContinuousProblem`` over an analytic benchmark function and the study's
repetition seeds -- into one reproducible run and validates the slice. It pins the
step-4/5 closure criteria rather than the GA's solution quality: the run improves on
its initial population, its best-quality history is non-decreasing under elitism, and
a fixed seed reproduces the run bit-for-bit.

The metric differs from the TSP and knapsack slices. Those normalize against a
non-zero optimum, but every benchmark here has optimum value 0, so a relative gap
``(optimum - best) / optimum`` divides by zero and does not carry over. Since
``f(x) >= 0`` everywhere and the optimum is 0, the raw ``best_objective`` (the minimal
``f`` found, recovered by the engine as ``min f``) is already the absolute error to the
optimum, so the best fitness is reported directly. The wiring guard is correspondingly
loose: ``best_objective`` must be finite and non-negative. Two facts make this a real
guard rather than a tautology. First, Rastrigin is non-negative by construction (each
per-axis term is bounded below by -10 and cancelled by the +10d offset), so the minimum
found can never dip below zero -- a negative value would mean a sign or objective
mix-up (for a minimization problem ``best_objective = direction.sign * max(g) = min f``,
and an inverted sign would drive it below zero). Second, finiteness catches a NaN or
infinity leaking past repair, since the domain-unaware operators can push a coordinate
out of the box and only the problem's box-clip repair legalizes it.

Configuration mirrors the methodology's CGA-on-continuous reference: SBX ``p_c=0.9``
and polynomial mutation ``1/d`` per variable, on Rastrigin at ``d=10`` (the multimodal
sensitivity instance), N=50, G=50, tournament k=3 and one elite (the engine defaults),
one evaluation per child, so reproduction events equal T_AOS = (N-1)*G. ``p_m=1.0``
because polynomial mutation already draws its own ``1/d`` per-variable rate inside the
operator; the pipeline probability only triggers it, it does not scale a per-genome
rate. Unlike the TSP and knapsack slices there is no dataset or manifest: the benchmark
functions are analytic, so the problem is constructed directly rather than loaded.
"""

from __future__ import annotations

import math

import pytest

from aos_ga.core.engine import RunResult, run
from aos_ga.operators.real import SBX, PolynomialMutation
from aos_ga.rng import run_generator
from aos_ga.variation.canonical import CanonicalPipeline
from experiments.datasets.seeds import load_repetition_seeds
from experiments.problems.continuous import RASTRIGIN, ContinuousProblem

# The methodology's CGA-on-continuous reference at its representative point: the
# multimodal sensitivity function (Rastrigin) at the larger dimension, the larger
# population and the largest budget, over a few of the shared repetition seeds.
# tournament_k=3 and one elite are the engine defaults, so they are not passed here.
_FUNCTION = RASTRIGIN
_DIMENSION = 10
_POPULATION_SIZE = 50
_GENERATIONS = 50
_SMOKE_SEED_COUNT = 3


def _pipeline(span: float) -> CanonicalPipeline[list[float]]:
    """Build the CGA-on-continuous variation step: SBX ``p_c=0.9`` then polynomial ``p_m=1.0``.

    The polynomial step scales with the box width ``span = upper - lower`` of the problem.
    """
    return CanonicalPipeline(SBX(), 0.9, PolynomialMutation(span=span), 1.0)


def _run(problem: ContinuousProblem, seed: int) -> RunResult[list[float]]:
    """One reproducible CGA run on ``problem`` from ``seed`` (fresh generator, engine defaults)."""
    return run(
        problem,
        _pipeline(problem.upper - problem.lower),
        run_generator(seed),
        population_size=_POPULATION_SIZE,
        generations=_GENERATIONS,
    )


@pytest.fixture(scope="module")
def problem() -> ContinuousProblem:
    """The Rastrigin d=10 problem, built directly from the analytic function spec."""
    return ContinuousProblem(_FUNCTION, _DIMENSION)


@pytest.fixture(scope="module")
def seeds() -> list[int]:
    """The first few shared repetition seeds used for the smoke run."""
    return load_repetition_seeds()[:_SMOKE_SEED_COUNT]


@pytest.fixture(scope="module")
def results(problem: ContinuousProblem, seeds: list[int]) -> list[RunResult[list[float]]]:
    """One CGA run per smoke seed, evolved once and shared across the assertions."""
    return [_run(problem, seed) for seed in seeds]


# --- history shape -------------------------------------------------------------


def test_history_has_one_entry_per_generation_plus_the_start(
    results: list[RunResult[list[float]]],
) -> None:
    for result in results:
        assert len(result.best_quality_history) == _GENERATIONS + 1


# --- convergence (step-4/5 closure) --------------------------------------------


def test_run_improves_on_its_initial_population(
    results: list[RunResult[list[float]]], seeds: list[int]
) -> None:
    # Relative signal, no magic threshold: the final best quality g must exceed the
    # best of the initial population (g is more-is-better, so a lower function value).
    for seed, result in zip(seeds, results, strict=True):
        history = result.best_quality_history
        assert history[-1] > history[0], f"no improvement for seed {seed}"


def test_best_quality_history_is_non_decreasing(
    results: list[RunResult[list[float]]], seeds: list[int]
) -> None:
    # Elitism guarantees the best g never regresses across generations.
    for seed, result in zip(seeds, results, strict=True):
        history = result.best_quality_history
        assert all(a <= b for a, b in zip(history, history[1:], strict=False)), (
            f"best quality regressed for seed {seed}"
        )


# --- reproducibility -----------------------------------------------------------


def test_run_is_reproducible_for_a_fixed_seed(
    problem: ContinuousProblem, results: list[RunResult[list[float]]], seeds: list[int]
) -> None:
    first = results[0]
    second = _run(problem, seeds[0])
    assert second.best_objective == first.best_objective
    assert second.best_quality_history == first.best_quality_history
    assert second.best == first.best


def test_distinct_seeds_yield_distinct_runs(
    results: list[RunResult[list[float]]],
) -> None:
    # Different seeds must drive different searches, else the run ignores its seed.
    histories = [result.best_quality_history for result in results]
    assert any(histories[0] != other for other in histories[1:])


# --- best fitness (metric wiring guard) ----------------------------------------


def test_best_objective_is_finite_and_non_negative(
    results: list[RunResult[list[float]]], seeds: list[int]
) -> None:
    # Loose guard in place of a gap: the optimum is 0, so best_objective = min f is the
    # absolute error itself. Rastrigin is non-negative by construction, so a value below
    # zero would mean an inverted objective sign; a non-finite value would mean a NaN or
    # infinity leaked past the box-clip repair.
    for seed, result in zip(seeds, results, strict=True):
        assert math.isfinite(result.best_objective), (
            f"best_objective is not finite for seed {seed}: {result.best_objective}"
        )
        assert result.best_objective >= 0.0, (
            f"best_objective below zero (inverted sign?) for seed {seed}: {result.best_objective}"
        )


# --- budget accounting ---------------------------------------------------------


def test_reproduction_events_match_the_generation_budget(
    results: list[RunResult[list[float]]],
) -> None:
    # One elite per generation, one evaluation per child: T_AOS = (N-1)*G.
    expected_events = (_POPULATION_SIZE - 1) * _GENERATIONS
    for result in results:
        assert result.reproduction_events == expected_events
        assert result.evaluations == _POPULATION_SIZE + expected_events


# --- legality (end-to-end) -----------------------------------------------------


def test_best_is_within_the_box_after_repair(
    problem: ContinuousProblem, results: list[RunResult[list[float]]]
) -> None:
    # The continuous analog of the TSP slice's valid-permutation guard: the whole
    # variation-then-repair path must still yield a length-d vector inside the box.
    for result in results:
        assert len(result.best) == problem.dimension
        assert all(problem.lower <= x <= problem.upper for x in result.best), (
            f"best escaped the box [{problem.lower}, {problem.upper}]: {result.best}"
        )
