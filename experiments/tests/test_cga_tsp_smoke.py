"""End-to-end smoke test for the canonical GA slice on TSP.

This composes the already-implemented pieces -- the generational engine ``run``,
the ``CanonicalPipeline`` (OX ``p_c=0.9`` then inversion ``p_m=0.1``), the
``TSPProblem`` over a checksummed TSPLIB instance and the study's repetition
seeds -- into one reproducible run and validates the slice through gap-to-optimum,
where ``gap = (best_objective - optimum) / optimum`` with ``best_objective`` already
in tour-length units. It pins the Iteration-1 closure criteria rather than the GA's
solution quality: the run improves on its initial population, its best-quality
history is non-decreasing under elitism, and a fixed seed reproduces the run
bit-for-bit. The absolute gap is only bounded loosely, as a wiring guard -- a
canonical GA at this budget stays well short of the optimum by design, so no tight
threshold is asserted (that belongs in the separate reporting script).

Configuration mirrors the methodology's CGA-on-TSP reference: eil51, N=50, G=50,
tournament k=3 and one elite (the engine defaults), one evaluation per child, so
reproduction events equal T_AOS = (N-1)*G. The known-tour length sanity
(evaluate(optimal_tour) == 426) is already covered by the dataset integrity test,
so the optimum is read from the manifest here rather than re-derived.
"""

from __future__ import annotations

import pytest

from aos_ga.core.engine import RunResult, run
from aos_ga.operators.permutation import OrderCrossover, SegmentInversion
from aos_ga.rng import run_generator
from aos_ga.variation.canonical import CanonicalPipeline
from experiments.datasets.seeds import load_repetition_seeds
from experiments.datasets.tsplib import load_instance, load_manifest
from experiments.problems.tsp import TSPProblem

# The methodology's CGA-on-TSP reference at its cheapest meaningful point: one
# instance with a real TSPLIB optimum, the larger population and the largest
# budget, over a few of the shared repetition seeds. tournament_k=3 and one elite
# are the engine defaults, so they are not passed here.
_INSTANCE = "eil51"
_POPULATION_SIZE = 50
_GENERATIONS = 50
_SMOKE_SEED_COUNT = 3

# Loose wiring guard, not a quality claim: a canonical GA (OX + inversion, no local
# search) at this budget lands well within 2.5x the optimum, while a unit or optimum
# mix-up would blow past it. Real gap distribution belongs in the reporting script.
_GAP_UPPER_BOUND = 1.5


def _pipeline() -> CanonicalPipeline[list[int]]:
    """Build the fixed CGA-on-TSP variation step: OX ``p_c=0.9`` then inversion ``p_m=0.1``."""
    return CanonicalPipeline(OrderCrossover(), 0.9, SegmentInversion(), 0.1)


def _run(problem: TSPProblem, seed: int) -> RunResult[list[int]]:
    """One reproducible CGA run on ``problem`` from ``seed`` (fresh generator, engine defaults)."""
    return run(
        problem,
        _pipeline(),
        run_generator(seed),
        population_size=_POPULATION_SIZE,
        generations=_GENERATIONS,
    )


@pytest.fixture(scope="module")
def problem() -> TSPProblem:
    """The eil51 TSP problem, loaded from the checksummed dataset (built once)."""
    return TSPProblem(load_instance(_INSTANCE))


@pytest.fixture(scope="module")
def optimum() -> int:
    """The instance's optimal tour length, from the manifest (single source of truth)."""
    entry = next(e for e in load_manifest() if e.instance_id == _INSTANCE)
    return entry.optimal_length


@pytest.fixture(scope="module")
def seeds() -> list[int]:
    """The first few shared repetition seeds used for the smoke run."""
    return load_repetition_seeds()[:_SMOKE_SEED_COUNT]


@pytest.fixture(scope="module")
def results(problem: TSPProblem, seeds: list[int]) -> list[RunResult[list[int]]]:
    """One CGA run per smoke seed, evolved once and shared across the assertions."""
    return [_run(problem, seed) for seed in seeds]


# --- history shape -------------------------------------------------------------


def test_history_has_one_entry_per_generation_plus_the_start(
    results: list[RunResult[list[int]]],
) -> None:
    for result in results:
        assert len(result.best_quality_history) == _GENERATIONS + 1


# --- convergence (Iteration-1 closure) -----------------------------------------


def test_run_improves_on_its_initial_population(
    results: list[RunResult[list[int]]], seeds: list[int]
) -> None:
    # Relative signal, no magic threshold: the final best quality g must exceed the
    # best of the initial population (g is more-is-better, so a shorter tour).
    for seed, result in zip(seeds, results, strict=True):
        history = result.best_quality_history
        assert history[-1] > history[0], f"no improvement for seed {seed}"


def test_best_quality_history_is_non_decreasing(
    results: list[RunResult[list[int]]], seeds: list[int]
) -> None:
    # Elitism guarantees the best g never regresses across generations.
    for seed, result in zip(seeds, results, strict=True):
        history = result.best_quality_history
        assert all(a <= b for a, b in zip(history, history[1:], strict=False)), (
            f"best quality regressed for seed {seed}"
        )


# --- reproducibility -----------------------------------------------------------


def test_run_is_reproducible_for_a_fixed_seed(
    problem: TSPProblem, results: list[RunResult[list[int]]], seeds: list[int]
) -> None:
    first = results[0]
    second = _run(problem, seeds[0])
    assert second.best_objective == first.best_objective
    assert second.best_quality_history == first.best_quality_history
    assert second.best == first.best


def test_distinct_seeds_yield_distinct_runs(
    results: list[RunResult[list[int]]],
) -> None:
    # Different seeds must drive different searches, else the run ignores its seed.
    histories = [result.best_quality_history for result in results]
    assert any(histories[0] != other for other in histories[1:])


# --- gap-to-optimum (metric wiring guard) --------------------------------------


def test_gap_to_optimum_is_non_negative_and_within_a_loose_bound(
    results: list[RunResult[list[int]]], optimum: int, seeds: list[int]
) -> None:
    for seed, result in zip(seeds, results, strict=True):
        gap = (result.best_objective - optimum) / optimum
        assert gap >= 0.0, f"gap below zero (beats the optimum?) for seed {seed}: {gap}"
        assert gap < _GAP_UPPER_BOUND, f"gap unexpectedly large for seed {seed}: {gap}"


# --- budget accounting ---------------------------------------------------------


def test_reproduction_events_match_the_generation_budget(
    results: list[RunResult[list[int]]],
) -> None:
    # One elite per generation, one evaluation per child: T_AOS = (N-1)*G.
    expected_events = (_POPULATION_SIZE - 1) * _GENERATIONS
    for result in results:
        assert result.reproduction_events == expected_events
        assert result.evaluations == _POPULATION_SIZE + expected_events


# --- legality (end-to-end) -----------------------------------------------------


def test_best_is_a_valid_permutation(
    problem: TSPProblem, results: list[RunResult[list[int]]]
) -> None:
    # The whole variation-then-repair path must still yield a legal tour.
    for result in results:
        assert sorted(result.best) == list(range(problem.dimension))
