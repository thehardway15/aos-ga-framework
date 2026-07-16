"""End-to-end smoke test for the canonical GA slice on the 0/1 knapsack.

This composes the already-implemented pieces -- the generational engine ``run``,
the ``CanonicalPipeline`` (uniform crossover ``p_c=0.9`` then bit-flip ``p_m=1.0``),
the ``KnapsackProblem`` over a checksummed Pisinger instance and the study's
repetition seeds -- into one reproducible run and validates the slice through
gap-to-optimum. Because the knapsack is a maximization problem, the gap is
``gap = (optimum - best_objective) / optimum`` with ``best_objective`` already in
item-value units, so it is positive when the run falls short of the exact optimum.
It pins the step-4/5 closure criteria rather than the GA's solution quality: the
run improves on its initial population, its best-quality history is non-decreasing
under elitism, and a fixed seed reproduces the run bit-for-bit.

The gap is bounded on both sides as a wiring guard, not a quality claim. Two facts
make the bounds tight and non-flaky rather than arbitrary. First, ``f(x) <= optimum``
for every ``x`` -- feasible genomes score their value (at most the exact optimum),
while the big-M penalty (``rho = sum(values) + 1``) drives any infeasible genome
below zero -- so the gap can never go negative; a negative gap would mean a wrong
optimum, instance or sign. Second, the best-of-run is feasible at this budget
(``best_objective >= 0``), which is the binary analog of the TSP slice's
valid-permutation guard and bounds the gap at one from above.

Configuration mirrors the methodology's CGA-on-knapsack reference: uniform
``p_c=0.9`` and bit-flip ``1/n`` per bit, on n50_strongly (the representative,
hardest-of-class binary instance), N=50, G=50, tournament k=3 and one elite (the
engine defaults), one evaluation per child, so reproduction events equal
T_AOS = (N-1)*G. ``p_m=1.0`` because bit-flip already draws its own ``1/n`` per-bit
rate inside the operator; the pipeline probability only triggers it, it does not
scale a per-genome rate. The exact optimum and its selection are cross-checked
against ``evaluate`` by the build test, so the optimum is read from optima.json
here rather than re-derived.
"""

from __future__ import annotations

import pytest

from aos_ga.core.engine import RunResult, run
from aos_ga.operators.binary import BitFlipMutation, UniformCrossover
from aos_ga.rng import run_generator
from aos_ga.variation.canonical import CanonicalPipeline
from experiments.datasets.knapsack import load_instance, load_optima
from experiments.datasets.seeds import load_repetition_seeds
from experiments.problems.knapsack import KnapsackProblem

# The methodology's CGA-on-knapsack reference at its representative point: the
# hardest-of-class binary instance (n=50, strongly correlated -- the sensitivity
# instance) with a known exact optimum, the larger population and the largest
# budget, over a few of the shared repetition seeds. tournament_k=3 and one elite
# are the engine defaults, so they are not passed here.
_INSTANCE = "n50_strongly"
_POPULATION_SIZE = 50
_GENERATIONS = 50
_SMOKE_SEED_COUNT = 3


def _pipeline() -> CanonicalPipeline[list[int]]:
    """Build the CGA-on-knapsack variation step: uniform ``p_c=0.9`` then bit-flip ``p_m=1.0``."""
    return CanonicalPipeline(UniformCrossover(), 0.9, BitFlipMutation(), 1.0)


def _run(problem: KnapsackProblem, seed: int) -> RunResult[list[int]]:
    """One reproducible CGA run on ``problem`` from ``seed`` (fresh generator, engine defaults)."""
    return run(
        problem,
        _pipeline(),
        run_generator(seed),
        population_size=_POPULATION_SIZE,
        generations=_GENERATIONS,
    )


@pytest.fixture(scope="module")
def problem() -> KnapsackProblem:
    """The n50_strongly knapsack problem, loaded from the checksummed dataset (built once)."""
    return KnapsackProblem(load_instance(_INSTANCE))


@pytest.fixture(scope="module")
def optimum() -> int:
    """The instance's exact optimal value, from optima.json (single source of truth)."""
    return next(entry for entry in load_optima() if entry.instance_id == _INSTANCE).optimum


@pytest.fixture(scope="module")
def seeds() -> list[int]:
    """The first few shared repetition seeds used for the smoke run."""
    return load_repetition_seeds()[:_SMOKE_SEED_COUNT]


@pytest.fixture(scope="module")
def results(problem: KnapsackProblem, seeds: list[int]) -> list[RunResult[list[int]]]:
    """One CGA run per smoke seed, evolved once and shared across the assertions."""
    return [_run(problem, seed) for seed in seeds]


# --- history shape -------------------------------------------------------------


def test_history_has_one_entry_per_generation_plus_the_start(
    results: list[RunResult[list[int]]],
) -> None:
    for result in results:
        assert len(result.best_quality_history) == _GENERATIONS + 1


# --- convergence (step-4/5 closure) --------------------------------------------


def test_run_improves_on_its_initial_population(
    results: list[RunResult[list[int]]], seeds: list[int]
) -> None:
    # Relative signal, no magic threshold: the final best quality g must exceed the
    # best of the initial population (g is more-is-better, so a higher-value packing).
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
    problem: KnapsackProblem, results: list[RunResult[list[int]]], seeds: list[int]
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


def test_gap_to_optimum_is_non_negative(
    results: list[RunResult[list[int]]], optimum: int, seeds: list[int]
) -> None:
    # Nothing can beat the exact DP optimum: f(x) <= optimum for every x (infeasible
    # genomes are driven below zero by the big-M penalty), so the gap stays >= 0. A
    # negative gap would mean a wrong optimum, instance or sign.
    for seed, result in zip(seeds, results, strict=True):
        gap = (optimum - result.best_objective) / optimum
        assert gap >= 0.0, f"gap below zero (beats the optimum?) for seed {seed}: {gap}"


def test_best_of_run_is_feasible_and_bounds_the_gap(
    results: list[RunResult[list[int]]], optimum: int, seeds: list[int]
) -> None:
    # The binary analog of the TSP slice's valid-permutation guard: the big-M penalty
    # plus selection must escape infeasibility, so the best-of-run scores a feasible,
    # non-negative value. That in turn bounds the gap at one from above.
    for seed, result in zip(seeds, results, strict=True):
        assert result.best_objective >= 0.0, (
            f"best-of-run is infeasible for seed {seed}: {result.best_objective}"
        )
        gap = (optimum - result.best_objective) / optimum
        assert gap <= 1.0, f"gap above one for a feasible best for seed {seed}: {gap}"


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


def test_best_is_a_valid_bitstring(
    problem: KnapsackProblem, results: list[RunResult[list[int]]]
) -> None:
    # The whole variation-then-repair path must still yield a length-n 0/1 vector.
    for result in results:
        assert len(result.best) == problem.dimension
        assert set(result.best) <= {0, 1}
