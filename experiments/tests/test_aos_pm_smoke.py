"""End-to-end smoke test for the Probability Matching slice on TSP.

Composes the already-implemented pieces -- the generational engine ``run``, the
``AdaptiveOperatorStep`` over the full permutation pool, ``ProbabilityMatching`` with
the a priori parameters, instant-reward credit, the ``TSPProblem`` over a checksummed
TSPLIB instance and the study's repetition seeds -- into one reproducible adaptive
run. It pins the closure criteria of the iteration rather than the strategy's solution
quality: the run improves on its initial population, its best-quality history is
non-decreasing under elitism, a fixed seed reproduces the run bit-for-bit, and the
strategy demonstrably learned -- its estimates left their uniform start while the
selection distribution stayed a valid one with every arm above the exploration floor.

No comparison against Random or the fixed-best operator is made here: the levels
(ceiling, floor, adaptive middle) are assembled from the versioned sweep artifacts in
the validation phase, not from a smoke run on one instance.

Configuration mirrors the CGA-on-TSP smoke so the two are directly comparable: eil51,
N=50, G=50, tournament k=3 and one elite (the engine defaults), one evaluation per
child, so reproduction events equal T_AOS = (N-1)*G. The full pool is used because it
is the harder case for a learning strategy at a short budget -- six arms to tell apart
in 2450 steps.
"""

from __future__ import annotations

import pytest

from aos_ga.aos.probability_matching import ProbabilityMatching
from aos_ga.core.engine import RunResult, run
from aos_ga.core.representation import Representation
from aos_ga.rng import run_generator
from aos_ga.variation.adaptive_operator import AdaptiveOperatorStep
from experiments.configs.pools import PoolVariant, build_pool, pool_ids
from experiments.datasets.seeds import load_repetition_seeds
from experiments.datasets.tsplib import load_instance, load_manifest
from experiments.problems.tsp import TSPProblem

_INSTANCE = "eil51"
_POPULATION_SIZE = 50
_GENERATIONS = 50
_SMOKE_SEED_COUNT = 3
_REPRESENTATION = Representation.PERMUTATION
_VARIANT = PoolVariant.FULL

# The a priori parameters from the strategy table, restated here so a silent change of
# the defaults shows up as a failing smoke test rather than as shifted results.
_ALPHA = 0.1
_P_MIN = 0.05

# Loose wiring guard, not a quality claim: an adaptive GA at this budget stays well
# within 2.5x the optimum, while a unit or optimum mix-up would blow past it.
_GAP_UPPER_BOUND = 1.5


def _strategy() -> ProbabilityMatching:
    """A fresh Probability Matching over the full permutation pool's arms."""
    return ProbabilityMatching(pool_ids(_REPRESENTATION, _VARIANT), alpha=_ALPHA, p_min=_P_MIN)


def _run(problem: TSPProblem, strategy: ProbabilityMatching, seed: int) -> RunResult[list[int]]:
    """One reproducible adaptive run on ``problem`` from ``seed`` (fresh generator, defaults)."""
    step = AdaptiveOperatorStep(build_pool(_REPRESENTATION, _VARIANT), strategy)

    return run(
        problem,
        step,
        run_generator(seed),
        population_size=_POPULATION_SIZE,
        generations=_GENERATIONS,
    )


def _probabilities(strategy: ProbabilityMatching) -> dict[str, float]:
    """The strategy's current selection distribution, typed."""
    probabilities = strategy.snapshot_state()["probabilities"]
    assert isinstance(probabilities, dict)
    return {str(arm): float(value) for arm, value in probabilities.items()}


def _estimates(strategy: ProbabilityMatching) -> dict[str, float]:
    """The strategy's current quality estimates, typed."""
    estimates = strategy.snapshot_state()["quality_estimates"]
    assert isinstance(estimates, dict)
    return {str(arm): float(value) for arm, value in estimates.items()}


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
def runs(
    problem: TSPProblem, seeds: list[int]
) -> list[tuple[ProbabilityMatching, RunResult[list[int]]]]:
    """One adaptive run per smoke seed, each with its OWN strategy, evolved once.

    A fresh strategy per seed is the point, not an accident: the estimates are run
    state, so sharing one instance across seeds would let run *n* start from what runs
    *1..n-1* learned. Each pair keeps the strategy that drove its run, so the learned
    state can be inspected afterwards.
    """
    pairs = []
    for seed in seeds:
        strategy = _strategy()
        pairs.append((strategy, _run(problem, strategy, seed)))
    return pairs


# --- history shape ---------------------------------------------------------------


def test_history_has_one_entry_per_generation_plus_the_start(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
) -> None:
    for _, result in runs:
        assert len(result.best_quality_history) == _GENERATIONS + 1


# --- convergence -----------------------------------------------------------------


def test_run_improves_on_its_initial_population(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]], seeds: list[int]
) -> None:
    # Relative signal, no magic threshold: the final best quality g must exceed the
    # best of the initial population (g is more-is-better, so a shorter tour).
    for seed, (_, result) in zip(seeds, runs, strict=True):
        history = result.best_quality_history
        assert history[-1] > history[0], f"no improvement for seed {seed}"


def test_best_quality_history_is_non_decreasing(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]], seeds: list[int]
) -> None:
    # Elitism guarantees the best g never regresses across generations.
    for seed, (_, result) in zip(seeds, runs, strict=True):
        history = result.best_quality_history
        assert all(a <= b for a, b in zip(history, history[1:], strict=False)), (
            f"best quality regressed for seed {seed}"
        )


# --- reproducibility -------------------------------------------------------------


def test_run_is_reproducible_for_a_fixed_seed(
    problem: TSPProblem,
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
    seeds: list[int],
) -> None:
    # The whole adaptive path -- operator selection, parent draws, operator internals --
    # replays from one seed, so results stay reproducible despite the added learning.
    first = runs[0][1]
    second = _run(problem, _strategy(), seeds[0])
    assert second.best_objective == first.best_objective
    assert second.best_quality_history == first.best_quality_history
    assert second.best == first.best


def test_distinct_seeds_yield_distinct_runs(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
) -> None:
    histories = [result.best_quality_history for _, result in runs]
    assert any(histories[0] != other for other in histories[1:])


# --- the strategy actually learned ------------------------------------------------


def test_estimates_leave_their_uniform_start(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]], seeds: list[int]
) -> None:
    # The credit path is wired end to end only if rewards moved the arms apart: this is
    # what distinguishes the adaptive step from the Random baseline it starts out as.
    for seed, (strategy, _) in zip(seeds, runs, strict=True):
        estimates = _estimates(strategy)
        assert len(set(estimates.values())) > 1, f"estimates stayed uniform for seed {seed}"


def test_selection_stays_a_valid_distribution(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
) -> None:
    for strategy, _ in runs:
        probabilities = _probabilities(strategy)
        assert set(probabilities) == set(pool_ids(_REPRESENTATION, _VARIANT))
        assert sum(probabilities.values()) == pytest.approx(1.0)


def test_no_arm_is_driven_below_the_exploration_floor(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
) -> None:
    # Even after 2450 AOS steps the weakest operator keeps its p_min share, which is
    # what makes a short run recoverable from an unlucky start.
    for strategy, _ in runs:
        assert min(_probabilities(strategy).values()) >= _P_MIN


# --- gap-to-optimum (metric wiring guard) -----------------------------------------


def test_gap_to_optimum_is_non_negative_and_within_a_loose_bound(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
    optimum: int,
    seeds: list[int],
) -> None:
    for seed, (_, result) in zip(seeds, runs, strict=True):
        gap = (result.best_objective - optimum) / optimum
        assert gap >= 0.0, f"gap below zero (beats the optimum?) for seed {seed}: {gap}"
        assert gap < _GAP_UPPER_BOUND, f"gap unexpectedly large for seed {seed}: {gap}"


# --- budget accounting -------------------------------------------------------------


def test_reproduction_events_match_the_generation_budget(
    runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]],
) -> None:
    # One elite per generation, one evaluation per child: T_AOS = (N-1)*G. The adaptive
    # step spends exactly the same budget as the canonical pipeline and the baselines.
    expected_events = (_POPULATION_SIZE - 1) * _GENERATIONS
    for _, result in runs:
        assert result.reproduction_events == expected_events
        assert result.evaluations == _POPULATION_SIZE + expected_events


# --- legality (end-to-end) ---------------------------------------------------------


def test_best_is_a_valid_permutation(
    problem: TSPProblem, runs: list[tuple[ProbabilityMatching, RunResult[list[int]]]]
) -> None:
    # Six operators of mixed arity, all of them driven by the strategy, must still leave
    # a legal tour behind.
    for _, result in runs:
        assert sorted(result.best) == list(range(problem.dimension))
