"""Contract spec for the GA skeleton: tournament, run loop, run result.

The engine is the shared, invariant part of the genetic algorithm -- generic over
``Problem``/``Genome``, with the variation model factored out to an interchangeable
``VariationStep``. It owns: population initialization and evaluation, tournament
selection (k=3), elitism (one elite occupies one of the N slots, uncopied and
NOT re-evaluated), succession, the generation budget, and evaluating each child
exactly once. Everything decides direction through the unified g (more-is-better),
so a minimization problem (TSP) and a maximization one run through the same code.

The names below are not implemented yet: this file is the executable specification.
Expected public names (in ``aos_ga.core.engine``): ``tournament_select``, ``run``
and ``RunResult``.

Frozen contract (skeleton):
- ``tournament_select(qualities, k, rng) -> index`` draws k competitors via
  ``rng.integers(0, len(qualities), size=k)`` (with replacement) and returns the
  index of the highest-quality competitor; ties go to the earliest one sampled.
  Rejects ``k < 1`` with ``ValueError``.
- ``run(problem, variation, rng, *, population_size N, generations G,
  tournament_k=3, elite_count=1)`` executes:
    1. init: call ``problem.initialize(rng)`` N times, then score each with
       ``problem.g`` (N evaluations); record the best g as history entry 0.
    2. each of G generations: carry the elite (argmax g, ties -> lowest index)
       into the new population uncopied and WITHOUT re-evaluating it, then build
       N-1 children -- each ``variation.produce(select_parent, rng)`` ->
       ``problem.repair`` -> one ``problem.g`` -> ``variation.observe(g)`` --
       where ``select_parent`` runs a tournament over the current, fully scored
       population and returns a ``Parent(index, genome, quality)``.
- one evaluation per child: total ``problem.g`` calls = N + (N-1)*G;
  ``reproduction_events`` = (N-1)*G = ``T_AOS``.
- determinism: all randomness comes from the injected ``rng`` (init, tournament,
  operators via ``produce``); a fixed seed reproduces the whole run.
- elitism: the incumbent best never dies, so ``best_quality_history`` is
  non-decreasing in g.
- validation: ``N < 2``, ``G < 1``, ``tournament_k < 1`` or
  ``elite_count >= N`` raise ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.random import Generator

from aos_ga.core.engine import RunResult, run, tournament_select
from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation
from aos_ga.core.variation import Parent, VariationStep

# --- Test doubles --------------------------------------------------------------


class _CountingSumProblem(Problem[list[int]]):
    """Lightweight problem double: f = sum(vector); counts ``evaluate`` calls.

    Genomes are fixed-length integer vectors sampled from ``[low, high)``. The
    direction is configurable, so the same double exercises both minimization and
    maximization through the unified g. ``evaluate`` tallies calls so tests can
    pin the exact evaluation budget.
    """

    representation = Representation.BINARY

    def __init__(
        self,
        direction: Direction = Direction.MAXIMIZE,
        *,
        length: int = 5,
        low: int = 1,
        high: int = 10,
    ) -> None:
        self.name = "counting-sum-double"
        self.direction = direction
        self._length = length
        self._low = low
        self._high = high
        self.eval_count = 0

    def evaluate(self, individual: list[int]) -> float:
        self.eval_count += 1
        return float(sum(individual))

    def initialize(self, rng: Generator) -> list[int]:
        return [int(value) for value in rng.integers(self._low, self._high, size=self._length)]


class _CopyFirstParent(VariationStep[list[int]]):
    """Trivial step: draw one parent, return a fresh copy (no operator applied)."""

    def produce(self, select_parent: Callable[[], Parent[list[int]]], rng: Generator) -> list[int]:
        return list(select_parent().genome)


class _ConstantChild(VariationStep[list[int]]):
    """Step that ignores parents and always emits the same fixed child.

    Still draws one parent to keep rng consumption realistic; used to prove
    elitism protects the incumbent when every offspring is strictly worse.
    """

    def __init__(self, child: list[int]) -> None:
        self._child = child

    def produce(self, select_parent: Callable[[], Parent[list[int]]], rng: Generator) -> list[int]:
        select_parent()
        return list(self._child)


class _RecordingStep(VariationStep[list[int]]):
    """Copy-one-parent step that records each produced child and each observe call."""

    def __init__(self) -> None:
        self.produced: list[list[int]] = []
        self.observed: list[float] = []

    def produce(self, select_parent: Callable[[], Parent[list[int]]], rng: Generator) -> list[int]:
        child = list(select_parent().genome)
        self.produced.append(child)
        return child

    def observe(self, child_quality: float) -> None:
        self.observed.append(child_quality)


def _initial_population(
    seed: int, size: int, *, length: int = 5, low: int = 1, high: int = 10
) -> list[list[int]]:
    """Reproduce the population the engine builds by calling ``initialize`` size times.

    Mirrors ``_CountingSumProblem.initialize`` under the same seed, so a test can
    predict what the run starts from without reaching into the engine.
    """
    gen = np.random.default_rng(seed)
    return [[int(v) for v in gen.integers(low, high, size=length)] for _ in range(size)]


# --- tournament_select ---------------------------------------------------------


def test_tournament_returns_the_best_of_the_sampled_indices() -> None:
    qualities = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    k = 3
    for seed in range(64):
        sampled = np.random.default_rng(seed).integers(0, len(qualities), size=k)
        winner = int(sampled[0])
        for candidate in (int(x) for x in sampled[1:]):
            if qualities[candidate] > qualities[winner]:  # strict -> earliest max
                winner = candidate
        assert tournament_select(qualities, k, np.random.default_rng(seed)) == winner


def test_tournament_breaks_ties_toward_the_earliest_sampled() -> None:
    qualities = [5.0] * 6
    for seed in range(32):
        sampled = np.random.default_rng(seed).integers(0, len(qualities), size=3)
        assert tournament_select(qualities, 3, np.random.default_rng(seed)) == int(sampled[0])


def test_tournament_with_k_one_returns_the_single_draw() -> None:
    qualities = [2.0, 7.0, 1.0, 9.0]
    for seed in range(16):
        sampled = np.random.default_rng(seed).integers(0, len(qualities), size=1)
        assert tournament_select(qualities, 1, np.random.default_rng(seed)) == int(sampled[0])


def test_tournament_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        tournament_select([1.0, 2.0], 0, np.random.default_rng(0))


# --- run: reproducibility ------------------------------------------------------


def test_run_is_reproducible_for_a_fixed_seed() -> None:
    first: RunResult[list[int]] = run(
        _CountingSumProblem(),
        _CopyFirstParent(),
        np.random.default_rng(11),
        population_size=8,
        generations=6,
    )
    second: RunResult[list[int]] = run(
        _CountingSumProblem(),
        _CopyFirstParent(),
        np.random.default_rng(11),
        population_size=8,
        generations=6,
    )
    assert first == second


# --- run: evaluation budget (one evaluation per child) -------------------------


@pytest.mark.parametrize(
    ("population_size", "generations"),
    [(20, 20), (20, 30), (20, 50), (50, 20), (50, 30), (50, 50)],
)
def test_evaluation_budget_matches_the_formula(population_size: int, generations: int) -> None:
    problem = _CountingSumProblem()
    result = run(
        problem,
        _CopyFirstParent(),
        np.random.default_rng(0),
        population_size=population_size,
        generations=generations,
    )
    assert result.reproduction_events == (population_size - 1) * generations
    assert result.evaluations == population_size + (population_size - 1) * generations
    assert problem.eval_count == result.evaluations  # elite is never re-evaluated


# --- run: history shape and monotonicity ---------------------------------------


def test_best_quality_history_has_one_entry_per_generation_plus_init() -> None:
    generations = 8
    result = run(
        _CountingSumProblem(),
        _CopyFirstParent(),
        np.random.default_rng(1),
        population_size=12,
        generations=generations,
    )
    assert len(result.best_quality_history) == generations + 1
    assert result.best_quality == result.best_quality_history[-1]


def test_best_quality_never_decreases_under_elitism() -> None:
    result = run(
        _CountingSumProblem(),
        _CopyFirstParent(),
        np.random.default_rng(2),
        population_size=12,
        generations=10,
    )
    history = result.best_quality_history
    assert all(earlier <= later for earlier, later in zip(history, history[1:], strict=False))


# --- run: elitism protects the incumbent ---------------------------------------


def test_elitism_preserves_the_incumbent_when_every_child_is_worse() -> None:
    population = _initial_population(seed=5, size=10)
    best_genome = max(population, key=lambda ind: sum(ind))  # first max -> lowest index
    best_quality = float(sum(best_genome))
    generations = 7

    result = run(
        _CountingSumProblem(),  # MAXIMIZE; sums >= length, always beat the constant child
        _ConstantChild([0] * 5),  # sum 0 -> strictly worse than any real individual
        np.random.default_rng(5),
        population_size=10,
        generations=generations,
    )

    assert result.best == best_genome
    assert result.best_quality == best_quality
    assert result.best_quality_history == [best_quality] * (generations + 1)


# --- run: direction is honoured through g --------------------------------------


def test_direction_is_honoured_through_g() -> None:
    population = _initial_population(seed=8, size=10)
    lowest = float(min(sum(ind) for ind in population))
    highest = float(max(sum(ind) for ind in population))

    minimizing = run(
        _CountingSumProblem(Direction.MINIMIZE),
        _CopyFirstParent(),
        np.random.default_rng(8),
        population_size=10,
        generations=6,
    )
    maximizing = run(
        _CountingSumProblem(Direction.MAXIMIZE),
        _CopyFirstParent(),
        np.random.default_rng(8),
        population_size=10,
        generations=6,
    )

    # Copy-first invents no new individual, so elitism holds each extreme.
    assert minimizing.best_objective == lowest
    assert maximizing.best_objective == highest


# --- run: the observe hook fires once per child, with the child's g ------------


def test_observe_is_called_once_per_child_with_its_quality() -> None:
    problem = _CountingSumProblem()
    step = _RecordingStep()
    population_size, generations = 9, 5

    result = run(
        problem,
        step,
        np.random.default_rng(4),
        population_size=population_size,
        generations=generations,
    )

    expected = (population_size - 1) * generations
    assert len(step.produced) == expected  # N-1 offspring per generation
    assert len(step.observed) == expected
    assert result.reproduction_events == expected
    sign = problem.direction.sign
    assert step.observed == [sign * float(sum(child)) for child in step.produced]


# --- run: interchangeable step -- the skeleton holds its invariants -------------


def test_skeleton_holds_invariants_with_a_trivial_step() -> None:
    # A stand-in step that is neither CGA nor AOS still runs on the same skeleton
    # with the same budget and elitism -- proof the loop does not hardcode variation.
    problem = _CountingSumProblem()
    result = run(
        problem,
        _ConstantChild([1, 1, 1, 1, 1]),
        np.random.default_rng(0),
        population_size=10,
        generations=5,
    )
    assert result.reproduction_events == (10 - 1) * 5
    assert problem.eval_count == 10 + (10 - 1) * 5


# --- run: validation of degenerate configurations ------------------------------


def test_run_rejects_degenerate_configurations() -> None:
    problem = _CountingSumProblem()
    step = _CopyFirstParent()

    with pytest.raises(ValueError):  # no room for offspring
        run(problem, step, np.random.default_rng(0), population_size=1, generations=5)
    with pytest.raises(ValueError):  # no generations
        run(problem, step, np.random.default_rng(0), population_size=10, generations=0)
    with pytest.raises(ValueError):  # empty tournament
        run(
            problem,
            step,
            np.random.default_rng(0),
            population_size=10,
            generations=5,
            tournament_k=0,
        )
    with pytest.raises(ValueError):  # elite fills the whole population
        run(
            problem,
            step,
            np.random.default_rng(0),
            population_size=5,
            generations=5,
            elite_count=5,
        )
