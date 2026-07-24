"""The generational GA skeleton: the invariant half of the algorithm.

Generic over ``Problem``/``Genome``, with the variation model factored out to an
interchangeable :class:`~aos_ga.core.variation.VariationStep`. The engine owns
population initialization and evaluation, tournament selection, elitism,
succession and the generation budget, and evaluates every child exactly once. All
ordering decisions go through the unified quality ``g`` (more-is-better), so a
minimization problem and a maximization one run through the same code. Every draw
comes from the injected ``Generator``, so a fixed seed reproduces the whole run.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np
from numpy.random import Generator

from .problem import Problem
from .representation import Genome
from .variation import Parent, VariationStep


@dataclass(frozen=True)
class RunResult(Generic[Genome]):
    """Outcome of one GA run.

    ``best`` is the highest-``g`` genome found; ``best_quality`` is its ``g``
    (more-is-better) and ``best_objective`` its raw objective ``f`` in the
    problem's own units (e.g. tour length), recovered as
    ``direction.sign * best_quality`` without a further evaluation.
    ``best_quality_history`` holds the best ``g`` after initialization and after
    each generation (length ``generations + 1``), so it is non-decreasing under
    elitism. ``reproduction_events`` is the number of variation steps
    ``(population_size - elite_count) * generations`` (``T_AOS`` when
    ``elite_count`` is 1), and ``evaluations`` the total ``g`` calls
    ``population_size + reproduction_events``.
    """

    best: Genome
    best_quality: float
    best_objective: float
    best_quality_history: list[float]
    reproduction_events: int
    evaluations: int


def run(
    problem: Problem[Genome],
    variation: VariationStep[Genome],
    rng: Generator,
    *,
    population_size: int,
    generations: int,
    tournament_k: int = 3,
    elite_count: int = 1,
) -> RunResult[Genome]:
    """Evolve ``problem`` for ``generations`` with ``variation`` from one seed.

    Initializes ``population_size`` individuals, then in each generation carries
    the top ``elite_count`` (by ``g``, ties to the lowest index) forward uncopied
    and un-re-evaluated, and fills the rest with children -- each produced by
    ``variation`` from tournament-selected parents, legalized by
    ``Problem.repair`` and evaluated exactly once. All randomness is drawn from
    ``rng``, so a fixed seed reproduces the run. Raises ``ValueError`` on a
    degenerate configuration (``population_size < 2``, ``generations < 1``,
    ``tournament_k < 1`` or ``elite_count >= population_size``).
    """
    if population_size <= 1:
        raise ValueError(f"population_size must be > 1, got {population_size}")
    if generations <= 0:
        raise ValueError(f"generations must be positive, got {generations}")
    if tournament_k <= 0:
        raise ValueError(f"tournament_k must be positive, got {tournament_k}")
    if elite_count < 1 or elite_count >= population_size:
        raise ValueError(f"elite_count must be in [1, {population_size}), got {elite_count}")

    population = [problem.initialize(rng) for _ in range(population_size)]
    qualities = [problem.g(ind) for ind in population]
    history = [max(qualities)]

    def select_parent() -> Parent[Genome]:
        idx = tournament_select(qualities, tournament_k, rng)
        return Parent(index=idx, genome=population[idx], quality=qualities[idx])

    for _ in range(generations):
        new_population = []
        new_qualities = []

        # Elitism: carry over the best individuals. The genome is copied rather than
        # aliased: an elite survives many generations, so a single operator that ever
        # wrote through a parent would corrupt it retroactively, in every generation it
        # already survived, and the run would fail no test. No operator does that today
        # -- the copy makes the invariant structural instead of conventional, and costs
        # one shallow copy per generation. The quality carries over unevaluated.
        elite_indices = sorted(range(len(qualities)), key=lambda i: qualities[i], reverse=True)[
            :elite_count
        ]
        for idx in elite_indices:
            new_population.append(copy.copy(population[idx]))
            new_qualities.append(qualities[idx])

        while len(new_population) < population_size:
            child = variation.produce(select_parent, rng)
            child = problem.repair(child)
            child_quality = problem.g(child)
            variation.observe(child_quality)
            new_population.append(child)
            new_qualities.append(child_quality)

        population = new_population
        qualities = new_qualities
        history.append(max(qualities))

    reproduction_events = (population_size - elite_count) * generations
    evaluations = (
        population_size + reproduction_events
    )  # initial population + all children evaluated

    return RunResult(
        best=population[int(np.argmax(qualities))],
        best_quality=max(qualities),
        best_objective=problem.direction.sign * max(qualities),
        best_quality_history=history,
        reproduction_events=reproduction_events,
        evaluations=evaluations,
    )


def tournament_select(qualities: Sequence[float], k: int, rng: Generator) -> int:
    """Return the index of a tournament winner among ``k`` sampled competitors.

    Draws ``k`` competitors with replacement via ``rng.integers(0, N, size=k)``
    and returns the index of the highest-quality one; ties go to the earliest
    competitor sampled. Reads only ``g`` values, never genomes, so it is a
    standalone, reproducible unit of parent selection. Raises ``ValueError`` if
    ``k < 1``.
    """
    if k <= 0:
        raise ValueError(f"tournament size k must be positive, got {k}")

    sampled = rng.integers(0, len(qualities), size=k)
    best_pos = int(np.argmax([qualities[i] for i in sampled]))
    return int(sampled[best_pos])
