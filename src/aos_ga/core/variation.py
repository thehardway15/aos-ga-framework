"""Interchangeable variation step for the GA skeleton.

The engine builds every offspring by delegating to a variation step: one call
turns the current population into exactly one child. The step draws its parents
through an injected tournament service and its randomness from an injected
``Generator`` -- it never sees the population or its fitnesses directly. This is
the seam that lets the classic GA (crossover then mutation) and a future adaptive
operator-selection strategy (one pooled operator plus credit) run on the very same
loop, differing only in this class and never in the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from numpy.random import Generator

from .representation import Genome


@dataclass(frozen=True)
class Parent(Generic[Genome]):
    """One tournament winner handed to a variation step.

    Immutable, so a step can read but never corrupt the population behind it. It
    carries the winner's ``index`` (a ``parent_id`` for adaptive-operator-selection
    logs), its ``genome`` and its quality ``g`` (``quality``) -- the last lets an
    AOS step read ``g_ref = max(parent.quality ...)`` without re-evaluating any
    parent, since parents are already scored.
    """

    index: int
    genome: Genome
    quality: float


class VariationStep(ABC, Generic[Genome]):
    """Produces one child per reproduction event -- the swappable half of the GA.

    Subclasses implement :meth:`produce`; :meth:`observe` is an optional
    post-evaluation hook (a no-op by default). The engine owns everything else --
    selection, elitism, evaluation, succession -- so a classic two-operator
    pipeline and a pooled AOS strategy differ only in this class, never in the loop.
    """

    @abstractmethod
    def produce(self, select_parent: Callable[[], Parent[Genome]], rng: Generator) -> Genome:
        """Build exactly ONE child (unevaluated) from the current population.

        Call ``select_parent`` once per parent needed (arity is the step's own
        concern) and draw randomness only from ``rng`` -- the same generator the
        tournament behind ``select_parent`` draws from, so the whole step stays
        reproducible from the run seed. Never mutate a parent's genome. The
        skeleton evaluates and legalizes (``Problem.repair``) the returned child.
        """

    def observe(self, child_quality: float) -> None:
        """Post-evaluation hook: the skeleton passes the child's g right after
        evaluating it. No-op by default (the classic GA ignores it); an AOS step
        overrides it to turn quality into a reward and update its statistics."""
        return None
