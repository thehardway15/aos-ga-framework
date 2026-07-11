"""Problem abstraction and the unified quality function g(x).

g(x) is the single source of truth for optimization direction: every reward,
metric and baseline compares solutions through g, so the sign of the objective
is decided here once and never re-derived downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic

from numpy.random import Generator

from .representation import Genome, Representation


class Direction(Enum):
    """Optimization sense of a problem's raw objective f(x)."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @property
    def sign(self) -> int:
        """+1 for maximization, -1 for minimization: the f -> g sign."""
        return 1 if self is Direction.MAXIMIZE else -1


def quality(objective: float, direction: Direction) -> float:
    """g(x): the unified 'more is better' scalar. g = f (max), g = -f (min)."""
    return direction.sign * objective


class Problem(ABC, Generic[Genome]):
    """A test problem: representation, seeded init, legalization, f and g."""

    name: str
    direction: Direction
    representation: Representation

    @abstractmethod
    def evaluate(self, individual: Genome) -> float:
        """Raw objective f(x). Total and finite; never raises on infeasible."""

    @abstractmethod
    def initialize(self, rng: Generator) -> Genome:
        """Sample one legal individual using only `rng` (no global state)."""

    def repair(self, individual: Genome) -> Genome:
        """Legalize a genome after variation. Identity unless overridden."""
        return individual

    def g(self, individual: Genome) -> float:
        """Unified quality g(x) for this problem (direction-correct)."""
        return quality(self.evaluate(individual), self.direction)
