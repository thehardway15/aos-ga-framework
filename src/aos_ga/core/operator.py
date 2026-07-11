"""Variation-operator interface for the shared AOS pool.

An operator is one arm of the adaptive operator-selection problem: a single
application consumes ``arity`` parents and produces exactly one child to be
evaluated -- one operator, one child, one evaluation. Operators know only their
representation, never a concrete problem; legalizing a child against a problem's
constraints is the caller's job (``Problem.repair``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Generic

from numpy.random import Generator

from .representation import Genome, Representation


class OperatorKind(Enum):
    """Search role of an operator, independent of its arity."""

    RECOMBINATIVE = "recombinative"
    PERTURBATIVE = "perturbative"


class Operator(ABC, Generic[Genome]):
    """A variation operator over one genome representation.

    Subclasses set the four metadata fields and implement :meth:`apply`. The
    metadata is readable without applying the operator, so the AOS strategy and
    the logs can identify the arm (``operator_id``), match it to a problem
    (``representation``), size its parent set (``arity``, 1 or 2) and record its
    search role (``kind``).
    """

    operator_id: str
    representation: Representation
    arity: int
    kind: OperatorKind

    @abstractmethod
    def apply(self, parents: Sequence[Genome], rng: Generator) -> Genome:
        """Produce one child from exactly ``arity`` parents, using only ``rng``.

        Returns a single child genome and never mutates ``parents`` -- the same
        parent is reused across offspring in the shared-pool model. Draws only
        from the injected ``rng`` (no global state), so a fixed seed reproduces
        the child. A crossover whose textbook form yields a pair returns one of
        the two chosen uniformly at random via ``rng``; the discarded child is
        never evaluated, keeping one evaluation per step.
        """
