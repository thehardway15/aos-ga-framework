"""The single-operator variation step: one fixed operator is the whole variation.

``SingleOperatorStep`` is the concrete
:class:`~aos_ga.core.variation.VariationStep` that applies a single fixed operator
on every reproduction event. Unlike the canonical pipeline -- a fixed
crossover-then-mutation pair gated by ``p_c``/``p_m`` -- it has no stages and no
probabilities: the operator is applied unconditionally, producing exactly one child
per call. This makes it the step that runs each operator in isolation as an upper
reference point for the adaptive operator-selection strategies, and the primitive
the adaptive variation step reduces to once a fixed operator is drawn from the pool.
It is generic over the genome type and reuses the existing
:class:`~aos_ga.core.operator.Operator` implementations by composition, so the same
class carries a permutation, binary or real-valued operator without ever hardcoding
a representation. Each call returns exactly one unevaluated child, so on the skeleton
it costs one evaluation per child -- the same budget as the shared pool. Every draw
comes from the injected ``Generator`` in a fixed order, so a seed reproduces the child.
"""

from __future__ import annotations

from collections.abc import Callable

from numpy.random import Generator

from ..core.operator import Operator
from ..core.representation import Genome
from ..core.variation import Parent, VariationStep


class SingleOperatorStep(VariationStep[Genome]):
    """Applies one fixed operator per reproduction event -- the simplest variation step.

    The operator is the entire variation: there is no operator pair and no
    application probability, so it fires on every call. ``observe`` stays the
    inherited no-op -- this baseline assigns no operator credit, so it is left out
    of the adaptive reward machinery.
    """

    def __init__(self, operator: Operator[Genome]) -> None:
        """Store the single operator that is the whole variation.

        Validates nothing about it: unlike the canonical pipeline there is no arity
        slot to mis-fill (a single operator of arity 1 or 2 is a legal whole
        variation), and the :class:`~aos_ga.core.operator.Operator` contract already
        fixes ``arity`` at 1 or 2. The operator's ``kind`` is irrelevant here.
        """
        self.operator = operator

    def produce(self, select_parent: Callable[[], Parent[Genome]], rng: Generator) -> Genome:
        """Build one unevaluated child by applying the operator to fresh parents.

        Draws exactly ``operator.arity`` parents by calling ``select_parent`` that
        many times (one for a unary operator, two for a binary one), then applies
        the operator to their genomes -- unconditionally, since no ``p_c``/``p_m``
        coin gates it. Returns the operator's single child directly: the step adds no
        copy of its own, because ``Operator.apply`` already returns a fresh,
        non-aliased child and never mutates its parents. All randomness is drawn from
        ``rng`` in this fixed order, so a fixed seed reproduces the child.
        """
        parents = [select_parent().genome for _ in range(self.operator.arity)]
        child = self.operator.apply(parents, rng)
        return child
