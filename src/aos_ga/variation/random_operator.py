"""The random-operator variation step: one operator drawn from the pool per event.

``RandomOperatorStep`` is the concrete
:class:`~aos_ga.core.variation.VariationStep` that, on every reproduction event, draws
one operator uniformly at random from an injected pool and makes that single draw the
whole variation -- no crossover-then-mutation pipeline and no application probabilities.
It is the lower reference point for the adaptive operator-selection strategies (Random
selection, ``p_i = 1/K``) and the thinnest slice of the AOS layer: the same "draw an
operator from the pool, then apply it" shape onto which the full strategies later
collapse. It assigns no operator credit, so ``observe`` stays the inherited no-op -- it
needs neither a reward update nor a dynamics snapshot. It is generic over the genome
type and reuses the existing :class:`~aos_ga.core.operator.Operator` implementations by
composition, so the same class carries a permutation, binary or real-valued pool without
ever hardcoding a representation. Each call returns exactly one unevaluated child -- one
evaluation per child, the same budget as the shared pool. The operator is drawn first
and every draw comes from the injected ``Generator``, so a seed reproduces both the
selection and the child.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from numpy.random import Generator

from ..core.operator import Operator
from ..core.representation import Genome
from ..core.variation import Parent, VariationStep


class RandomOperatorStep(VariationStep[Genome]):
    """Applies one uniformly drawn operator per reproduction event -- the Random baseline.

    A fresh operator is drawn from the pool on every call, so unlike
    :class:`~aos_ga.variation.single_operator.SingleOperatorStep` the variation is not
    fixed; unlike the canonical pipeline there is still no operator pair and no
    application probability. ``observe`` stays the inherited no-op: Random assigns no
    operator credit, so it is left out of the adaptive reward machinery.
    """

    def __init__(self, pool: Sequence[Operator[Genome]]) -> None:
        """Store the pool of operators to draw from; reject an empty one.

        An empty pool has no arm to draw, so it is rejected with ``ValueError``.
        Nothing else is validated: a single operator of arity 1 or 2 is a legal whole
        variation, the :class:`~aos_ga.core.operator.Operator` contract already fixes
        ``arity`` at 1 or 2, and the pool's representation homogeneity is the pool
        builder's guarantee. The operators' ``kind`` is irrelevant here.
        """
        if not pool:
            raise ValueError("RandomOperatorStep requires a non-empty pool of operators")
        self.pool = pool

    def produce(self, select_parent: Callable[[], Parent[Genome]], rng: Generator) -> Genome:
        """Build one unevaluated child by applying a randomly drawn pool operator.

        Draws the operator index ``rng.integers(len(pool))`` uniformly first -- so the
        selection is reproducible from the seed and independent of the parents -- then
        draws exactly ``operator.arity`` parents by calling ``select_parent`` that many
        times, and applies the operator to their genomes unconditionally (no
        ``p_c``/``p_m`` coin gates it). Returns the operator's single child directly: the
        step adds no copy of its own, because ``Operator.apply`` already returns a fresh,
        non-aliased child and never mutates its parents. All randomness is drawn from
        ``rng`` in this fixed order, so a fixed seed reproduces the child.
        """
        index = int(rng.integers(len(self.pool)))
        operator = self.pool[index]
        parents = [select_parent().genome for _ in range(operator.arity)]
        child = operator.apply(parents, rng)
        return child
