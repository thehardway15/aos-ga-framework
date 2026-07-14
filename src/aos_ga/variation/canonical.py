"""The canonical GA's variation step: a fixed crossover-then-mutation pipeline.

``CanonicalPipeline`` is the first concrete
:class:`~aos_ga.core.variation.VariationStep` and the reference baseline the AOS
strategies are measured against. Unlike the shared-pool model -- one operator drawn
per reproduction event -- it applies a fixed pair of complementary operators to a
single offspring: crossover with probability ``p_c``, then mutation with probability
``p_m``. It is generic over the genome type and takes its operators and
probabilities in the constructor, so the same class assembles the permutation
baseline (OX ``p_c=0.9`` then inversion ``p_m=0.1``) and, later, the binary and
real-valued ones -- reusing the existing :class:`~aos_ga.core.operator.Operator`
implementations by composition and never hardcoding a representation. Each call
returns exactly one unevaluated child, so on the skeleton it costs one evaluation
per child, the same budget as the shared pool. Every draw comes from the injected
``Generator`` in a fixed order, so a seed reproduces the child.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from numpy.random import Generator

from ..core.operator import Operator
from ..core.representation import Genome
from ..core.variation import Parent, VariationStep


class CanonicalPipeline(VariationStep[Genome]):
    """Fixed two-operator pipeline: crossover with ``p_c``, then mutation with ``p_m``.

    The classic GA variation model. ``observe`` stays the inherited no-op -- the
    canonical GA assigns no operator credit, so it is left out of the AOS reward
    machinery.
    """

    def __init__(
        self,
        crossover: Operator[Genome],
        p_c: float,
        mutation: Operator[Genome],
        p_m: float,
    ) -> None:
        """Store the operator pair and their per-individual application probabilities.

        Raises ``ValueError`` if either probability falls outside ``[0, 1]``, if
        ``mutation`` is not unary (the pipeline mutates exactly one child), or if
        ``crossover`` is not at least binary (recombination needs two parents, and
        the crossover-skipped path copies the first drawn parent).
        """
        if not (0 <= p_c <= 1):
            raise ValueError(f"crossover probability must be in [0, 1], got {p_c}")
        if not (0 <= p_m <= 1):
            raise ValueError(f"mutation probability must be in [0, 1], got {p_m}")
        if mutation.arity != 1:
            raise ValueError(f"mutation operator must have arity 1, got {mutation.arity}")
        if crossover.arity < 2:
            raise ValueError(f"crossover operator must have arity >= 2, got {crossover.arity}")

        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m

    def produce(self, select_parent: Callable[[], Parent[Genome]], rng: Generator) -> Genome:
        """Build one unevaluated child: crossover with ``p_c``, then mutation with ``p_m``.

        Draws exactly ``crossover.arity`` parents (unconditionally, before the
        ``p_c`` coin), then with probability ``p_c`` recombines them, otherwise
        copies the first drawn parent's genome (a fresh, non-aliased copy). With
        probability ``p_m`` that child is mutated in turn. All randomness is drawn
        from ``rng`` in this fixed order, and no parent is ever mutated.
        """
        parents = [select_parent().genome for _ in range(self.crossover.arity)]
        if rng.random() < self.p_c:
            child = self.crossover.apply(parents, rng)
        else:
            child = copy.copy(parents[0])  # crossover skipped: copy the first parent

        if rng.random() < self.p_m:
            child = self.mutation.apply([child], rng)

        return child
