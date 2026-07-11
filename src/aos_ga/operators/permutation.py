"""Permutation-representation variation operators.

The reduced operator pool the classic GA baseline uses for the travelling-salesman
problem: Order Crossover for recombination and Simple Inversion Mutation for
perturbation. Both implement the shared :class:`~aos_ga.core.operator.Operator`
interface over ``list[int]`` city permutations -- one application returns exactly one
child and draws all randomness from the injected generator, so a fixed seed
reproduces the result.
"""

from collections.abc import Sequence

from numpy.random import Generator

from ..core.operator import Operator, OperatorKind
from ..core.representation import Representation


class OrderCrossover(Operator[list[int]]):
    """Order Crossover (OX1): inherit one contiguous segment, fill the rest in order.

    A random segment of one parent (the donor) is copied verbatim onto the child at
    the same positions; the remaining cities are taken from the other parent in the
    order they appear there, scanned cyclically from just past the segment. The child
    therefore keeps a block of the donor's tour and the relative order of the other
    cities from the second parent -- the property that suits OX to order-based problems.
    """

    operator_id = "ox"
    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Recombine two parent tours into one child permutation.

        A single random bit picks which parent donates the copied segment; only that
        one child is built, so the textbook pair of offspring costs one evaluation.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        n = len(parents[0])
        b = int(rng.integers(2))
        donor, order_source = parents[b], parents[1 - b]
        i, j = sorted(int(x) for x in rng.integers(0, n, size=2))

        # Positions to fill: everything outside segment [i, j], cyclically from j+1.
        fill_positions = list(range(j + 1, n)) + list(range(i))
        # Values: order_source read cyclically from j+1, skipping the copied segment.
        segment = set(donor[i : j + 1])
        fill_values = [
            order_source[(j + 1 + t) % n]
            for t in range(n)
            if order_source[(j + 1 + t) % n] not in segment
        ]

        child = list(donor)  # the segment is already in place; overwrite the rest
        for pos, val in zip(fill_positions, fill_values, strict=True):
            child[pos] = val
        return child


class SegmentInversion(Operator[list[int]]):
    """Simple Inversion Mutation: reverse one random contiguous segment of the tour.

    Two distinct cut points are drawn and the sub-tour between them is reversed, a
    local rearrangement of the route. Distinct points (segment length >= 2) keep every
    application a real change rather than a no-op.
    """

    operator_id = "inversion"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Return a copy of the parent tour with one random segment reversed."""
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        child = list(parents[0])
        i, j = (int(x) for x in sorted(rng.choice(len(child), size=2, replace=False)))
        child[i : j + 1] = child[i : j + 1][::-1]
        return child
