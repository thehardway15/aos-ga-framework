"""Binary-representation variation operators.

The reduced operator pool the classic GA baseline uses for the 0/1 knapsack problem:
Uniform Crossover for recombination and Bit-Flip Mutation for perturbation. Both implement
the shared :class:`~aos_ga.core.operator.Operator` interface over ``list[int]`` 0/1
item-selection vectors -- one application returns exactly one child and draws all
randomness from the injected generator, so a fixed seed reproduces the result.
"""

from collections.abc import Sequence

from numpy.random import Generator

from ..core.operator import Operator, OperatorKind
from ..core.representation import Representation


class BitFlipMutation(Operator[list[int]]):
    """Bit-Flip Mutation: flip each bit independently with probability ``1/n``.

    With ``n`` the genome length, every selection bit is inverted with probability ``1/n``,
    so a single application flips one bit on average -- adding or dropping items and thus
    able to change the number of ones in the chromosome. The flips are independent and
    unconstrained: an application may flip several bits or none at all, in which case the
    child equals the parent. This is the deliberate contrast with the permutation inversion
    operator, which forces a real change on every application.
    """

    operator_id = "bitflip"
    representation = Representation.BINARY
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Return a copy of the parent bitstring with each bit flipped at probability 1/n.

        The mutation rate ``1/n`` is derived from the parent's own length, drawn from
        ``rng`` alone, and applied independently per bit; the parent is never mutated. With
        no lower bound on the number of flips, an unchanged child is a legal outcome.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")
        n = len(parents[0])
        flips = rng.random(size=n) < 1.0 / n
        return [int(1 - parents[0][i]) if flips[i] else int(parents[0][i]) for i in range(n)]


class UniformCrossover(Operator[list[int]]):
    """Uniform Crossover: inherit each bit from one of the two parents at random.

    A per-bit Bernoulli(1/2) selector chooses, independently at every position, which
    parent donates that bit, so the child mixes both parents with no positional bias -- the
    property that suits uniform crossover to the knapsack's uncoupled representation.
    Positions where the parents already agree pass their shared bit through unchanged.
    """

    operator_id = "uniform"
    representation = Representation.BINARY
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Recombine two parent bitstrings into one child via a per-bit parent selector.

        Draws one selector mask from ``rng`` and returns the single child it defines; the
        complementary child (the mirror selector) is never built, so the textbook pair of
        offspring costs one evaluation. Because the mask's complement is equiprobable, that
        one child is already a uniform draw from the pair, so no extra parent-order bit is
        needed (unlike order crossover). Parents are never mutated.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")
        n = len(parents[0])
        mask = rng.integers(0, 2, size=n)
        return [int(parents[mask[i]][i]) for i in range(n)]
