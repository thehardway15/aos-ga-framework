"""Binary-representation variation operators.

The full operator pool for the 0/1 knapsack problem: two crossovers -- Single-Point
Crossover and Uniform Crossover -- and two mutations -- Bit-Flip Mutation and Swap-Bit
Mutation. The classic GA baseline and the reduced AOS pool use uniform + bit-flip;
single-point and swap-bit complete the full pool. All implement the shared
:class:`~aos_ga.core.operator.Operator` interface over ``list[int]`` 0/1 item-selection
vectors -- one application returns exactly one child and draws all randomness from the
injected generator, so a fixed seed reproduces the result.
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


class SinglePointCrossover(Operator[list[int]]):
    """Single-Point Crossover: copy a prefix from one parent and the suffix from the other.

    A single cut point ``k`` is drawn uniformly from ``[1, n)`` and a random head bit picks
    which parent donates the prefix; the child is that parent's ``head[:k]`` followed by the
    other parent's ``tail[k:]``. Unlike uniform crossover's per-bit mask, a single cut is not
    self-complementing -- fixing the head parent would only ever reach one child of the pair
    -- so the explicit head bit is what makes the returned child a uniform draw from the two
    complementary offspring, exactly as for order crossover.
    """

    operator_id = "singlepoint"
    representation = Representation.BINARY
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Recombine two parent bitstrings into one child via a single crossover point.

        A random head bit picks which parent donates the prefix and one cut point is drawn
        from ``rng``; only that one child of the complementary pair is built, so the textbook
        pair of offspring costs one evaluation. The head bit makes it a uniform draw from the
        pair (unlike uniform crossover, whose per-bit mask needs no such bit). Parents are
        never mutated.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")
        n = len(parents[0])
        b = int(rng.integers(2))
        head, tail = parents[b], parents[1 - b]

        point = rng.integers(1, n)
        return head[:point] + tail[point:]


class SwapBitMutation(Operator[list[int]]):
    """Swap-Bit Mutation: exchange one selected bit with one unselected bit.

    One position holding a 1 and one holding a 0 are drawn uniformly and independently, and
    their values are exchanged. The operator therefore always conserves the number of ones in
    the chromosome and always makes a real change -- turning one item off while turning
    another on -- with no structural no-op (the "1<->0" variant). This is the deliberate
    contrast with bit-flip, which can change the number of ones and may leave the genome
    unchanged. Edge case: an all-ones or all-zeros parent has no 1/0 pair to exchange, so the
    child is a fresh copy of the parent.
    """

    operator_id = "swapbit"
    representation = Representation.BINARY
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Return a copy of the parent bitstring with one 1 and one 0 exchanged.

        The two positions are drawn independently from ``rng`` -- one from the ones, one from
        the zeros -- so the swap always conserves the number of ones. When the parent is all
        ones or all zeros there is no pair to exchange and the child is a plain copy. The
        parent is never mutated.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        zeros_indices = [i for i, bit in enumerate(parents[0]) if bit == 0]
        ones_indices = [i for i, bit in enumerate(parents[0]) if bit == 1]
        child = list(parents[0])

        if zeros_indices and ones_indices:
            i = rng.choice(zeros_indices)
            j = rng.choice(ones_indices)
            child[i], child[j] = child[j], child[i]

        return child
