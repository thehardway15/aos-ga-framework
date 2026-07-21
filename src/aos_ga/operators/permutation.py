"""Permutation-representation variation operators.

The full operator pool for the travelling-salesman problem: three crossovers -- Order
Crossover (OX1), Partially Mapped Crossover (PMX) and Cycle Crossover (CX) -- and three
mutations -- swap, Simple Inversion Mutation and insert. The classic GA baseline uses
OX + inversion; the reduced AOS pool is OX, CX and inversion. All implement the shared
:class:`~aos_ga.core.operator.Operator` interface over ``list[int]`` city permutations
-- one application returns exactly one child and draws all randomness from the injected
generator, so a fixed seed reproduces the result.
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


class CycleCrossover(Operator[list[int]]):
    """Cycle Crossover (CX): every position is inherited from a parent at its own index.

    The two parents partition the positions into *cycles*: starting at a position, take
    the city the other parent holds there, follow it to the position it occupies in this
    parent, and repeat until the start returns. Within a cycle the two parents hold
    exactly the same set of cities -- only arranged differently -- so a whole cycle can be
    taken from either parent without ever duplicating or dropping a city. CX exploits this
    by assigning whole cycles alternately: odd-numbered cycles from one parent, even ones
    from the other. Every position therefore keeps ``p1[k]`` or ``p2[k]``, which makes CX
    the least positionally disruptive of the permutation crossovers.

    A textbook CX yields the complementary pair (swap which parent owns the odd cycles);
    one application returns a single child picked by one random bit. The cycle structure
    is fully determined by the parents, so that bit is CX's only source of randomness.
    """

    operator_id = "cx"
    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Recombine two parent tours into one child by alternating whole position cycles.

        A single random bit picks which parent owns the odd-numbered cycles. The child
        starts as a copy of the other parent, and each odd cycle is overwritten from the
        chosen one. Because a cycle carries the same city set in both parents, every
        whole-cycle assignment is already a valid permutation, so no repair is needed.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        n = len(parents[0])
        b = int(rng.integers(2))
        first, second = parents[b], parents[1 - b]

        position_in_first = {city: idx for idx, city in enumerate(first)}
        child = list(second)
        visited = [False] * n
        odd_cycle = True

        for start in range(n):
            if visited[start]:
                continue

            positions: list[int] = []
            pos = start
            while not visited[pos]:
                visited[pos] = True
                positions.append(pos)
                pos = position_in_first[second[pos]]

            if odd_cycle:
                for p in positions:
                    child[p] = first[p]
            odd_cycle = not odd_cycle
        return child


class InsertMutation(Operator[list[int]]):
    """Insert Mutation: remove one city and reinsert it elsewhere in the tour.

    A single city is removed from its current position and inserted at a new random
    position, shifting the other cities to make room. The new position is drawn from
    the remaining indices, so every application is guaranteed to change the tour.
    """

    operator_id = "insert"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Return a copy of the parent tour with one city moved to a new position."""
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        child = list(parents[0])
        n = len(child)
        i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
        city = child.pop(i)
        child.insert(j, city)
        return child


class PartiallyMappedCrossover(Operator[list[int]]):
    """Partially Mapped Crossover (PMX): inherit a segment and map the rest by position.

    A random segment of one parent (the donor) is copied verbatim onto the child at the
    same positions. Every remaining position then takes the value the other parent holds
    at that same position; when that value already lies in the copied segment it is
    replaced by its partner under the segment mapping (donor value to other-parent value)
    and the replacement repeats until a value outside the segment is reached. The child
    therefore keeps a block of the donor's tour and, elsewhere, the absolute positions of
    the other parent wherever the mapping allows -- the property that distinguishes PMX
    from OX, which preserves relative order instead.
    """

    operator_id = "pmx"
    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Recombine two parent tours into one child permutation.

        A single random bit picks which parent donates the copied segment; only that
        one child is built, so a fixed seed reproduces the result.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        n = len(parents[0])
        b = int(rng.integers(2))
        donor, order_source = parents[b], parents[1 - b]
        i, j = sorted(int(x) for x in rng.integers(0, n, size=2))

        # Map each donor segment value to the order_source value at the same position.
        mapping = {donor[k]: order_source[k] for k in range(i, j + 1)}

        child = list(donor)  # the segment is already in place; resolve the rest by position
        for k in range(n):
            if i <= k <= j:
                continue
            val = order_source[k]
            while val in mapping:  # duplicate of the copied segment: follow the chain out
                val = mapping[val]
            child[k] = val
        return child


class SwapMutation(Operator[list[int]]):
    """Swap Mutation: exchange the cities at two positions of the tour.

    Two distinct positions are drawn and the cities at them are exchanged, leaving the
    rest of the route in place -- a local rearrangement. Distinct positions (i != j)
    keep every application a real change rather than a no-op.
    """

    operator_id = "swap"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        """Return a copy of the parent tour with two cities swapped."""
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        child = list(parents[0])
        i, j = (int(x) for x in rng.choice(len(child), size=2, replace=False))
        child[i], child[j] = child[j], child[i]
        return child
