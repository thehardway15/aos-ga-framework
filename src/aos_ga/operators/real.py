"""Real-representation variation operators.

The reduced operator pool the classic GA baseline uses for the continuous benchmark
functions: Simulated Binary Crossover for recombination and Polynomial Mutation for
perturbation. Both implement the shared :class:`~aos_ga.core.operator.Operator` interface
over ``list[float]`` coordinate vectors -- one application returns exactly one child and
draws all randomness from the injected generator, so a fixed seed reproduces the result.
The operators are deliberately domain-unaware: an offspring may fall outside the box, and
legalizing it back into ``[lower, upper]`` is the continuous problem's box-clip repair.
"""

from collections.abc import Sequence

from numpy.random import Generator

from ..core.operator import Operator, OperatorKind
from ..core.representation import Representation


class SBX(Operator[list[float]]):
    """Simulated Binary Crossover (SBX): recombine two vectors with a per-variable spread.

    Following Deb & Agrawal (1995), each variable draws its own spread factor ``beta`` from
    the SBX distribution (index ``eta``; larger ``eta`` keeps children near the parents),
    producing two offspring symmetric about the parent midpoint. Because one application
    yields a single child, a whole-vector coin returns either the first or the second
    offspring -- the pick is shared across every variable, so the child leans consistently
    toward one parent. This is the unbounded variant: no boundary correction is applied, so
    a child may spread beyond the parents and the problem's repair is what legalizes it.
    """

    operator_id = "sbx"
    representation = Representation.REAL
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def __init__(self, eta: float = 20.0):
        self.eta = eta

    def _spread(self, u: float) -> float:
        """Sample the spread factor ``beta`` for one variable from the SBX distribution.

        Inverts the distribution's CDF for a uniform draw ``u``: ``u <= 0.5`` yields a
        contracting spread (``beta <= 1``), otherwise an expanding one (``beta >= 1``).
        """
        if u <= 0.5:
            return float((2 * u) ** (1 / (self.eta + 1)))
        else:
            return float((1 / (2 * (1 - u))) ** (1 / (self.eta + 1)))

    def apply(self, parents: Sequence[list[float]], rng: Generator) -> list[float]:
        """Recombine two parent vectors into one child via per-variable SBX spreads.

        Each variable draws its own ``beta``; the two symmetric offspring are built and a
        single coin returns one of them, so the textbook pair costs one evaluation. No
        boundary correction is applied -- an out-of-box child is left for repair to clip.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        d = len(parents[0])
        u = rng.random(size=d).tolist()

        beta = [self._spread(u_i) for u_i in u]
        c1 = [
            0.5 * ((1 + beta_i) * p1 + (1 - beta_i) * p2)
            for beta_i, p1, p2 in zip(beta, parents[0], parents[1], strict=True)
        ]
        c2 = [
            0.5 * ((1 - beta_i) * p1 + (1 + beta_i) * p2)
            for beta_i, p1, p2 in zip(beta, parents[0], parents[1], strict=True)
        ]

        if rng.random() < 0.5:
            return c1
        return c2


class PolynomialMutation(Operator[list[float]]):
    """Polynomial Mutation: perturb each variable with probability ``1/d`` by a bounded step.

    Following Deb & Goyal (1996), every variable is visited with probability ``1/d`` (with
    ``d`` the vector length), so one application perturbs a single variable on average. A
    visited variable is shifted by ``delta`` drawn from the polynomial distribution (index
    ``eta``), with ``|delta| <= 1`` and either sign; unvisited variables pass through, so an
    unchanged child is a legal outcome. Like SBX this is domain-unaware -- the step is added
    directly and any out-of-box result is legalized by the problem's repair.
    """

    operator_id = "polynomial"
    representation = Representation.REAL
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, eta: float = 20.0):
        self.eta = eta

    def _mutate_gene(self, gene: float, r: float) -> float:
        """Return ``gene`` shifted by the polynomial perturbation ``delta`` for a draw ``r``.

        ``r < 0.5`` gives a negative ``delta`` (down), otherwise a non-negative one (up);
        ``|delta|`` never exceeds 1, so the step is bounded independently of the domain.
        """
        if r < 0.5:
            delta = (2.0 * r) ** (1.0 / (self.eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (self.eta + 1.0))
        return float(gene + delta)

    def apply(self, parents: Sequence[list[float]], rng: Generator) -> list[float]:
        """Return a copy of the parent vector with each variable mutated at probability 1/d.

        The rate ``1/d`` is derived from the parent's own length and drawn from ``rng``
        alone; the parent is never mutated and an all-unchanged child is a legal outcome.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        d = len(parents[0])
        mask = rng.random(size=d) < 1.0 / d
        r_value = rng.random(size=d)
        return [
            self._mutate_gene(parents[0][i], r_value[i]) if mask[i] else parents[0][i]
            for i in range(d)
        ]
