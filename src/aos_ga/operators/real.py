"""Real-representation variation operators.

The full operator pool for the continuous benchmark functions: two crossovers -- Simulated
Binary Crossover (SBX) and arithmetic crossover -- and two mutations -- polynomial mutation
and gaussian mutation. The classic GA baseline (CGA slice) uses SBX + polynomial mutation;
the reduced AOS pool is SBX + gaussian mutation; arithmetic crossover and gaussian mutation
complete the full pool. All implement the shared
:class:`~aos_ga.core.operator.Operator` interface over ``list[float]`` coordinate vectors --
one application returns exactly one child and draws all randomness from the injected
generator, so a fixed seed reproduces the result.

The crossovers are domain-unaware: an offspring may fall outside the box, and legalizing it
back into ``[lower, upper]`` is the continuous problem's box-clip repair. The two mutations
scale their step with the domain width and take that scale through the constructor (the
polynomial's ``span = u - l``, the gaussian's ``sigma``), never learning the bounds inside
``apply``.
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
        u = rng.random(size=d)

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
    """Polynomial Mutation: perturb each variable with probability ``1/d`` by a span-scaled step.

    Following Deb & Goyal (1996), every variable is visited with probability ``1/d`` (with
    ``d`` the vector length), so one application perturbs a single variable on average. A
    visited variable is shifted by ``span * delta``, where ``delta`` is drawn from the
    polynomial distribution (index ``eta``) with ``|delta| <= 1`` and either sign, and
    ``span = u - l`` is the domain width supplied to the constructor (standard Deb form
    ``x_i + (u - l) delta_i``). Unvisited variables pass through, so an unchanged child is a
    legal outcome. The operator is domain-aware only through ``span``; it never clips, so an
    out-of-box result is legalized by the problem's repair.
    """

    operator_id = "polynomial"
    representation = Representation.REAL
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, span: float, eta: float = 20.0):
        self.span = span
        self.eta = eta

    def _mutate_gene(self, gene: float, r: float) -> float:
        """Return ``gene`` shifted by the span-scaled polynomial step for a draw ``r``.

        ``r < 0.5`` gives a negative ``delta`` (down), otherwise a non-negative one (up);
        ``|delta|`` never exceeds 1, so the shift ``span * delta`` is bounded by ``span``.
        """
        if r < 0.5:
            delta = (2.0 * r) ** (1.0 / (self.eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (self.eta + 1.0))
        return float(gene + delta * self.span)

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


class ArithmeticCrossover(Operator[list[float]]):
    """Arithmetic Crossover: recombine two vectors with a per-variable convex combination.

    Each variable is an independent convex combination ``alpha_i * p1_i + (1 - alpha_i) *
    p2_i`` with its own weight ``alpha_i`` drawn from ``rng``, so the child fills the box
    spanned by the parents rather than the line segment between them. Because
    ``alpha_i ~ U(0, 1)`` is symmetric under ``alpha <-> 1 - alpha``, one draw already samples
    the offspring pair uniformly and no "which parent first" bit is needed. A convex blend
    stays within the parents on every axis, so -- unlike SBX -- it never spreads past them.
    """

    operator_id = "arithmetic"
    representation = Representation.REAL
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[float]], rng: Generator) -> list[float]:
        """Recombine two parent vectors into one child via a per-variable convex combination.

        Each variable draws its own weight ``alpha_i`` from ``rng``, so the child fills the
        box spanned by the parents. No boundary correction is applied, but a convex blend of
        in-box parents is already in-box, so repair is a no-op here.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        d = len(parents[0])
        alpha = rng.random(size=d)
        return [float(alpha[i] * parents[0][i] + (1 - alpha[i]) * parents[1][i]) for i in range(d)]


class GaussianMutation(Operator[list[float]]):
    """Gaussian Mutation: perturb every variable by a Gaussian step of scale ``sigma``.

    Every variable is shifted by ``sigma * z_i`` with ``z_i ~ N(0, 1)`` drawn from ``rng``,
    so the whole vector moves at once -- unlike polynomial mutation's ``1/d`` subset. Because
    the noise is continuous and applied on every axis, an application always changes the
    genome, and ``z``'s unbounded support lets a step exceed any fixed bound. The operator is
    domain-aware only through ``sigma`` (a tenth of the domain width); it never clips, so
    an out-of-box child is legalized by the problem's repair.
    """

    operator_id = "gaussian"
    representation = Representation.REAL
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, sigma: float):
        self.sigma = sigma

    def apply(self, parents: Sequence[list[float]], rng: Generator) -> list[float]:
        """Return a copy of the parent vector with every variable shifted by ``sigma * z_i``.

        The Gaussian steps ``z_i ~ N(0, 1)`` are drawn from ``rng`` alone and scaled by
        ``sigma``; the parent is never mutated. Every coordinate is perturbed, so the child
        always differs from the parent.
        """
        if len(parents) != self.arity:
            raise ValueError(f"Expected {self.arity} parents, got {len(parents)}")

        d = len(parents[0])
        z = rng.standard_normal(size=d)
        gaussian_steps = z * self.sigma
        return [float(parents[0][i] + gaussian_steps[i]) for i in range(d)]
