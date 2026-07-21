"""Study operator pools: pool membership as configuration, plus the factory that builds it.

This is experiment-layer configuration, not framework. It defines *which* of the framework's
operators make up each representation's full and reduced pool for this study, and builds them --
the pool composition is a treatment of the study (full vs reduced pool), so it lives here rather
than in :mod:`aos_ga`, which ships only the operators as reusable tools.

The definition is kept in two separable halves:

- ``POOL_MEMBERSHIP`` and ``pool_ids`` give the pools as pure configuration -- the
  ``operator_id`` of each arm, in a fixed order, without constructing anything.
- ``build_pool`` is the factory that instantiates one pool. It is the single place that knows
  how to construct each operator, including the domain scaling the two real-valued mutations
  require (the polynomial's ``span`` and the gaussian's ``sigma``).

Membership is data; construction is behaviour. Each reduced pool is a subset of its full pool
and always keeps at least one recombinative and one perturbative operator, so the AOS strategy
still chooses between distinct search roles.
"""

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

from aos_ga.core.operator import Operator
from aos_ga.core.representation import Representation
from aos_ga.operators.binary import (
    BitFlipMutation,
    SinglePointCrossover,
    SwapBitMutation,
    UniformCrossover,
)
from aos_ga.operators.permutation import (
    CycleCrossover,
    InsertMutation,
    OrderCrossover,
    PartiallyMappedCrossover,
    SegmentInversion,
    SwapMutation,
)
from aos_ga.operators.real import SBX, ArithmeticCrossover, GaussianMutation, PolynomialMutation


class PoolVariant(Enum):
    """Which operator pool to build for a representation: the full pool or its reduced subset."""

    FULL = "full"
    REDUCED = "reduced"


POOL_MEMBERSHIP: Mapping[Representation, Mapping[PoolVariant, tuple[str, ...]]] = {
    Representation.PERMUTATION: {
        PoolVariant.FULL: ("ox", "pmx", "cx", "swap", "inversion", "insert"),
        PoolVariant.REDUCED: ("ox", "cx", "inversion"),
    },
    Representation.BINARY: {
        PoolVariant.FULL: ("singlepoint", "uniform", "bitflip", "swapbit"),
        PoolVariant.REDUCED: ("uniform", "bitflip"),
    },
    Representation.REAL: {
        PoolVariant.FULL: ("sbx", "arithmetic", "gaussian", "polynomial"),
        PoolVariant.REDUCED: ("sbx", "gaussian"),
    },
}

_DISCRETE_FACTORIES: Mapping[str, Callable[[], Operator[Any]]] = {
    "ox": OrderCrossover,
    "pmx": PartiallyMappedCrossover,
    "cx": CycleCrossover,
    "swap": SwapMutation,
    "inversion": SegmentInversion,
    "insert": InsertMutation,
    "singlepoint": SinglePointCrossover,
    "uniform": UniformCrossover,
    "bitflip": BitFlipMutation,
    "swapbit": SwapBitMutation,
}


_REAL_FACTORIES: Mapping[str, Callable[[float], Operator[Any]]] = {
    "sbx": lambda span: SBX(),
    "arithmetic": lambda span: ArithmeticCrossover(),
    "gaussian": lambda span: GaussianMutation(sigma=0.1 * span),
    "polynomial": lambda span: PolynomialMutation(span=span),
}


def pool_ids(representation: Representation, variant: PoolVariant) -> tuple[str, ...]:
    """Return the operator IDs for the given representation and pool variant."""
    return POOL_MEMBERSHIP[representation][variant]


def build_pool(
    representation: Representation,
    variant: PoolVariant,
    *,
    real_bounds: tuple[float, float] | None = None,
) -> list[Operator[Any]]:
    """Return a list of operator instances for the given representation and pool variant.

    The operators are built with their standard parameters and returned in the membership
    order. Building a REAL pool requires ``real_bounds = (lower, upper)``: the polynomial
    mutation is scaled by ``span = upper - lower`` and the gaussian mutation by
    ``sigma = 0.1 * (upper - lower)`` (ten percent of the domain width), while ``sbx`` and
    ``polynomial`` keep the fixed ``eta = 20`` shared with the classic GA. Omitting
    ``real_bounds`` -- or passing non-increasing bounds -- for a REAL pool raises
    ``ValueError``; for the permutation and binary pools ``real_bounds`` is unused.
    """
    ids = pool_ids(representation, variant)

    if representation is Representation.REAL:
        if real_bounds is None:
            raise ValueError("real_bounds must be specified for real-valued operators")
        lower, upper = real_bounds
        if lower >= upper:
            raise ValueError(
                f"Invalid real_bounds: lower ({lower}) must be less than upper ({upper})"
            )

        span = upper - lower
        return [_REAL_FACTORIES[op_id](span) for op_id in ids]

    return [_DISCRETE_FACTORIES[op_id]() for op_id in ids]
