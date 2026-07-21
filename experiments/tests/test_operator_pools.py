"""Contract spec for the study's operator-pool registry.

The registry turns the fixed full and reduced operator pools into two separable pieces:

- **Membership** -- a pure configuration mapping ``Representation -> {FULL, REDUCED}`` to a
  tuple of ``operator_id`` strings. It names the arms of the AOS problem without ever
  instantiating an operator, so a strategy or a log can enumerate a pool by id alone.
- **Factory** -- ``build_pool`` instantiates the operators of a chosen pool. It is the only
  place that knows how to construct each operator, including the domain scaling the two
  real-valued mutations need.

The two are deliberately decoupled: membership is data (comparable to the reference table),
construction is behaviour. The pools compose the framework's operators but the composition is
a treatment of this study, so it lives in the experiment layer (``experiments.configs.pools``),
not in ``aos_ga``.

The concrete module is the executable target of this specification. Expected public names
(in ``experiments.configs.pools``):

- ``PoolVariant`` -- an ``Enum`` with members ``FULL`` and ``REDUCED``.
- ``POOL_MEMBERSHIP`` -- a mapping ``Representation -> {PoolVariant -> tuple[str, ...]}``.
- ``pool_ids(representation, variant) -> tuple[str, ...]`` -- the ids of one pool, in the
  table's order (the order the AOS engine indexes its arms by, hence fixed).
- ``build_pool(representation, variant, *, real_bounds=None) -> list[Operator[Any]]`` --
  the operator instances of one pool, in the same order as ``pool_ids``.

Frozen contract:
- ``POOL_MEMBERSHIP`` and ``pool_ids`` agree exactly with the reference table below,
  order included, and hold only ``operator_id`` strings -- never operator instances.
- Every reduced pool is a subset (by id) of its full pool, and keeps at least one
  recombinative and at least one perturbative operator, so the AOS strategy still chooses
  between distinct search roles (the reduction rationale, not an arbitrary trimming).
- ``build_pool`` returns fresh operator instances whose ``operator_id`` sequence equals
  ``pool_ids`` and whose ``representation`` is the pool's representation.
- Real-pool construction is domain-aware and requires ``real_bounds = (lower, upper)``:
  the polynomial mutation is built with ``span = upper - lower`` and the gaussian mutation
  with ``sigma = 0.1 * (upper - lower)`` (a tenth of the domain width); ``sbx`` and
  ``polynomial`` keep the fixed ``eta = 20`` shared with the classic GA. A REAL pool built
  without ``real_bounds`` -- or with non-increasing bounds -- raises ``ValueError``.
- For the permutation and binary pools ``real_bounds`` is unused and silently ignored.
"""

from __future__ import annotations

from typing import Any

import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.real import SBX, GaussianMutation, PolynomialMutation
from experiments.configs.pools import POOL_MEMBERSHIP, PoolVariant, build_pool, pool_ids

# --- The reference pool table, transcribed as the executable specification ---------
#
# The reference table gives the pools by operator name; the ids below are the
# ``operator_id`` of each named operator. Name/id subtleties baked into this table:
# the knapsack "swap" is the binary ``swapbit`` (distinct from the permutation ``swap``),
# "single-point" is ``singlepoint`` and "bit-flip" is ``bitflip``. The tuple order mirrors
# the table's left-to-right order and is itself part of the contract.
EXPECTED_MEMBERSHIP: dict[Representation, dict[PoolVariant, tuple[str, ...]]] = {
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

# Every (representation, variant) case, for the pool-agnostic parametrised checks.
_ALL_CASES: list[tuple[Representation, PoolVariant]] = [
    (representation, variant) for representation in Representation for variant in PoolVariant
]
_CASE_IDS: list[str] = [f"{rep.value}-{variant.value}" for rep, variant in _ALL_CASES]

# The Sphere/Rastrigin domain, used wherever a REAL pool must be built.
# span = 10.24 and sigma = 1.024 fall straight out of these bounds.
_REAL_BOUNDS: tuple[float, float] = (-5.12, 5.12)


def _build(representation: Representation, variant: PoolVariant) -> list[Operator[Any]]:
    """Build a pool, supplying ``real_bounds`` only where the representation needs them."""
    bounds = _REAL_BOUNDS if representation is Representation.REAL else None
    return build_pool(representation, variant, real_bounds=bounds)


def _find(pool: list[Operator[Any]], operator_id: str) -> Operator[Any]:
    """Return the single operator with ``operator_id`` in ``pool`` (self-checking helper)."""
    matches = [op for op in pool if op.operator_id == operator_id]
    assert len(matches) == 1, f"expected exactly one {operator_id!r}, got {len(matches)}"
    return matches[0]


# --- Membership is a pure, table-faithful configuration of ids ---------------------


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_pool_ids_match_the_reference_table(
    representation: Representation, variant: PoolVariant
) -> None:
    # The heart of the contract: each pool's ids equal the transcribed table, ORDER included
    # (tuple equality is order-sensitive), because the AOS engine indexes arms by this order.
    assert pool_ids(representation, variant) == EXPECTED_MEMBERSHIP[representation][variant]


def test_pool_membership_constant_matches_the_reference_table() -> None:
    # The public constant is the same data the accessor serves, so a consumer may read either.
    assert POOL_MEMBERSHIP == EXPECTED_MEMBERSHIP


def test_pool_membership_covers_every_representation_and_variant() -> None:
    # No representation or variant may be silently missing from the registry.
    assert set(POOL_MEMBERSHIP) == set(Representation)
    for variants in POOL_MEMBERSHIP.values():
        assert set(variants) == set(PoolVariant)


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_membership_holds_only_string_ids_not_operators(
    representation: Representation, variant: PoolVariant
) -> None:
    # Membership is configuration, decoupled from the factory: it names arms by id and never
    # constructs an operator (which for REAL would need the domain bounds it does not have).
    ids = pool_ids(representation, variant)
    assert all(isinstance(operator_id, str) for operator_id in ids)


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_ids_are_unique_within_a_pool(representation: Representation, variant: PoolVariant) -> None:
    # A well-formed pool lists each arm once.
    ids = pool_ids(representation, variant)
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("representation", list(Representation), ids=lambda r: r.value)
def test_reduced_pool_is_a_subset_of_the_full_pool(representation: Representation) -> None:
    # The reduced pool drops arms from the full pool; it never introduces a new one.
    full = set(pool_ids(representation, PoolVariant.FULL))
    reduced = set(pool_ids(representation, PoolVariant.REDUCED))
    assert reduced <= full


# --- Reduced pools keep both search roles (the reduction rationale) ----------------


@pytest.mark.parametrize("representation", list(Representation), ids=lambda r: r.value)
def test_reduced_pool_keeps_both_search_roles(representation: Representation) -> None:
    # The reduction was not arbitrary: each reduced pool retains at least one recombinative and
    # at least one perturbative operator, so the AOS strategy still chooses between roles rather
    # than among near-identical operators. Read the roles off the built operators.
    kinds = {op.kind for op in _build(representation, PoolVariant.REDUCED)}
    assert OperatorKind.RECOMBINATIVE in kinds
    assert OperatorKind.PERTURBATIVE in kinds


# --- build_pool: instances faithful to the membership ------------------------------


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_build_pool_returns_operator_instances(
    representation: Representation, variant: PoolVariant
) -> None:
    pool = _build(representation, variant)
    assert isinstance(pool, list)
    assert all(isinstance(op, Operator) for op in pool)


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_build_pool_ids_match_membership_in_order(
    representation: Representation, variant: PoolVariant
) -> None:
    # The built operators are exactly the membership arms, in the membership order -- the
    # factory neither reorders, drops nor adds operators.
    pool = _build(representation, variant)
    assert [op.operator_id for op in pool] == list(pool_ids(representation, variant))


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_build_pool_operators_carry_the_pool_representation(
    representation: Representation, variant: PoolVariant
) -> None:
    # Every operator the factory builds for a pool belongs to that pool's representation.
    pool = _build(representation, variant)
    assert all(op.representation is representation for op in pool)


@pytest.mark.parametrize(("representation", "variant"), _ALL_CASES, ids=_CASE_IDS)
def test_build_pool_returns_fresh_instances(
    representation: Representation, variant: PoolVariant
) -> None:
    # Each call yields independent operator objects rather than shared singletons, so a run can
    # never leak mutable operator state into another pool.
    first = _build(representation, variant)
    second = _build(representation, variant)
    assert all(a is not b for a, b in zip(first, second, strict=True))


# --- REAL pool: domain-aware construction and its required bounds ------------------


@pytest.mark.parametrize("variant", list(PoolVariant), ids=lambda v: v.value)
def test_build_real_pool_requires_bounds(variant: PoolVariant) -> None:
    # A REAL pool cannot be scaled without the domain width, so omitting ``real_bounds`` is a
    # loud error, never a silently mis-scaled operator.
    with pytest.raises(ValueError):
        build_pool(Representation.REAL, variant)


@pytest.mark.parametrize(
    "bounds", [(1.0, 1.0), (2.0, 1.0)], ids=["equal-bounds", "reversed-bounds"]
)
def test_build_real_pool_rejects_non_increasing_bounds(bounds: tuple[float, float]) -> None:
    # ``lower < upper`` is required: a zero or negative span would give degenerate operators, so
    # it is rejected rather than propagated.
    with pytest.raises(ValueError):
        build_pool(Representation.REAL, PoolVariant.FULL, real_bounds=bounds)


def test_build_real_pool_scales_polynomial_span_from_bounds() -> None:
    # The polynomial mutation is built with ``span = upper - lower`` (Deb form
    # ``x_i + (u - l) delta_i``); the factory, not the operator, computes the scale.
    lower, upper = _REAL_BOUNDS
    pool = build_pool(Representation.REAL, PoolVariant.FULL, real_bounds=_REAL_BOUNDS)
    polynomial = _find(pool, "polynomial")
    assert isinstance(polynomial, PolynomialMutation)
    assert polynomial.span == pytest.approx(upper - lower)


def test_build_real_pool_scales_gaussian_sigma_from_bounds() -> None:
    # The gaussian mutation is built with ``sigma = 0.1 * (upper - lower)`` (ten percent of
    # the domain width).
    lower, upper = _REAL_BOUNDS
    pool = build_pool(Representation.REAL, PoolVariant.FULL, real_bounds=_REAL_BOUNDS)
    gaussian = _find(pool, "gaussian")
    assert isinstance(gaussian, GaussianMutation)
    assert gaussian.sigma == pytest.approx(0.1 * (upper - lower))


def test_build_real_pool_uses_the_fixed_distribution_index() -> None:
    # ``eta = 20`` is fixed for sbx and polynomial and shared with the classic GA (the only
    # intended difference between AOS and CGA is operator selection, not operator parameters),
    # so the factory does not expose it -- the built operators simply carry the standard value.
    pool = build_pool(Representation.REAL, PoolVariant.FULL, real_bounds=_REAL_BOUNDS)
    sbx = _find(pool, "sbx")
    polynomial = _find(pool, "polynomial")
    assert isinstance(sbx, SBX)
    assert isinstance(polynomial, PolynomialMutation)
    assert sbx.eta == 20
    assert polynomial.eta == 20


# --- Permutation and binary pools ignore real_bounds -------------------------------


@pytest.mark.parametrize(
    "representation",
    [Representation.PERMUTATION, Representation.BINARY],
    ids=lambda r: r.value,
)
def test_non_real_pool_ignores_real_bounds(representation: Representation) -> None:
    # ``real_bounds`` is meaningful only for the two domain-scaled real mutations; for the other
    # representations it is unused, so passing it is neither an error nor a change of result.
    without_bounds = build_pool(representation, PoolVariant.FULL)
    with_bounds = build_pool(representation, PoolVariant.FULL, real_bounds=_REAL_BOUNDS)
    assert [op.operator_id for op in without_bounds] == [op.operator_id for op in with_bounds]
