"""Contract spec for the reduced-pool real-valued operators used by the continuous CGA slice.

Two operators are pinned here -- Simulated Binary Crossover (SBX) and Polynomial Mutation --
because the classic GA baseline on the continuous benchmark functions uses exactly this pair:
SBX for recombination and polynomial mutation for perturbation. They implement the frozen
:class:`~aos_ga.core.operator.Operator` interface over the ``list[float]`` real representation
(real-valued decision vectors).

The concrete classes are not implemented yet: this file is the executable specification of
their behaviour. Expected public names (in ``aos_ga.operators.real``): ``SBX``
(``operator_id="sbx"``, recombinative, arity 2) and ``PolynomialMutation``
(``operator_id="polynomial"``, perturbative, arity 1). Both take a distribution index ``eta``
defaulting to 20.0.

Frozen contract (specialises the operator interface for real genomes):
- ``apply(parents, rng) -> child`` returns ONE child that is a fresh, native ``list[float]``
  of the same length as its parents, drawing randomness only from the injected ``rng`` and
  being deterministic for a fixed seed.
- ``len(parents)`` must equal ``arity`` or ``ValueError`` is raised.
- The child is always a fresh list; parents are never mutated.

Unbounded variant (decision: operators know no bounds; the continuous problem's box-clip
``repair`` legalises the child):
- SBX draws a per-variable spread factor ``beta_i`` from the SBX distribution
  (``beta = (2u)**(1/(eta+1))`` for ``u <= 0.5``, else ``beta = (1/(2(1-u)))**(1/(eta+1))``)
  and forms the textbook offspring pair
  ``c1_i = 0.5[(1+beta_i) p1_i + (1-beta_i) p2_i]`` and
  ``c2_i = 0.5[(1-beta_i) p1_i + (1+beta_i) p2_i]``. There is NO boundary correction of
  ``beta`` (that would need the domain). It returns ONE child -- one of the two WHOLE
  offspring chosen uniformly at random -- so a single evaluation is spent. Because the pair
  is symmetric under swapping the parents, no extra "which parent first" bit is needed
  (unlike order crossover). Identical parents are reproduced: any ``beta`` leaves ``p`` fixed.
- Polynomial mutation visits each variable independently with probability ``1/d`` for
  ``d = len(parent)`` measured inside the operator, and perturbs a visited variable by
  ``x_i + delta`` where ``delta`` is drawn from the polynomial distribution
  (``delta = (2r)**(1/(eta+1)) - 1`` for ``r < 0.5``, else
  ``delta = 1 - (2(1-r))**(1/(eta+1))``), so ``delta`` lies in ``[-1, 1)``. The perturbation
  is applied DIRECTLY, with no domain-width scaling (that scaling is reserved for the deferred
  gaussian mutation), so every coordinate moves by at most 1. There is NO guarantee that any
  variable is visited, so ``child == parent`` is a legal outcome -- the contrast with the
  permutation inversion operator, which always changes the genome.

The rest of the real pool (arithmetic crossover and gaussian mutation for the full pool) is
out of the slice and specified when it is assembled.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.real import SBX, PolynomialMutation

_SBX = SBX()
_POLY = PolynomialMutation()
_SLICE_OPERATORS: list[Operator[list[float]]] = [_SBX, _POLY]


# --- Test oracle (verifier, not generator) -------------------------------------


def _is_sbx_child(child: list[float], first: list[float], second: list[float]) -> bool:
    """True if ``child`` is one whole SBX offspring of ``first`` and ``second``.

    Each SBX coordinate sits on the line through the parents at the midpoint
    ``m_i = (first_i + second_i) / 2``, displaced by ``0.5 * beta_i * (first_i - second_i)``
    with ``beta_i >= 0``. Because ONE whole offspring is chosen (``c1`` or ``c2``, never a
    per-variable mix), the displacement is consistently signed relative to
    ``first_i - second_i`` across all coordinates: either every coordinate leans to ``first``'s
    side (``child`` is ``c1``) or every coordinate leans to ``second``'s side (``child`` is
    ``c2``). Coordinates where the parents agree carry no information and are ignored, as are
    near-zero displacements (``beta_i`` ~ 0), which are sign-ambiguous.

    This checks the operator's output without reproducing its random ``beta`` draws.
    """
    if len(child) != len(first) or len(child) != len(second):
        return False
    products = [
        (child[i] - 0.5 * (first[i] + second[i])) * (first[i] - second[i])
        for i in range(len(child))
        if first[i] != second[i]
    ]
    tol = 1e-9 * max((abs(p) for p in products), default=1.0)
    leans_first = any(p > tol for p in products)
    leans_second = any(p < -tol for p in products)
    return not (leans_first and leans_second)


# --- Oracle self-checks (the verifier above must itself be correct) ------------

# Parents symmetric about the origin, so the midpoint is 0 on every axis and a child
# coordinate's sign alone reveals which parent it leans toward.
_ORACLE_FIRST = [1.0, 3.0]
_ORACLE_SECOND = [-1.0, -3.0]


def test_sbx_oracle_accepts_offspring_leaning_to_one_parent() -> None:
    # Every coordinate pushed to `first`'s side (a valid c1, e.g. beta = 2 on both axes).
    assert _is_sbx_child([2.0, 6.0], _ORACLE_FIRST, _ORACLE_SECOND)
    # Every coordinate pushed to `second`'s side (a valid c2).
    assert _is_sbx_child([-2.0, -6.0], _ORACLE_FIRST, _ORACLE_SECOND)
    # The midpoint (beta = 0 everywhere) is a degenerate but valid offspring.
    assert _is_sbx_child([0.0, 0.0], _ORACLE_FIRST, _ORACLE_SECOND)


def test_sbx_oracle_rejects_a_per_variable_mix() -> None:
    # Axis 0 leans to `first` (+2) while axis 1 leans to `second` (-6): a per-variable mix
    # that is neither c1 nor c2, so it cannot be a single SBX offspring.
    assert not _is_sbx_child([2.0, -6.0], _ORACLE_FIRST, _ORACLE_SECOND)


# --- Sample parents ------------------------------------------------------------


def _sample_parents(operator: Operator[list[float]]) -> list[list[float]]:
    """Two equal-length real vectors, trimmed to the operator's arity.

    They differ on every axis (and are symmetric about the origin), so the SBX oracle can
    read each coordinate's lean and every axis carries recombination information.
    """
    parents = [
        [1.0, -2.0, 3.5, -0.5, 4.0, -3.0, 2.5, -1.5],
        [-1.0, 2.0, -3.5, 0.5, -4.0, 3.0, -2.5, 1.5],
    ]
    return parents[: operator.arity]


# A longer vector for the statistical mutation checks: d = 10 gives a per-variable rate
# 1/d = 0.1 and thus one visited variable per application on average.
_MUTATION_PARENT = [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 0.25, -0.25]


def _changed_count(parent: list[float], child: list[float]) -> int:
    """Number of coordinates the operator actually moved."""
    return sum(1 for a, b in zip(parent, child, strict=True) if a != b)


# --- Metadata ------------------------------------------------------------------


def test_operators_are_operator_instances() -> None:
    for operator in _SLICE_OPERATORS:
        assert isinstance(operator, Operator)


def test_sbx_metadata() -> None:
    assert _SBX.operator_id == "sbx"
    assert _SBX.representation is Representation.REAL
    assert _SBX.arity == 2
    assert _SBX.kind is OperatorKind.RECOMBINATIVE


def test_polynomial_metadata() -> None:
    assert _POLY.operator_id == "polynomial"
    assert _POLY.representation is Representation.REAL
    assert _POLY.arity == 1
    assert _POLY.kind is OperatorKind.PERTURBATIVE


# --- Arity enforcement ---------------------------------------------------------


def test_sbx_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _SBX.apply([[1.0, 0.0, 1.0]], np.random.default_rng(0))  # one parent, needs two


def test_polynomial_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _POLY.apply([[1.0, 0.0], [0.0, 1.0]], np.random.default_rng(0))  # two, needs one


# --- Shared contract: one fresh real child of equal length, parents untouched ---


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_returns_a_fresh_real_child(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert all(isinstance(value, float) for value in child)  # a real vector
    assert len(child) == len(parents[0])  # one child, same genome length
    for parent in parents:
        assert child is not parent  # a fresh genome, never an aliased parent


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_children_are_native_python_floats(operator: Operator[list[float]]) -> None:
    # The contract returns a native ``list[float]`` (as the continuous problem's repair does
    # via ``.tolist()``), so NumPy scalars must not leak into the genome. ``np.float64`` is a
    # subclass of ``float``, so ``isinstance`` would miss it -- check the exact type.
    child = operator.apply(_sample_parents(operator), np.random.default_rng(0))
    assert all(type(value) is float for value in child)


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_does_not_mutate_parents(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    snapshot = [list(parent) for parent in parents]
    operator.apply(parents, np.random.default_rng(1))
    assert parents == snapshot


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_is_deterministic_for_the_same_seed(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    first = operator.apply(parents, np.random.default_rng(7))
    second = operator.apply(parents, np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_uses_only_the_injected_generator(operator: Operator[list[float]]) -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    parents = _sample_parents(operator)
    before = pickle.dumps(np.random.get_state())
    operator.apply(parents, np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_is_rng_driven_not_fixed(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    seen = {tuple(operator.apply(parents, np.random.default_rng(seed))) for seed in range(32)}
    assert len(seen) >= 2


# --- SBX: blended offspring, one whole child of a pair, may spread past the parents ---


def test_sbx_output_is_a_single_offspring_across_seeds() -> None:
    first, second = _sample_parents(_SBX)
    for seed in range(32):
        child = _SBX.apply([first, second], np.random.default_rng(seed))
        assert _is_sbx_child(child, first, second)


def test_sbx_returns_either_offspring_of_the_pair() -> None:
    # The single returned child is one of the two SBX offspring chosen uniformly, so over
    # seeds both leans appear: some children sit on `first`'s side of the midpoint, others on
    # `second`'s side. Read on axis 0, where the parents differ and the midpoint is 0.
    first, second = _sample_parents(_SBX)
    midpoint0 = 0.5 * (first[0] + second[0])
    children = [_SBX.apply([first, second], np.random.default_rng(seed)) for seed in range(32)]
    toward_first = any((c[0] - midpoint0) * (first[0] - second[0]) > 0 for c in children)
    toward_second = any((c[0] - midpoint0) * (first[0] - second[0]) < 0 for c in children)
    assert toward_first and toward_second


def test_sbx_preserves_identical_parents() -> None:
    # With p1 == p2 the blend collapses: 0.5[(1+beta)p + (1-beta)p] = p for any beta, so any
    # selector and any spread reproduce the parent (up to floating-point rounding).
    parent = [1.0, -2.0, 3.5, -0.5, 4.0, -3.0, 2.5, -1.5]
    for seed in range(16):
        child = _SBX.apply([list(parent), list(parent)], np.random.default_rng(seed))
        assert child == pytest.approx(parent)


def test_sbx_can_spread_beyond_the_parents() -> None:
    # SBX draws beta > 1 about half the time per variable (u > 0.5), pushing a child
    # coordinate outside the [min, max] parent interval on that axis -- the exploratory spread
    # that makes the box-clip repair necessary. A convex blend could never leave the interval.
    first, second = _sample_parents(_SBX)
    lows = [min(a, b) for a, b in zip(first, second, strict=True)]
    highs = [max(a, b) for a, b in zip(first, second, strict=True)]
    children = (_SBX.apply([first, second], np.random.default_rng(seed)) for seed in range(32))
    assert any(any(c[i] < lows[i] or c[i] > highs[i] for i in range(len(c))) for c in children)


# --- Polynomial mutation: independent 1/d visits, direct delta, no forced change ---


def test_polynomial_visits_about_one_variable_per_application() -> None:
    # Expected visited variables per application = d * (1/d) = 1. Averaging over many seeds
    # pins the 1/d per-variable rate and rules out a much higher one. Loose bounds keep the
    # check robust to sampling noise. A visited variable is one whose value changed.
    parent = _MUTATION_PARENT
    n_seeds = 300
    total_changed = sum(
        _changed_count(parent, _POLY.apply([list(parent)], np.random.default_rng(seed)))
        for seed in range(n_seeds)
    )
    assert 0.5 <= total_changed / n_seeds <= 1.5


def test_polynomial_can_leave_the_parent_unchanged() -> None:
    # With 1/d per variable and no forced move, every variable can be skipped, so child ==
    # parent is a legal outcome -- the deliberate contrast with inversion, which always changes
    # the genome.
    parent = _MUTATION_PARENT
    assert any(
        _POLY.apply([list(parent)], np.random.default_rng(seed)) == parent for seed in range(50)
    )


def test_polynomial_can_move_more_than_one_variable() -> None:
    # Variables are visited independently, so an application can move several at once -- it is
    # not a fixed single-variable move.
    parent = _MUTATION_PARENT
    assert any(
        _changed_count(parent, _POLY.apply([list(parent)], np.random.default_rng(seed))) >= 2
        for seed in range(50)
    )


def test_polynomial_perturbation_is_bounded_by_one() -> None:
    # The unbounded polynomial step delta lies in [-1, 1) and is applied directly, with no
    # domain-width scaling, so every coordinate moves by at most 1. This is the invariant of
    # the bounds-free variant (scaling by domain width is reserved for gaussian mutation). The
    # epsilon absorbs floating-point noise in ``child - parent``.
    parent = _MUTATION_PARENT
    for seed in range(64):
        child = _POLY.apply([list(parent)], np.random.default_rng(seed))
        assert all(abs(c - p) <= 1.0 + 1e-9 for p, c in zip(parent, child, strict=True))


def test_polynomial_moves_in_both_directions() -> None:
    # delta is symmetric about 0 (r < 0.5 lowers a variable, r > 0.5 raises it), so over seeds
    # the operator perturbs coordinates both up and down, not in one fixed direction.
    parent = _MUTATION_PARENT
    children = [_POLY.apply([list(parent)], np.random.default_rng(seed)) for seed in range(64)]
    deltas = [c - p for child in children for p, c in zip(parent, child, strict=True)]
    assert any(d > 0 for d in deltas)  # some coordinate raised
    assert any(d < 0 for d in deltas)  # some coordinate lowered
