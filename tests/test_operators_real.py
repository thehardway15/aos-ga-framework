"""Contract spec for the real-valued operators of the continuous benchmark pool.

Four operators make up the full continuous pool: two crossovers -- Simulated Binary
Crossover (SBX) and arithmetic crossover -- and two
mutations -- polynomial mutation and gaussian mutation. The classic GA baseline (CGA
slice) uses SBX + polynomial; the reduced AOS pool is SBX + gaussian; arithmetic and
gaussian complete the full pool. All implement the frozen
:class:`~aos_ga.core.operator.Operator` interface over the ``list[float]`` real
representation (real-valued decision vectors).

The concrete classes are the executable target of this specification. Expected public
names (in ``aos_ga.operators.real``):

- ``SBX`` (``operator_id="sbx"``, recombinative, arity 2), distribution index ``eta=20``.
- ``ArithmeticCrossover`` (``operator_id="arithmetic"``, recombinative, arity 2).
- ``PolynomialMutation`` (``operator_id="polynomial"``, perturbative, arity 1),
  constructed ``PolynomialMutation(span, eta=20.0)`` -- ``span = u - l`` is required.
- ``GaussianMutation`` (``operator_id="gaussian"``, perturbative, arity 1),
  constructed ``GaussianMutation(sigma)`` -- the perturbation scale is required.

Frozen contract (specialises the operator interface for real genomes):
- ``apply(parents, rng) -> child`` returns ONE child that is a fresh, native ``list[float]``
  of the same length as its parents, drawing randomness only from the injected ``rng`` and
  being deterministic for a fixed seed.
- ``len(parents)`` must equal ``arity`` or ``ValueError`` is raised.
- The child is always a fresh list; parents are never mutated.

Domain handling (decision: the crossovers are domain-unaware and the continuous problem's
box-clip ``repair`` legalises an out-of-box child; the two mutations that scale with the
domain take that scale through the constructor, never through ``apply``):
- SBX draws a per-variable spread factor ``beta_i`` from the SBX distribution
  (``beta = (2u)**(1/(eta+1))`` for ``u <= 0.5``, else ``beta = (1/(2(1-u)))**(1/(eta+1))``)
  and forms the textbook offspring pair
  ``c1_i = 0.5[(1+beta_i) p1_i + (1-beta_i) p2_i]`` and
  ``c2_i = 0.5[(1-beta_i) p1_i + (1+beta_i) p2_i]``. There is NO boundary correction of
  ``beta`` (that would need the domain). It returns ONE child -- one of the two WHOLE
  offspring chosen uniformly at random -- so a single evaluation is spent. Because the pair
  is symmetric under swapping the parents, no extra "which parent first" bit is needed
  (unlike order crossover). Identical parents are reproduced: any ``beta`` leaves ``p`` fixed.
- Arithmetic crossover draws a per-variable weight ``alpha_i ~ U(0,1)`` and returns the
  convex combination ``alpha_i * p1_i + (1-alpha_i) * p2_i``. Because ``alpha`` is symmetric
  under ``alpha <-> 1-alpha``, one draw already samples the offspring pair uniformly, so no
  "which parent first" bit is needed (unlike single-point crossover). A convex blend stays
  within ``[min, max]`` on every axis, so -- unlike SBX -- arithmetic never spreads past the
  parents and, given in-box parents, never leaves the box. It takes an intermediate value,
  the contrast with binary uniform crossover, which copies one whole parent's value per axis.
- Polynomial mutation visits each variable independently with probability ``1/d`` for
  ``d = len(parent)`` measured inside the operator, and perturbs a visited variable by
  ``x_i + span * delta`` where ``delta`` is drawn from the polynomial distribution
  (``delta = (2r)**(1/(eta+1)) - 1`` for ``r < 0.5``, else
  ``delta = 1 - (2(1-r))**(1/(eta+1))``, so ``delta`` in ``[-1, 1)``) and ``span = u - l`` is
  the domain width passed to the constructor (standard Deb form ``x_i + (u-l) delta_i``).
  Every coordinate therefore moves by at most ``span``. There is NO guarantee that any
  variable is visited, so ``child == parent`` is a legal outcome -- the contrast with the
  permutation inversion operator, which always changes the genome.
- Gaussian mutation perturbs EVERY variable by ``x_i + sigma * z_i`` with ``z_i ~ N(0,1)``
  and ``sigma`` the perturbation scale passed to the constructor (a tenth of the domain
  width). Because the noise is continuous and applied on every axis, an
  application always changes the genome (the contrast with polynomial's 1/d, which may skip
  every variable), and ``z``'s unbounded support lets a step exceed any fixed bound, so an
  out-of-box child is expected and left to repair. No boundary correction is applied.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.real import (
    SBX,
    ArithmeticCrossover,
    GaussianMutation,
    PolynomialMutation,
)

_SBX = SBX()
_ARITHMETIC = ArithmeticCrossover()
# Concrete scales for the two domain-aware mutations: a span > 1 makes the polynomial's
# (u-l) factor visible (steps can exceed 1, impossible for the old bounds-free variant), and
# sigma = 1 makes the gaussian scale checks read directly.
_POLY_SPAN = 4.0
_POLY = PolynomialMutation(span=_POLY_SPAN)
_GAUSSIAN_SIGMA = 1.0
_GAUSSIAN = GaussianMutation(sigma=_GAUSSIAN_SIGMA)
_REAL_OPERATORS: list[Operator[list[float]]] = [_SBX, _ARITHMETIC, _POLY, _GAUSSIAN]


# --- Test oracles (verifiers, not generators) ----------------------------------


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


def _is_arithmetic_child(child: list[float], first: list[float], second: list[float]) -> bool:
    """True if ``child`` is a per-axis convex combination of ``first`` and ``second``.

    Each coordinate of an arithmetic child is ``alpha_i * first_i + (1-alpha_i) * second_i``
    with ``alpha_i`` in ``[0, 1]``, which is exactly the closed interval
    ``[min(first_i, second_i), max(first_i, second_i)]``. Axes where the parents agree pin the
    child to that shared value. This checks the operator's output without reproducing its
    random ``alpha`` draws.
    """
    if len(child) != len(first) or len(child) != len(second):
        return False
    tol = 1e-9
    for c, a, b in zip(child, first, second, strict=True):
        low, high = (a, b) if a <= b else (b, a)
        if not (low - tol <= c <= high + tol):
            return False
    return True


# --- Oracle self-checks (the verifiers above must themselves be correct) ------------

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


def test_arithmetic_oracle_accepts_a_convex_combination() -> None:
    # The midpoint and a per-axis lean both lie inside the parent interval on every axis.
    assert _is_arithmetic_child([0.0, 0.0], _ORACLE_FIRST, _ORACLE_SECOND)
    assert _is_arithmetic_child([0.5, -1.5], _ORACLE_FIRST, _ORACLE_SECOND)


def test_arithmetic_oracle_rejects_a_value_outside_the_parents() -> None:
    # 2.0 on axis 0 lies beyond `first` (1.0): a spread past the parents that a convex
    # combination can never produce -- that is SBX's behaviour, not arithmetic's.
    assert not _is_arithmetic_child([2.0, 0.0], _ORACLE_FIRST, _ORACLE_SECOND)


# --- Sample parents ------------------------------------------------------------


def _sample_parents(operator: Operator[list[float]]) -> list[list[float]]:
    """Two equal-length real vectors, trimmed to the operator's arity.

    They differ on every axis (and are symmetric about the origin), so the blend oracles can
    read each coordinate's lean and every axis carries recombination information.
    """
    parents = [
        [1.0, -2.0, 3.5, -0.5, 4.0, -3.0, 2.5, -1.5],
        [-1.0, 2.0, -3.5, 0.5, -4.0, 3.0, -2.5, 1.5],
    ]
    return parents[: operator.arity]


# A longer vector for the statistical mutation checks: d = 10 gives a per-variable rate
# 1/d = 0.1 and thus one visited variable per polynomial application on average.
_MUTATION_PARENT = [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 0.25, -0.25]


def _changed_count(parent: list[float], child: list[float]) -> int:
    """Number of coordinates the operator actually moved."""
    return sum(1 for a, b in zip(parent, child, strict=True) if a != b)


# --- Metadata ------------------------------------------------------------------


def test_operators_are_operator_instances() -> None:
    for operator in _REAL_OPERATORS:
        assert isinstance(operator, Operator)


def test_sbx_metadata() -> None:
    assert _SBX.operator_id == "sbx"
    assert _SBX.representation is Representation.REAL
    assert _SBX.arity == 2
    assert _SBX.kind is OperatorKind.RECOMBINATIVE


def test_arithmetic_metadata() -> None:
    assert _ARITHMETIC.operator_id == "arithmetic"
    assert _ARITHMETIC.representation is Representation.REAL
    assert _ARITHMETIC.arity == 2
    assert _ARITHMETIC.kind is OperatorKind.RECOMBINATIVE


def test_polynomial_metadata() -> None:
    assert _POLY.operator_id == "polynomial"
    assert _POLY.representation is Representation.REAL
    assert _POLY.arity == 1
    assert _POLY.kind is OperatorKind.PERTURBATIVE


def test_gaussian_metadata() -> None:
    assert _GAUSSIAN.operator_id == "gaussian"
    assert _GAUSSIAN.representation is Representation.REAL
    assert _GAUSSIAN.arity == 1
    assert _GAUSSIAN.kind is OperatorKind.PERTURBATIVE


# --- Arity enforcement ---------------------------------------------------------


def test_sbx_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _SBX.apply([[1.0, 0.0, 1.0]], np.random.default_rng(0))  # one parent, needs two


def test_arithmetic_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _ARITHMETIC.apply([[1.0, 0.0, 1.0]], np.random.default_rng(0))  # one parent, needs two


def test_polynomial_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _POLY.apply([[1.0, 0.0], [0.0, 1.0]], np.random.default_rng(0))  # two, needs one


def test_gaussian_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _GAUSSIAN.apply([[1.0, 0.0], [0.0, 1.0]], np.random.default_rng(0))  # two, needs one


# --- Shared contract: one fresh real child of equal length, parents untouched ---


@pytest.mark.parametrize("operator", _REAL_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_returns_a_fresh_real_child(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert all(isinstance(value, float) for value in child)  # a real vector
    assert len(child) == len(parents[0])  # one child, same genome length
    for parent in parents:
        assert child is not parent  # a fresh genome, never an aliased parent


@pytest.mark.parametrize("operator", _REAL_OPERATORS, ids=lambda op: op.operator_id)
def test_children_are_native_python_floats(operator: Operator[list[float]]) -> None:
    # The contract returns a native ``list[float]`` (as the continuous problem's repair does
    # via ``.tolist()``), so NumPy scalars must not leak into the genome. ``np.float64`` is a
    # subclass of ``float``, so ``isinstance`` would miss it -- check the exact type.
    child = operator.apply(_sample_parents(operator), np.random.default_rng(0))
    assert all(type(value) is float for value in child)


@pytest.mark.parametrize("operator", _REAL_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_does_not_mutate_parents(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    snapshot = [list(parent) for parent in parents]
    operator.apply(parents, np.random.default_rng(1))
    assert parents == snapshot


@pytest.mark.parametrize("operator", _REAL_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_is_deterministic_for_the_same_seed(operator: Operator[list[float]]) -> None:
    parents = _sample_parents(operator)
    first = operator.apply(parents, np.random.default_rng(7))
    second = operator.apply(parents, np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("operator", _REAL_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_uses_only_the_injected_generator(operator: Operator[list[float]]) -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    parents = _sample_parents(operator)
    before = pickle.dumps(np.random.get_state())
    operator.apply(parents, np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("operator", _REAL_OPERATORS, ids=lambda op: op.operator_id)
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


# --- Arithmetic: convex blend, one whole child of a pair, always within the parent box ---


def test_arithmetic_output_is_a_convex_combination_across_seeds() -> None:
    first, second = _sample_parents(_ARITHMETIC)
    for seed in range(32):
        child = _ARITHMETIC.apply([first, second], np.random.default_rng(seed))
        assert _is_arithmetic_child(child, first, second)


def test_arithmetic_never_spreads_beyond_the_parents() -> None:
    # A convex blend stays within [min, max] on every axis -- the sharp contrast with SBX,
    # which can push a child outside the parent interval. So arithmetic (given in-box parents)
    # never leaves the box and its repair is a no-op.
    first, second = _sample_parents(_ARITHMETIC)
    lows = [min(a, b) for a, b in zip(first, second, strict=True)]
    highs = [max(a, b) for a, b in zip(first, second, strict=True)]
    for seed in range(32):
        child = _ARITHMETIC.apply([first, second], np.random.default_rng(seed))
        assert all(lows[i] - 1e-9 <= child[i] <= highs[i] + 1e-9 for i in range(len(child)))


def test_arithmetic_produces_intermediate_values() -> None:
    # Unlike binary uniform (which copies one parent's whole value per axis), arithmetic takes
    # an intermediate value: on an axis where the parents differ, some child coordinate is
    # strictly between them, equal to neither parent.
    first, second = _sample_parents(_ARITHMETIC)
    strictly_interior = False
    for seed in range(32):
        child = _ARITHMETIC.apply([first, second], np.random.default_rng(seed))
        for c, a, b in zip(child, first, second, strict=True):
            low, high = (a, b) if a < b else (b, a)
            if low < c < high:
                strictly_interior = True
    assert strictly_interior


def test_arithmetic_reaches_both_sides_of_the_pair() -> None:
    # alpha_i ~ U(0,1) is symmetric under alpha <-> 1-alpha, so one draw already samples the
    # pair uniformly -- no head bit needed. Over seeds axis 0 leans both toward first and
    # toward second of the midpoint.
    first, second = _sample_parents(_ARITHMETIC)
    midpoint0 = 0.5 * (first[0] + second[0])
    children = [
        _ARITHMETIC.apply([first, second], np.random.default_rng(seed)) for seed in range(32)
    ]
    toward_first = any((c[0] - midpoint0) * (first[0] - second[0]) > 0 for c in children)
    toward_second = any((c[0] - midpoint0) * (first[0] - second[0]) < 0 for c in children)
    assert toward_first and toward_second


def test_arithmetic_preserves_identical_parents() -> None:
    # With p1 == p2 the blend collapses: alpha*p + (1-alpha)*p = p for any alpha, so any weight
    # reproduces the parent (up to floating-point rounding).
    parent = [1.0, -2.0, 3.5, -0.5, 4.0, -3.0, 2.5, -1.5]
    for seed in range(16):
        child = _ARITHMETIC.apply([list(parent), list(parent)], np.random.default_rng(seed))
        assert child == pytest.approx(parent)


def test_arithmetic_draws_a_weight_per_variable() -> None:
    # alpha is drawn PER VARIABLE (alpha_i), so within one child the
    # recovered weight (child_i - second_i) / (first_i - second_i) differs across axes: the
    # child fills the box spanned by the parents, not just the line segment between them that a
    # single shared alpha would trace. A whole-vector (one shared alpha) implementation gives
    # the same weight on every axis and fails this.
    first, second = _sample_parents(_ARITHMETIC)
    varied = False
    for seed in range(32):
        child = _ARITHMETIC.apply([first, second], np.random.default_rng(seed))
        weights = [
            (c - b) / (a - b) for c, a, b in zip(child, first, second, strict=True) if a != b
        ]
        if max(weights) - min(weights) > 1e-6:
            varied = True
            break
    assert varied


# --- Polynomial mutation: 1/d visits, span-scaled delta, no forced change -----------


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


def test_polynomial_perturbation_is_bounded_by_span() -> None:
    # The polynomial step delta lies in [-1, 1) and is scaled by the domain span passed to the
    # constructor (standard Deb form x_i + (u-l) delta_i), so every coordinate moves by at most
    # ``span``. The epsilon absorbs floating-point noise in ``child - parent``.
    parent = _MUTATION_PARENT
    for seed in range(64):
        child = _POLY.apply([list(parent)], np.random.default_rng(seed))
        assert all(abs(c - p) <= _POLY_SPAN + 1e-9 for p, c in zip(parent, child, strict=True))


def test_polynomial_step_scales_linearly_with_span() -> None:
    # The step is span * delta, linear in span: span is a constructor constant and draws no
    # randomness, so with the same seed both operators visit the same variables with the same
    # delta. Tripling the span therefore triples every coordinate's move. This pins the (u-l)
    # factor that the domain-aware polynomial restores (a bounds-free variant would ignore it).
    parent = _MUTATION_PARENT
    poly1 = PolynomialMutation(span=1.0)
    poly3 = PolynomialMutation(span=3.0)
    for seed in range(16):
        child1 = poly1.apply([list(parent)], np.random.default_rng(seed))
        child3 = poly3.apply([list(parent)], np.random.default_rng(seed))
        for p, a, b in zip(parent, child1, child3, strict=True):
            assert (b - p) == pytest.approx(3.0 * (a - p))


def test_polynomial_moves_in_both_directions() -> None:
    # delta is symmetric about 0 (r < 0.5 lowers a variable, r > 0.5 raises it), so over seeds
    # the operator perturbs coordinates both up and down, not in one fixed direction.
    parent = _MUTATION_PARENT
    children = [_POLY.apply([list(parent)], np.random.default_rng(seed)) for seed in range(64)]
    deltas = [c - p for child in children for p, c in zip(parent, child, strict=True)]
    assert any(d > 0 for d in deltas)  # some coordinate raised
    assert any(d < 0 for d in deltas)  # some coordinate lowered


# --- Gaussian mutation: every variable, sigma-scaled normal noise, unbounded step ---


def test_gaussian_perturbs_every_variable_across_seeds() -> None:
    # Gaussian adds sigma*z_i to EVERY coordinate (not a 1/d subset like polynomial), and
    # z_i = 0 has probability zero, so all d coordinates move on every application.
    parent = _MUTATION_PARENT
    for seed in range(32):
        child = _GAUSSIAN.apply([list(parent)], np.random.default_rng(seed))
        assert _changed_count(parent, child) == len(parent)


def test_gaussian_always_changes_the_parent() -> None:
    # Continuous noise on every axis: child == parent is a probability-zero event, so every
    # application is a real change -- the contrast with polynomial mutation, which can skip
    # every variable and leave the genome unchanged.
    parent = _MUTATION_PARENT
    for seed in range(32):
        child = _GAUSSIAN.apply([list(parent)], np.random.default_rng(seed))
        assert child != parent


def test_gaussian_perturbation_scale_matches_sigma() -> None:
    # The per-coordinate step is sigma*z with z ~ N(0,1), so the sample standard deviation of
    # (child - parent) over many draws pins the scale at sigma. Loose bounds absorb sampling
    # noise; many seeds keep the estimate tight.
    parent = _MUTATION_PARENT
    n_seeds = 200
    deltas = [
        c - p
        for seed in range(n_seeds)
        for p, c in zip(
            parent, _GAUSSIAN.apply([list(parent)], np.random.default_rng(seed)), strict=True
        )
    ]
    mean = sum(deltas) / len(deltas)
    std = (sum((d - mean) ** 2 for d in deltas) / len(deltas)) ** 0.5
    assert 0.85 * _GAUSSIAN_SIGMA <= std <= 1.15 * _GAUSSIAN_SIGMA


def test_gaussian_step_is_unbounded() -> None:
    # z has unbounded support, so a step can exceed 2*sigma -- impossible for polynomial
    # mutation, whose step is bounded by the domain span. This is why an out-of-box gaussian
    # child is expected and left to the box-clip repair.
    parent = _MUTATION_PARENT
    assert any(
        abs(c - p) > 2.0 * _GAUSSIAN_SIGMA
        for seed in range(64)
        for p, c in zip(
            parent, _GAUSSIAN.apply([list(parent)], np.random.default_rng(seed)), strict=True
        )
    )


def test_gaussian_moves_in_both_directions() -> None:
    # z is symmetric about 0, so over seeds the operator perturbs coordinates both up and down.
    parent = _MUTATION_PARENT
    deltas = [
        c - p
        for seed in range(32)
        for p, c in zip(
            parent, _GAUSSIAN.apply([list(parent)], np.random.default_rng(seed)), strict=True
        )
    ]
    assert any(d > 0 for d in deltas)  # some coordinate raised
    assert any(d < 0 for d in deltas)  # some coordinate lowered
