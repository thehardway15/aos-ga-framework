"""Contract tests for the framework's variation-operator interface (T8).

These pin the public API of :mod:`aos_ga.core.operator`: the ``OperatorKind`` of
a variation operator (recombinative vs perturbative) and the abstract
``Operator`` base every concrete operator implements. One operator application is
the atomic unit of the shared AOS pool -- it consumes ``arity`` parents and
produces exactly one child to be evaluated: one operator = one child = one
evaluation.

The names these tests import are not implemented yet: this file is the
executable specification of the contract. Expected public names: ``Operator``,
``OperatorKind`` (in ``aos_ga.core.operator``); ``Representation`` comes from
``aos_ga.core.representation``. ``Operator`` is generic over the genome type, like
``Problem``.

Frozen contract (T8):
- ``apply(parents, rng) -> child`` returns exactly ONE child genome, drawing
  randomness only from the injected ``rng`` (no global state) and being
  deterministic for a fixed rng seed.
- ``len(parents)`` must equal ``arity`` (1 for perturbative, 2 for recombinative
  by the pool convention); a wrong count raises ``ValueError``.
- An operator never mutates its parents: the same parent -- and the elite -- is
  read many times per generation in the shared-pool model.
- If a crossover's textbook definition yields a pair of children, the operator
  returns ONE of them, chosen uniformly at random via ``rng``. The choice is
  fitness-agnostic: evaluating the discarded child would spend a second
  evaluation and break the one-evaluation-per-step budget, so it is never done.
- Legalizing a genome against a *problem's* constraints is the caller's job
  (``Problem.repair``), not the operator's: an operator knows only its
  representation, never a concrete problem.

The four operator doubles below are minimal -- two permutation operators (a
recombinative pair-picking crossover and a perturbative segment inversion), one
binary crossover and one real-valued mutation -- used only to exercise the
abstract contract across all three representations. The real operator pools
(OX/CX/inversion, ...) live in the ``operators`` layer and are tested there.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.random import Generator

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation


def _require_arity(parents: Sequence[object], arity: int) -> None:
    """Reject a parent count that does not match the operator's arity."""
    if len(parents) != arity:
        raise ValueError(f"expected {arity} parents, got {len(parents)}")


class _PairPickCrossover(Operator[list[int]]):
    """Permutation crossover double: forms a pair, returns ONE picked via rng.

    The two candidates are the parents copied verbatim -- enough to prove that
    exactly one child is returned and that the choice is rng-driven, without
    depending on any real crossover mechanics.
    """

    operator_id = "test-pair-pick"
    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        _require_arity(parents, self.arity)
        candidate_a, candidate_b = list(parents[0]), list(parents[1])
        return candidate_a if int(rng.integers(2)) == 0 else candidate_b


class _SegmentInversion(Operator[list[int]]):
    """Permutation mutation double: reverse one random contiguous segment."""

    operator_id = "test-inversion"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        _require_arity(parents, self.arity)
        child = list(parents[0])
        i, j = sorted(int(k) for k in rng.integers(0, len(child), size=2))
        child[i : j + 1] = list(reversed(child[i : j + 1]))
        return child


class _UniformBinaryCrossover(Operator[list[int]]):
    """Binary crossover double: pick each bit from one of the two parents."""

    operator_id = "test-uniform"
    representation = Representation.BINARY
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        _require_arity(parents, self.arity)
        first, second = parents[0], parents[1]
        return [int(first[i]) if rng.random() < 0.5 else int(second[i]) for i in range(len(first))]


class _RealStepMutation(Operator[list[float]]):
    """Real-valued mutation double: add Gaussian noise; never legalizes bounds."""

    operator_id = "test-gauss-step"
    representation = Representation.REAL
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def apply(self, parents: Sequence[list[float]], rng: Generator) -> list[float]:
        _require_arity(parents, self.arity)
        return [float(value) + float(rng.normal(0.0, self.sigma)) for value in parents[0]]


_DOUBLES: list[Operator[Any]] = [
    _PairPickCrossover(),
    _SegmentInversion(),
    _UniformBinaryCrossover(),
    _RealStepMutation(),
]


def _sample_parents(operator: Operator[Any]) -> list[Any]:
    """Legal sample parents matching an operator's representation and arity."""
    if operator.representation is Representation.PERMUTATION:
        parents: list[Any] = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]
    elif operator.representation is Representation.BINARY:
        parents = [[1, 0, 1, 1, 0], [0, 0, 1, 0, 1]]
    else:  # REAL
        parents = [[0.5, -0.5, 0.0, 1.0], [-1.0, 0.25, 0.75, -0.25]]
    return parents[: operator.arity]


# --- OperatorKind --------------------------------------------------------------


def test_operator_kind_covers_recombinative_and_perturbative() -> None:
    assert {member.value for member in OperatorKind} == {"recombinative", "perturbative"}


def test_operator_kind_values_are_human_readable() -> None:
    # String values keep the kind serializable for configs and AOS logs.
    assert OperatorKind.RECOMBINATIVE.value == "recombinative"
    assert OperatorKind.PERTURBATIVE.value == "perturbative"


# --- Operator: abstractness ----------------------------------------------------


def test_operator_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Operator()  # type: ignore[abstract]


def test_subclass_missing_apply_is_abstract() -> None:
    class _Incomplete(Operator[list[int]]):
        operator_id = "incomplete"
        representation = Representation.PERMUTATION
        arity = 1
        kind = OperatorKind.PERTURBATIVE

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


# --- Operator: metadata --------------------------------------------------------


@pytest.mark.parametrize("operator", _DOUBLES, ids=lambda operator: operator.operator_id)
def test_operator_exposes_valid_metadata(operator: Operator[Any]) -> None:
    assert isinstance(operator.operator_id, str)
    assert operator.operator_id
    assert isinstance(operator.representation, Representation)
    assert operator.arity in (1, 2)
    assert isinstance(operator.kind, OperatorKind)


@pytest.mark.parametrize("operator", _DOUBLES, ids=lambda operator: operator.operator_id)
def test_pool_operators_follow_the_arity_kind_convention(operator: Operator[Any]) -> None:
    # Convention (not an interface invariant): recombinative operators take two
    # parents, perturbative ones take a single parent. arity and kind stay
    # independent metadata so AOS logs can report both.
    expected = 2 if operator.kind is OperatorKind.RECOMBINATIVE else 1
    assert operator.arity == expected


# --- Operator: arity enforcement -----------------------------------------------


def test_wrong_parent_count_raises() -> None:
    recombinative = _UniformBinaryCrossover()  # arity 2
    with pytest.raises(ValueError):
        recombinative.apply([[1, 0, 1]], np.random.default_rng(0))  # only one parent

    perturbative = _SegmentInversion()  # arity 1
    with pytest.raises(ValueError):
        perturbative.apply([[0, 1, 2], [2, 1, 0]], np.random.default_rng(0))  # two parents


# --- Operator: apply returns exactly one child of the representation -----------


@pytest.mark.parametrize("operator", _DOUBLES, ids=lambda operator: operator.operator_id)
def test_apply_returns_one_child_of_the_representation(operator: Operator[Any]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert child is not parents[0]  # a fresh genome, not an aliased parent
    assert len(child) == len(parents[0])  # one child, not a pair
    if operator.representation is Representation.PERMUTATION:
        assert sorted(child) == sorted(parents[0])  # a permutation of the same elements
    elif operator.representation is Representation.BINARY:
        assert all(bit in (0, 1) for bit in child)
    else:
        assert all(isinstance(value, float) for value in child)


# --- Operator: never mutates its parents ---------------------------------------


@pytest.mark.parametrize("operator", _DOUBLES, ids=lambda operator: operator.operator_id)
def test_apply_does_not_mutate_parents(operator: Operator[Any]) -> None:
    parents = _sample_parents(operator)
    snapshot = [list(parent) for parent in parents]
    operator.apply(parents, np.random.default_rng(1))
    assert parents == snapshot


# --- Operator: deterministic and rng-only --------------------------------------


@pytest.mark.parametrize("operator", _DOUBLES, ids=lambda operator: operator.operator_id)
def test_apply_is_deterministic_for_the_same_seed(operator: Operator[Any]) -> None:
    parents = _sample_parents(operator)
    first = operator.apply(parents, np.random.default_rng(7))
    second = operator.apply(parents, np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("operator", _DOUBLES, ids=lambda operator: operator.operator_id)
def test_apply_uses_only_the_injected_generator(operator: Operator[Any]) -> None:
    # Drawing from the injected Generator must not touch NumPy's global state,
    # keeping variation reproducible from the run seed alone.
    parents = _sample_parents(operator)
    before = pickle.dumps(np.random.get_state())
    operator.apply(parents, np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


# --- Operator: a pair yields exactly one child, chosen via rng -----------------


def test_pair_crossover_returns_one_child_chosen_via_rng() -> None:
    operator = _PairPickCrossover()
    first, second = [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]

    # Across seeds the pick varies -> the choice is rng-driven, and every result
    # is a single valid child equal to exactly one candidate (never a pair).
    seen: set[tuple[int, ...]] = set()
    for seed in range(16):
        child = operator.apply([first, second], np.random.default_rng(seed))
        assert child in (first, second)
        seen.add(tuple(child))
    assert seen == {tuple(first), tuple(second)}  # both candidates occur: not a fixed pick


# --- Operator: legalization is the caller's job, not the operator's ------------


def test_real_operator_does_not_legalize_bounds() -> None:
    # A real mutation may push values outside any nominal domain; the operator
    # returns them raw. Clipping to a problem's bounds is Problem.repair's job.
    operator = _RealStepMutation(sigma=1000.0)
    child = operator.apply([[0.0, 0.0, 0.0]], np.random.default_rng(3))
    assert any(abs(value) > 1.0 for value in child)
