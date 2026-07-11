"""Contract spec for the reduced-pool permutation operators used by the CGA slice.

Two operators are pinned here -- Order Crossover (OX1, Davis) and Simple Inversion
Mutation (SIM) -- because the classic GA baseline on TSP uses exactly this pair:
OX for recombination and inversion for perturbation. They implement the frozen
:class:`~aos_ga.core.operator.Operator` interface over the ``list[int]``
permutation representation.

The concrete classes are not implemented yet: this file is the executable
specification of their behaviour. Expected public names (in
``aos_ga.operators.permutation``): ``OrderCrossover`` (``operator_id="ox"``,
recombinative, arity 2) and ``SegmentInversion`` (``operator_id="inversion"``,
perturbative, arity 1).

Frozen contract (specialises the operator interface for permutations):
- ``apply(parents, rng) -> child`` returns ONE child that is a permutation of the
  same element multiset as its parents (``sorted(child) == sorted(parent)``),
  drawing randomness only from the injected ``rng`` and being deterministic for a
  fixed seed. Genomes are opaque distinct labels -- no assumption of ``0..n-1``.
- ``len(parents)`` must equal ``arity`` or ``ValueError`` is raised.
- The child is always a fresh list; parents are never mutated.
- OX (a textbook pair-producing crossover) returns ONE child: a single random bit
  drawn from ``rng`` picks which parent donates the copied segment, and only that
  child is built, so no second evaluation is spent.
- OX copies a contiguous segment verbatim from the donor and fills the rest from
  the other parent in that parent's relative order (scanning and writing from
  ``(j+1) mod n`` with wraparound), preserving relative order.
- SIM reverses one contiguous segment ``[i, j]`` with ``i < j`` (segment length
  >= 2), so every application is a real change: ``child != parent`` for ``n >= 2``.

The rest of the permutation pool (CX for the reduced pool, PMX/swap/insert for the
full pool) is out of the slice and specified when it is assembled.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.permutation import OrderCrossover, SegmentInversion

_OX = OrderCrossover()
_INVERSION = SegmentInversion()
_SLICE_OPERATORS: list[Operator[list[int]]] = [_OX, _INVERSION]


# --- Test oracles (verifiers, not generators) ----------------------------------


def _is_ox1_child_from(child: list[int], donor: list[int], order_source: list[int]) -> bool:
    """True if ``child`` is a valid OX1 offspring for some segment ``[i, j]``.

    Scans every candidate segment: the child must copy ``donor[i..j]`` verbatim
    and fill the remaining positions (cyclically from ``j+1``) with the elements
    of ``order_source`` in its own order (cyclically from ``j+1``), skipping the
    ones already taken from the segment. This is the definition of OX1, used to
    check the operator's output without reproducing its random cut points.
    """
    n = len(child)
    for i in range(n):
        for j in range(i, n):
            if list(child[i : j + 1]) != list(donor[i : j + 1]):
                continue
            segment = set(donor[i : j + 1])
            fill_positions = [(j + 1 + t) % n for t in range(n - (j - i + 1))]
            source_cycle = [order_source[(j + 1 + t) % n] for t in range(n)]
            expected = [value for value in source_cycle if value not in segment]
            if all(child[fill_positions[k]] == expected[k] for k in range(len(fill_positions))):
                return True
    return False


def _is_ox1_child(child: list[int], first: list[int], second: list[int]) -> bool:
    """True if ``child`` is a valid OX1 offspring with either parent as donor."""
    return _is_ox1_child_from(child, first, second) or _is_ox1_child_from(child, second, first)


def _inverted_segment(parent: list[int], child: list[int]) -> tuple[int, int] | None:
    """Return ``(i, j)`` with ``i < j`` if ``child`` reverses exactly ``parent[i..j]``.

    Returns ``None`` when the change is not a single contiguous reversal (including
    the degenerate no-op ``child == parent``), which the ``i < j`` rule forbids.
    """
    differing = [k for k in range(len(parent)) if parent[k] != child[k]]
    if not differing:
        return None
    i, j = differing[0], differing[-1]
    middle_reversed = child[i : j + 1] == list(reversed(parent[i : j + 1]))
    outside_intact = child[:i] == parent[:i] and child[j + 1 :] == parent[j + 1 :]
    if middle_reversed and outside_intact:
        return (i, j)
    return None


# --- Oracle self-checks (the verifiers above must themselves be correct) --------


def test_ox_oracle_accepts_a_hand_computed_child() -> None:
    # p1's segment [2..4] = [2,3,4] fixed; the rest filled from p2's order.
    donor = [0, 1, 2, 3, 4, 5, 6, 7]
    order_source = [7, 6, 5, 4, 3, 2, 1, 0]
    assert _is_ox1_child([6, 5, 2, 3, 4, 1, 0, 7], donor, order_source)


def test_ox_oracle_rejects_a_non_ox_permutation() -> None:
    # No contiguous block of this child matches either parent at its positions,
    # so it cannot be an OX1 offspring of the pair.
    assert not _is_ox1_child([3, 2, 1, 0], [0, 1, 2, 3], [1, 0, 3, 2])


def test_inversion_oracle_recovers_the_reversed_segment() -> None:
    assert _inverted_segment([0, 1, 2, 3, 4, 5], [0, 4, 3, 2, 1, 5]) == (1, 4)


def test_inversion_oracle_rejects_a_non_contiguous_change() -> None:
    assert _inverted_segment([0, 1, 2, 3, 4], [2, 1, 0, 4, 3]) is None


# --- Sample parents ------------------------------------------------------------


def _sample_parents(operator: Operator[list[int]]) -> list[list[int]]:
    """Two permutations of the same element set, trimmed to the operator's arity."""
    parents = [[0, 1, 2, 3, 4, 5, 6, 7], [3, 7, 0, 5, 1, 6, 2, 4]]
    return parents[: operator.arity]


# --- Metadata ------------------------------------------------------------------


def test_operators_are_operator_instances() -> None:
    for operator in _SLICE_OPERATORS:
        assert isinstance(operator, Operator)


def test_ox_metadata() -> None:
    assert _OX.operator_id == "ox"
    assert _OX.representation is Representation.PERMUTATION
    assert _OX.arity == 2
    assert _OX.kind is OperatorKind.RECOMBINATIVE


def test_inversion_metadata() -> None:
    assert _INVERSION.operator_id == "inversion"
    assert _INVERSION.representation is Representation.PERMUTATION
    assert _INVERSION.arity == 1
    assert _INVERSION.kind is OperatorKind.PERTURBATIVE


# --- Arity enforcement ---------------------------------------------------------


def test_ox_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _OX.apply([[0, 1, 2, 3]], np.random.default_rng(0))  # one parent, needs two


def test_inversion_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _INVERSION.apply([[0, 1, 2], [2, 1, 0]], np.random.default_rng(0))  # two, needs one


# --- Shared contract: one fresh valid-permutation child, parents untouched ------


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_returns_a_fresh_permutation_child(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert sorted(child) == sorted(parents[0])  # a permutation of the same elements
    for parent in parents:
        assert child is not parent  # a fresh genome, never an aliased parent


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_does_not_mutate_parents(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    snapshot = [list(parent) for parent in parents]
    operator.apply(parents, np.random.default_rng(1))
    assert parents == snapshot


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_is_deterministic_for_the_same_seed(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    first = operator.apply(parents, np.random.default_rng(7))
    second = operator.apply(parents, np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_uses_only_the_injected_generator(operator: Operator[list[int]]) -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    parents = _sample_parents(operator)
    before = pickle.dumps(np.random.get_state())
    operator.apply(parents, np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_stays_a_permutation_across_seeds(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    for seed in range(32):
        child = operator.apply(parents, np.random.default_rng(seed))
        assert sorted(child) == sorted(parents[0])


# --- OX: relative-order preservation and one-child-from-a-pair ------------------


def test_ox_output_is_a_valid_ox1_child_across_seeds() -> None:
    parents = _sample_parents(_OX)
    first, second = parents[0], parents[1]
    for seed in range(32):
        child = _OX.apply([first, second], np.random.default_rng(seed))
        # Copies a segment from one parent, fills the rest in the other's order.
        assert _is_ox1_child(child, first, second)


def test_ox_returns_a_single_child_never_a_pair() -> None:
    parents = _sample_parents(_OX)
    child = _OX.apply(parents, np.random.default_rng(0))
    assert len(child) == len(parents[0])  # one genome, not two concatenated


def test_ox_choice_is_rng_driven_not_fixed() -> None:
    # Over seeds the result varies: both the segment-donor bit and the cut points
    # come from rng, so a fixed operator produces more than one distinct child.
    parents = _sample_parents(_OX)
    seen = {tuple(_OX.apply(parents, np.random.default_rng(seed))) for seed in range(32)}
    assert len(seen) >= 2


def test_ox_can_preserve_an_identical_parent() -> None:
    # Identical parents leave OX nothing to recombine: the child is that permutation
    # (a fresh copy), still a valid permutation.
    parent = [0, 1, 2, 3, 4, 5]
    child = _OX.apply([list(parent), list(parent)], np.random.default_rng(3))
    assert child == parent


# --- Inversion: exactly one reversed segment with i < j ------------------------


def test_inversion_reverses_exactly_one_contiguous_segment_across_seeds() -> None:
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _INVERSION.apply([list(parent)], np.random.default_rng(seed))
        segment = _inverted_segment(parent, child)
        assert segment is not None  # a single contiguous reversal, never a no-op
        i, j = segment
        assert i < j  # segment length >= 2 (SIM: two distinct cut points)


def test_inversion_always_changes_the_parent() -> None:
    # With i < j and distinct labels, reversing swaps the endpoints, so the child
    # always differs from the parent -- no evaluation is spent on a no-op.
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _INVERSION.apply([list(parent)], np.random.default_rng(seed))
        assert child != parent


def test_inversion_leaves_positions_outside_the_segment_in_place() -> None:
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _INVERSION.apply([list(parent)], np.random.default_rng(seed))
        i, j = _inverted_segment(parent, child)  # type: ignore[misc]
        assert child[:i] == parent[:i]
        assert child[j + 1 :] == parent[j + 1 :]
