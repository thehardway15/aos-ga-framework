"""Contract spec for the permutation operators of the TSP operator pool.

Six operators make up the full TSP pool: three
crossovers -- Order Crossover (OX1, Davis), Partially Mapped Crossover (PMX,
Goldberg-Lingle) and Cycle Crossover (CX, Oliver) -- and three mutations -- swap,
Simple Inversion Mutation (SIM) and insert. The classic GA baseline (CGA slice) uses
OX + inversion; the *reduced* AOS pool is OX, CX, inversion; PMX, swap and insert
complete the *full* pool. All implement the frozen
:class:`~aos_ga.core.operator.Operator` interface over the ``list[int]`` permutation
representation.

The concrete classes are the executable target of this specification. Expected public
names (in ``aos_ga.operators.permutation``):

- ``OrderCrossover`` (``operator_id="ox"``, recombinative, arity 2)
- ``PartiallyMappedCrossover`` (``operator_id="pmx"``, recombinative, arity 2)
- ``CycleCrossover`` (``operator_id="cx"``, recombinative, arity 2)
- ``SegmentInversion`` (``operator_id="inversion"``, perturbative, arity 1)
- ``SwapMutation`` (``operator_id="swap"``, perturbative, arity 1)
- ``InsertMutation`` (``operator_id="insert"``, perturbative, arity 1)

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
- PMX copies a contiguous segment ``[i, j]`` (``i <= j``) verbatim from a donor
  parent -- picked by a single random bit, like OX -- and fills each outside position
  with the other parent's value there, resolving duplicates by chasing the segment
  mapping ``m(donor) = other`` until a value outside the segment is reached (the
  value-chain formulation). One child from the textbook pair via the donor bit.
- CX inherits every position from one of the parents (``child[k] in {p1[k], p2[k]}``
  for all ``k``): the position cycles between the parents are assigned by strict
  parity -- odd cycles from one parent, even cycles from the other. Like OX it is a
  pair-producing crossover reduced to ONE child by a single random bit that picks
  which parent owns the odd cycles; that bit is CX's only randomness (the cycle
  structure is fixed by the parents), so at most two distinct children can result.
- SIM reverses one contiguous segment ``[i, j]`` with ``i < j`` (segment length
  >= 2), so every application is a real change: ``child != parent`` for ``n >= 2``.
- swap draws two DISTINCT positions ``i != j`` and exchanges their values, leaving the
  rest in place -- a real change every time (``n >= 2``).
- insert removes the element at position ``i`` and reinserts it at position ``j != i``
  (``pop(i); insert(j)``), relocating a single element. The position pair is NOT
  sorted, so both branches of the definition are reachable (forward ``i < j`` and
  backward ``j < i``); a real change every time (``n >= 2``).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.permutation import (
    CycleCrossover,
    InsertMutation,
    OrderCrossover,
    PartiallyMappedCrossover,
    SegmentInversion,
    SwapMutation,
)

_OX = OrderCrossover()
_INVERSION = SegmentInversion()
_CX = CycleCrossover()
_PMX = PartiallyMappedCrossover()
_SWAP = SwapMutation()
_INSERT = InsertMutation()
# OX + inversion are the CGA slice; CX joins them in the reduced AOS pool; PMX, swap and
# insert complete the full pool. The shared permutation contract binds all six.
_PERMUTATION_OPERATORS: list[Operator[list[int]]] = [
    _OX,
    _INVERSION,
    _CX,
    _PMX,
    _SWAP,
    _INSERT,
]


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


def _cx_cycles(p1: list[int], p2: list[int]) -> list[list[int]]:
    """Partition positions into Cycle-Crossover cycles, ordered by first position.

    The cycle through a position follows ``phi(k) = index in p1 of p2[k]`` until it
    returns to the start. Iterating starts from position 0 upward makes cycle 0 the
    one containing position 0, giving a canonical (parity-defining) cycle order.
    Works for any distinct labels, not only ``0..n-1``.
    """
    index_in_p1 = {value: k for k, value in enumerate(p1)}
    visited = [False] * len(p1)
    cycles: list[list[int]] = []
    for start in range(len(p1)):
        if visited[start]:
            continue
        cycle: list[int] = []
        pos = start
        while not visited[pos]:
            visited[pos] = True
            cycle.append(pos)
            pos = index_in_p1[p2[pos]]
        cycles.append(cycle)
    return cycles


def _cx_children(p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
    """The two valid CX offspring: alternating cycle assignment and its complement.

    A cycle's 1-indexed number is odd exactly when the 0-indexed ``k`` here is even.
    The first child takes odd cycles from ``p1`` and even cycles from ``p2``; the
    second is its mirror image (the other value of CX's random bit).
    """
    child_a = list(p1)
    child_b = list(p1)
    for k, cycle in enumerate(_cx_cycles(p1, p2)):
        odd = k % 2 == 0  # 1-indexed cycle number is odd
        for pos in cycle:
            child_a[pos] = p1[pos] if odd else p2[pos]
            child_b[pos] = p2[pos] if odd else p1[pos]
    return child_a, child_b


def _is_cx_child(child: list[int], p1: list[int], p2: list[int]) -> bool:
    """True if ``child`` is one of the two strictly alternating CX offspring.

    Checks the output against both reconstructed children without reproducing the
    operator's random bit. A positionally inherited permutation that assigns cycles
    non-alternately (e.g. every cycle from the same parent when there is more than
    one) is rejected -- strict odd/even alternation is part of the CX contract.
    """
    child_a, child_b = _cx_children(p1, p2)
    return child in (child_a, child_b)


# --- PMX / swap / insert oracles (verifiers for the full-pool operators) --------


def _pmx_child_from(donor: list[int], order_source: list[int], i: int, j: int) -> list[int]:
    """Reconstruct the PMX child for donor segment ``[i, j]`` (02, value-chain form).

    ``donor[i..j]`` is copied verbatim; ``m`` maps each donor segment value to the
    ``order_source`` value at the same position. Every outside position takes
    ``order_source[k]``, but while that value already sits in the copied segment it is
    replaced by ``m`` of itself and the chase repeats until a value outside the segment
    is reached. Used to check the operator without reproducing its random cut points.
    Terminates for valid permutations: the chase visits distinct segment positions.
    """
    n = len(donor)
    segment = set(donor[i : j + 1])
    mapping = {donor[k]: order_source[k] for k in range(i, j + 1)}
    child = list(donor)  # the segment is already in place; fill the rest
    for k in range(n):
        if i <= k <= j:
            continue
        value = order_source[k]
        while value in segment:
            value = mapping[value]
        child[k] = value
    return child


def _is_pmx_child(child: list[int], first: list[int], second: list[int]) -> bool:
    """True if ``child`` is a valid PMX offspring for some segment and either donor."""
    n = len(child)
    for i in range(n):
        for j in range(i, n):
            if _pmx_child_from(first, second, i, j) == child:
                return True
            if _pmx_child_from(second, first, i, j) == child:
                return True
    return False


def _swapped_positions(parent: list[int], child: list[int]) -> tuple[int, int] | None:
    """Return ``(i, j)`` with ``i < j`` if ``child`` is ``parent`` with those exchanged.

    Returns ``None`` unless exactly two positions differ and they hold each other's
    parent value -- so a no-op (``child == parent``) or any change touching a number of
    positions other than two is rejected.
    """
    differing = [k for k in range(len(parent)) if parent[k] != child[k]]
    if len(differing) != 2:
        return None
    i, j = differing
    if child[i] == parent[j] and child[j] == parent[i]:
        return (i, j)
    return None


def _relocated_positions(parent: list[int], child: list[int]) -> tuple[int, int] | None:
    """Return ``(i, j)``, ``i != j``, if ``child`` moves ``parent[i]`` to position ``j``.

    Enumerates every ordered pair of distinct positions and rebuilds ``pop(i);
    insert(j)``, returning the first pair that reproduces ``child`` (adjacent
    relocations are direction-ambiguous, non-adjacent ones are unique) or ``None`` when
    ``child`` is not a single-element relocation of ``parent``.
    """
    n = len(parent)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            candidate = list(parent)
            candidate.insert(j, candidate.pop(i))
            if candidate == child:
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


def test_cx_oracle_accepts_both_hand_computed_children() -> None:
    # p1 identity, p2 forms position cycles {0,1,2} and {3,4,5}. The odd cycle from
    # p1 gives one child; its mirror gives the other.
    p1 = [0, 1, 2, 3, 4, 5]
    p2 = [1, 2, 0, 5, 3, 4]
    assert _is_cx_child([0, 1, 2, 5, 3, 4], p1, p2)  # odd cycle {0,1,2} from p1
    assert _is_cx_child([1, 2, 0, 3, 4, 5], p1, p2)  # the mirror child


def test_cx_oracle_rejects_a_non_alternating_assignment() -> None:
    # p1 itself takes BOTH cycles from p1: positionally inherited, but not the
    # strict odd/even alternation CX requires, so it is not a valid CX child.
    p1 = [0, 1, 2, 3, 4, 5]
    p2 = [1, 2, 0, 5, 3, 4]
    assert not _is_cx_child([0, 1, 2, 3, 4, 5], p1, p2)


def test_pmx_oracle_accepts_a_hand_computed_child() -> None:
    # Classic Goldberg-Lingle example: donor segment (0-based) [3, 5] copied verbatim,
    # outside positions resolved through the mapping chain (4->1, 5->8, 6->7).
    donor = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    order_source = [4, 5, 2, 1, 8, 7, 6, 9, 3]
    assert _is_pmx_child([1, 8, 2, 4, 5, 6, 7, 9, 3], donor, order_source)


def test_pmx_oracle_rejects_an_unreachable_permutation() -> None:
    # For this pair PMX yields five of the six permutations of {0, 1, 2}; the one it
    # never produces is [1, 2, 0], so the oracle must reject it.
    assert not _is_pmx_child([1, 2, 0], [0, 1, 2], [2, 0, 1])


def test_swap_oracle_recovers_the_exchanged_positions() -> None:
    assert _swapped_positions([0, 1, 2, 3], [0, 3, 2, 1]) == (1, 3)


def test_swap_oracle_rejects_a_non_swap_change() -> None:
    # A three-position rotation is not a single transposition.
    assert _swapped_positions([0, 1, 2, 3], [1, 2, 0, 3]) is None


def test_relocation_oracle_recovers_a_forward_and_a_backward_move() -> None:
    # pop(1); insert(3): the element at 1 moves right past positions 2..3.
    assert _relocated_positions([0, 1, 2, 3, 4], [0, 2, 3, 1, 4]) == (1, 3)
    # pop(3); insert(1): the element at 3 moves left in front of positions 1..2.
    assert _relocated_positions([0, 1, 2, 3, 4], [0, 3, 1, 2, 4]) == (3, 1)


def test_relocation_oracle_rejects_a_non_relocation() -> None:
    # A full reversal moves every element, not a single one.
    assert _relocated_positions([0, 1, 2, 3], [3, 2, 1, 0]) is None


# --- Sample parents ------------------------------------------------------------


def _sample_parents(operator: Operator[list[int]]) -> list[list[int]]:
    """Two permutations of the same element set, trimmed to the operator's arity."""
    parents = [[0, 1, 2, 3, 4, 5, 6, 7], [3, 7, 0, 5, 1, 6, 2, 4]]
    return parents[: operator.arity]


# --- Metadata ------------------------------------------------------------------


def test_operators_are_operator_instances() -> None:
    for operator in _PERMUTATION_OPERATORS:
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


def test_cx_metadata() -> None:
    assert _CX.operator_id == "cx"
    assert _CX.representation is Representation.PERMUTATION
    assert _CX.arity == 2
    assert _CX.kind is OperatorKind.RECOMBINATIVE


def test_pmx_metadata() -> None:
    assert _PMX.operator_id == "pmx"
    assert _PMX.representation is Representation.PERMUTATION
    assert _PMX.arity == 2
    assert _PMX.kind is OperatorKind.RECOMBINATIVE


def test_swap_metadata() -> None:
    assert _SWAP.operator_id == "swap"
    assert _SWAP.representation is Representation.PERMUTATION
    assert _SWAP.arity == 1
    assert _SWAP.kind is OperatorKind.PERTURBATIVE


def test_insert_metadata() -> None:
    assert _INSERT.operator_id == "insert"
    assert _INSERT.representation is Representation.PERMUTATION
    assert _INSERT.arity == 1
    assert _INSERT.kind is OperatorKind.PERTURBATIVE


# --- Arity enforcement ---------------------------------------------------------


def test_ox_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _OX.apply([[0, 1, 2, 3]], np.random.default_rng(0))  # one parent, needs two


def test_inversion_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _INVERSION.apply([[0, 1, 2], [2, 1, 0]], np.random.default_rng(0))  # two, needs one


def test_cx_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _CX.apply([[0, 1, 2, 3]], np.random.default_rng(0))  # one parent, needs two


def test_pmx_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _PMX.apply([[0, 1, 2, 3]], np.random.default_rng(0))  # one parent, needs two


def test_swap_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _SWAP.apply([[0, 1, 2], [2, 1, 0]], np.random.default_rng(0))  # two, needs one


def test_insert_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _INSERT.apply([[0, 1, 2], [2, 1, 0]], np.random.default_rng(0))  # two, needs one


# --- Shared contract: one fresh valid-permutation child, parents untouched ------


@pytest.mark.parametrize("operator", _PERMUTATION_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_returns_a_fresh_permutation_child(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert sorted(child) == sorted(parents[0])  # a permutation of the same elements
    for parent in parents:
        assert child is not parent  # a fresh genome, never an aliased parent


@pytest.mark.parametrize("operator", _PERMUTATION_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_does_not_mutate_parents(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    snapshot = [list(parent) for parent in parents]
    operator.apply(parents, np.random.default_rng(1))
    assert parents == snapshot


@pytest.mark.parametrize("operator", _PERMUTATION_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_is_deterministic_for_the_same_seed(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    first = operator.apply(parents, np.random.default_rng(7))
    second = operator.apply(parents, np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("operator", _PERMUTATION_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_uses_only_the_injected_generator(operator: Operator[list[int]]) -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    parents = _sample_parents(operator)
    before = pickle.dumps(np.random.get_state())
    operator.apply(parents, np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("operator", _PERMUTATION_OPERATORS, ids=lambda op: op.operator_id)
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


# --- CX: positional inheritance, one of two children, cycle edge cases ---------


def test_cx_output_is_a_valid_cx_child_across_seeds() -> None:
    first, second = _sample_parents(_CX)
    for seed in range(32):
        child = _CX.apply([first, second], np.random.default_rng(seed))
        # A strictly alternating cycle assignment (one of exactly two).
        assert _is_cx_child(child, first, second)


def test_cx_inherits_each_position_from_a_parent() -> None:
    # The defining CX invariant, checked directly (independent of the cycle oracle):
    # every position comes from p1 or p2, never a foreign value.
    first, second = _sample_parents(_CX)
    for seed in range(32):
        child = _CX.apply([first, second], np.random.default_rng(seed))
        assert all(child[k] in (first[k], second[k]) for k in range(len(child)))


def test_cx_reaches_both_children_of_the_pair() -> None:
    # The only randomness is one bit (which parent owns the odd cycles), so over
    # seeds exactly the two reconstructed children appear -- no more, no fewer.
    first, second = _sample_parents(_CX)
    child_a, child_b = _cx_children(first, second)
    seen = {tuple(_CX.apply([first, second], np.random.default_rng(seed))) for seed in range(32)}
    assert seen == {tuple(child_a), tuple(child_b)}


def test_cx_assigns_alternating_cycles_not_just_the_first() -> None:
    # Three position cycles {0,1}, {2,3}, {4,5}. The CX contract assigns cycles by
    # STRICT parity (odd cycles from one parent, even from the other),
    # so the third (odd) cycle returns to the first parent -- it is NOT "first cycle
    # from one parent, everything else from the other". Only a case with >= 3 cycles
    # (which random tours hit routinely) separates the two formulations, so it is
    # pinned explicitly here.
    first, second = [0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]
    child_a, child_b = _cx_children(first, second)  # [0,1,3,2,4,5] and [1,0,2,3,5,4]
    seen = {tuple(_CX.apply([first, second], np.random.default_rng(seed))) for seed in range(32)}
    assert seen == {tuple(child_a), tuple(child_b)}


def test_cx_preserves_identical_parents() -> None:
    # p1 == p2 makes every position its own singleton cycle, so both children equal
    # the parent: any bit reproduces it.
    parent = [0, 1, 2, 3, 4, 5]
    for seed in range(16):
        child = _CX.apply([list(parent), list(parent)], np.random.default_rng(seed))
        assert child == parent


def test_cx_with_a_single_cycle_copies_one_whole_parent() -> None:
    # One big cycle leaves nothing to alternate: the child is a whole parent, p1 or
    # p2 depending on the bit, and both appear over seeds.
    first, second = [0, 1, 2, 3], [1, 2, 3, 0]
    seen = {tuple(_CX.apply([first, second], np.random.default_rng(seed))) for seed in range(32)}
    assert seen == {tuple(first), tuple(second)}


# --- PMX: valid partially-mapped child, one child from the pair ----------------


def test_pmx_output_is_a_valid_pmx_child_across_seeds() -> None:
    first, second = _sample_parents(_PMX)
    for seed in range(32):
        child = _PMX.apply([first, second], np.random.default_rng(seed))
        # Copies a segment from one parent, maps the rest through the other.
        assert _is_pmx_child(child, first, second)


def test_pmx_returns_a_single_child_never_a_pair() -> None:
    parents = _sample_parents(_PMX)
    child = _PMX.apply(parents, np.random.default_rng(0))
    assert len(child) == len(parents[0])  # one genome, not two concatenated


def test_pmx_choice_is_rng_driven_not_fixed() -> None:
    # Both the donor bit and the cut points come from rng, so a fixed operator
    # produces more than one distinct child over seeds.
    parents = _sample_parents(_PMX)
    seen = {tuple(_PMX.apply(parents, np.random.default_rng(seed))) for seed in range(32)}
    assert len(seen) >= 2


def test_pmx_preserves_identical_parents() -> None:
    # Identical parents make the mapping an identity: every outside value already lies
    # outside the segment, so the child is that permutation (a fresh copy).
    parent = [0, 1, 2, 3, 4, 5]
    child = _PMX.apply([list(parent), list(parent)], np.random.default_rng(3))
    assert child == parent


# --- swap: exactly two positions exchanged, always a real change ---------------


def test_swap_exchanges_exactly_two_positions_across_seeds() -> None:
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _SWAP.apply([list(parent)], np.random.default_rng(seed))
        positions = _swapped_positions(parent, child)
        assert positions is not None  # exactly two positions, exchanged, rest in place
        i, j = positions
        assert i != j


def test_swap_always_changes_the_parent() -> None:
    # Two distinct positions with distinct labels always exchange into a new tour --
    # no evaluation is spent on a no-op.
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _SWAP.apply([list(parent)], np.random.default_rng(seed))
        assert child != parent


def test_swap_choice_is_rng_driven_not_fixed() -> None:
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    seen = {tuple(_SWAP.apply([parent], np.random.default_rng(seed))) for seed in range(32)}
    assert len(seen) >= 2


# --- insert: single-element relocation, both directions reachable --------------


def test_insert_relocates_exactly_one_element_across_seeds() -> None:
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _INSERT.apply([list(parent)], np.random.default_rng(seed))
        assert _relocated_positions(parent, child) is not None  # a single relocation


def test_insert_always_changes_the_parent() -> None:
    # pop(i); insert(j) with i != j never reproduces the parent (only j == i would).
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    for seed in range(32):
        child = _INSERT.apply([list(parent)], np.random.default_rng(seed))
        assert child != parent


def test_insert_reaches_both_relocation_directions() -> None:
    # The position pair is NOT sorted, so both branches of the definition appear over
    # seeds: forward moves (i < j) and backward moves (j < i). A sorting implementation
    # would only ever produce one direction. Non-adjacent draws give unambiguous
    # directions, so both booleans are reached.
    parent = [0, 1, 2, 3, 4, 5, 6, 7]
    directions = set()
    for seed in range(64):
        child = _INSERT.apply([list(parent)], np.random.default_rng(seed))
        positions = _relocated_positions(parent, child)
        assert positions is not None
        i, j = positions
        directions.add(i < j)
    assert directions == {True, False}  # both forward and backward relocations occur
