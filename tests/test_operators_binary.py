"""Contract spec for the binary operators of the knapsack operator pool.

Four operators make up the full knapsack pool: two crossovers -- single-point and
uniform -- and two mutations -- bit-flip and binary
swap. The classic GA baseline (CGA slice) and the *reduced* AOS pool use exactly
uniform + bit-flip; single-point and binary swap complete the *full* pool. All implement
the frozen :class:`~aos_ga.core.operator.Operator` interface over the ``list[int]`` binary
representation (bitstrings of 0/1 item-selection decisions).

The concrete classes are the executable target of this specification. Expected public
names (in ``aos_ga.operators.binary``):

- ``SinglePointCrossover`` (``operator_id="singlepoint"``, recombinative, arity 2)
- ``UniformCrossover`` (``operator_id="uniform"``, recombinative, arity 2)
- ``BitFlipMutation`` (``operator_id="bitflip"``, perturbative, arity 1)
- ``SwapBitMutation`` (``operator_id="swapbit"``, perturbative, arity 1)

Frozen contract (specialises the operator interface for binary genomes):
- ``apply(parents, rng) -> child`` returns ONE child that is a fresh ``list[int]`` of the
  same length as its parents with every element in ``{0, 1}``, drawing randomness only
  from the injected ``rng`` and being deterministic for a fixed seed.
- ``len(parents)`` must equal ``arity`` or ``ValueError`` is raised.
- The child is always a fresh list; parents are never mutated.
- Single-point crossover draws a cut point ``k in {1, ..., n-1}`` and a single random
  head bit that picks which parent donates the prefix, returning ``head[:k] + tail[k:]``
  -- ONE child from the textbook pair. Unlike uniform's per-bit mask, a single cut is NOT
  self-complementing, so the explicit head bit is what makes the returned child a uniform
  draw from the pair (as for order crossover). A valid cut needs ``n >= 2``, matching the
  permutation operators that implicitly require ``n >= 2``.
- Uniform crossover draws one per-bit Bernoulli(1/2) parent selector and sets
  ``child[i] = parents[selector[i]][i]``: every bit is inherited from one of the two
  parents at that position. It returns ONE child -- the complementary child (the mirror
  selector) is never built, so no second evaluation is spent. Because the selector's
  complement is equiprobable, that single child is already a uniform draw from the pair,
  and no extra "which parent first" bit is needed (unlike single-point/order crossover).
  At positions where the parents agree the child inherits that shared bit; identical
  parents are reproduced verbatim.
- Bit-flip mutation flips each bit independently with probability ``1/n`` for
  ``n = len(parent)`` measured inside the operator. There is NO guarantee of at least one
  flip, so ``child == parent`` is a legal outcome -- the deliberate contrast with the
  permutation inversion operator, which always changes the genome. It can flip zero, one
  or several bits, and can therefore change the number of ones in the chromosome.
- Binary swap mutation picks one position holding a 1 and one holding a 0 (uniformly and
  independently) and exchanges their values, so it ALWAYS conserves the number of ones and
  ALWAYS makes a real change -- the deliberate contrast with bit-flip, which can change the
  count and may be a no-op. This is the "1<->0" variant, which has no
  structural no-op. Edge case: an all-ones or all-zeros parent has no 1/0 pair to exchange,
  so the child is a fresh copy of the parent.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.binary import (
    BitFlipMutation,
    SinglePointCrossover,
    SwapBitMutation,
    UniformCrossover,
)

_UNIFORM = UniformCrossover()
_BITFLIP = BitFlipMutation()
_SINGLEPOINT = SinglePointCrossover()
_SWAPBIT = SwapBitMutation()
# uniform + bit-flip are the CGA slice and the reduced AOS pool; single-point and binary
# swap complete the full pool. The shared binary contract binds all four.
_BINARY_OPERATORS: list[Operator[list[int]]] = [_UNIFORM, _BITFLIP, _SINGLEPOINT, _SWAPBIT]


# --- Test oracles (verifiers, not generators) ----------------------------------


def _is_uniform_child(child: list[int], first: list[int], second: list[int]) -> bool:
    """True if every bit of ``child`` was inherited from a parent at that position.

    The definition of uniform crossover: for each index the child's bit must equal one
    of the two parents' bits there. This checks the operator's output without reproducing
    its random selector mask.
    """
    return len(child) == len(first) == len(second) and all(
        child[i] in (first[i], second[i]) for i in range(len(child))
    )


def _is_singlepoint_child(child: list[int], first: list[int], second: list[int]) -> bool:
    """True if ``child`` is a single-point offspring for some cut and head assignment.

    Scans every cut point ``k`` in ``1..n-1``: the child must equal ``head[:k] + tail[k:]``
    for one of the two head assignments. This is the definition of single-point crossover,
    used to check the operator's output without reproducing its random cut and head bit.
    """
    n = len(child)
    for k in range(1, n):
        if child == first[:k] + second[k:] or child == second[:k] + first[k:]:
            return True
    return False


def _swapped_bit_positions(parent: list[int], child: list[int]) -> tuple[int, int] | None:
    """Return ``(i, j)`` if ``child`` turns a single 1 into 0 at ``i`` and a 0 into 1 at ``j``.

    Requires exactly two differing positions holding complementary values -- so a no-op
    (``child == parent``) or a count-changing change (e.g. two 1s dropped, as bit-flip can
    do) is rejected. Used to check the "1<->0" swap without reproducing its random draws.
    """
    differing = [k for k in range(len(parent)) if parent[k] != child[k]]
    if len(differing) != 2:
        return None
    a, b = differing
    if parent[a] == 1 and child[a] == 0 and parent[b] == 0 and child[b] == 1:
        return (a, b)
    if parent[b] == 1 and child[b] == 0 and parent[a] == 0 and child[a] == 1:
        return (b, a)
    return None


# --- Oracle self-checks (the verifiers above must themselves be correct) --------


def test_uniform_oracle_accepts_a_hand_built_child() -> None:
    # bit 0 taken from `first`, bit 3 from `second`; bits 1-2 shared by both.
    first = [1, 0, 1, 0]
    second = [0, 0, 1, 1]
    assert _is_uniform_child([1, 0, 1, 1], first, second)


def test_uniform_oracle_rejects_a_bit_neither_parent_has() -> None:
    # At index 1 both parents hold 0, so a child bit of 1 there cannot be inherited.
    first = [1, 0, 1, 0]
    second = [0, 0, 1, 1]
    assert not _is_uniform_child([1, 1, 1, 1], first, second)


def test_singlepoint_oracle_accepts_a_hand_built_child() -> None:
    # head = first, cut k = 2: prefix [0, 0] from first, suffix [1, 1] from second.
    first = [0, 0, 0, 0]
    second = [1, 1, 1, 1]
    assert _is_singlepoint_child([0, 0, 1, 1], first, second)


def test_singlepoint_oracle_rejects_a_non_split_child() -> None:
    # [0, 1, 0, 1] is not any single contiguous prefix/suffix split of these parents.
    first = [0, 0, 0, 0]
    second = [1, 1, 1, 1]
    assert not _is_singlepoint_child([0, 1, 0, 1], first, second)


def test_swapbit_oracle_recovers_the_exchanged_bits() -> None:
    # parent holds 1 at index 0 and 0 at index 3; the child flips exactly those.
    assert _swapped_bit_positions([1, 0, 1, 0], [0, 0, 1, 1]) == (0, 3)


def test_swapbit_oracle_rejects_a_count_changing_change() -> None:
    # Two 1s dropped to 0 is a bit-flip-style change, not a 1<->0 swap: rejected.
    assert _swapped_bit_positions([1, 0, 1, 0], [0, 0, 0, 0]) is None


# --- Sample parents ------------------------------------------------------------


def _sample_parents(operator: Operator[list[int]]) -> list[list[int]]:
    """Two equal-length bitstrings, trimmed to the operator's arity.

    They agree at indices 0, 3, 4, 7 and differ at 1, 2, 5, 6, so tests can tell an
    inherited shared bit from a recombined one. The single (arity-1) parent is mixed --
    it holds both 0s and 1s -- so swap has a 1/0 pair to exchange.
    """
    parents = [[1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0]]
    return parents[: operator.arity]


# --- Metadata ------------------------------------------------------------------


def test_operators_are_operator_instances() -> None:
    for operator in _BINARY_OPERATORS:
        assert isinstance(operator, Operator)


def test_uniform_metadata() -> None:
    assert _UNIFORM.operator_id == "uniform"
    assert _UNIFORM.representation is Representation.BINARY
    assert _UNIFORM.arity == 2
    assert _UNIFORM.kind is OperatorKind.RECOMBINATIVE


def test_bitflip_metadata() -> None:
    assert _BITFLIP.operator_id == "bitflip"
    assert _BITFLIP.representation is Representation.BINARY
    assert _BITFLIP.arity == 1
    assert _BITFLIP.kind is OperatorKind.PERTURBATIVE


def test_singlepoint_metadata() -> None:
    assert _SINGLEPOINT.operator_id == "singlepoint"
    assert _SINGLEPOINT.representation is Representation.BINARY
    assert _SINGLEPOINT.arity == 2
    assert _SINGLEPOINT.kind is OperatorKind.RECOMBINATIVE


def test_swapbit_metadata() -> None:
    assert _SWAPBIT.operator_id == "swapbit"
    assert _SWAPBIT.representation is Representation.BINARY
    assert _SWAPBIT.arity == 1
    assert _SWAPBIT.kind is OperatorKind.PERTURBATIVE


# --- Arity enforcement ---------------------------------------------------------


def test_uniform_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _UNIFORM.apply([[1, 0, 1, 0]], np.random.default_rng(0))  # one parent, needs two


def test_bitflip_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _BITFLIP.apply([[1, 0, 1], [0, 1, 0]], np.random.default_rng(0))  # two, needs one


def test_singlepoint_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _SINGLEPOINT.apply([[1, 0, 1, 0]], np.random.default_rng(0))  # one parent, needs two


def test_swapbit_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _SWAPBIT.apply([[1, 0, 1], [0, 1, 0]], np.random.default_rng(0))  # two, needs one


# --- Shared contract: one fresh 0/1 child of equal length, parents untouched ----


@pytest.mark.parametrize("operator", _BINARY_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_returns_a_fresh_binary_child(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert all(bit in (0, 1) for bit in child)  # a 0/1 bitstring
    assert len(child) == len(parents[0])  # one child, same genome length
    for parent in parents:
        assert child is not parent  # a fresh genome, never an aliased parent


@pytest.mark.parametrize("operator", _BINARY_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_does_not_mutate_parents(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    snapshot = [list(parent) for parent in parents]
    operator.apply(parents, np.random.default_rng(1))
    assert parents == snapshot


@pytest.mark.parametrize("operator", _BINARY_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_is_deterministic_for_the_same_seed(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    first = operator.apply(parents, np.random.default_rng(7))
    second = operator.apply(parents, np.random.default_rng(7))
    assert first == second


@pytest.mark.parametrize("operator", _BINARY_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_uses_only_the_injected_generator(operator: Operator[list[int]]) -> None:
    # Drawing from the injected Generator must not touch NumPy's global state.
    parents = _sample_parents(operator)
    before = pickle.dumps(np.random.get_state())
    operator.apply(parents, np.random.default_rng(0))
    assert pickle.dumps(np.random.get_state()) == before


# --- Uniform crossover: bits inherited from both parents, one child from a pair --


def test_uniform_output_takes_each_bit_from_a_parent_across_seeds() -> None:
    first, second = _sample_parents(_UNIFORM)
    for seed in range(32):
        child = _UNIFORM.apply([first, second], np.random.default_rng(seed))
        assert _is_uniform_child(child, first, second)


def test_uniform_agrees_with_both_parents_where_they_share_a_bit() -> None:
    # At positions where the parents already agree there is nothing to recombine, so the
    # child must inherit that shared bit -- it cannot invent a value neither parent holds.
    first, second = _sample_parents(_UNIFORM)
    shared = [i for i in range(len(first)) if first[i] == second[i]]
    assert shared  # the sample parents do agree somewhere
    for seed in range(32):
        child = _UNIFORM.apply([first, second], np.random.default_rng(seed))
        for i in shared:
            assert child[i] == first[i]


def test_uniform_inherits_bits_from_both_parents() -> None:
    # With an all-ones and an all-zeros parent, a child bit reveals its origin: a 1 can
    # only come from the all-ones parent, a 0 only from the all-zeros one. Over seeds the
    # operator must draw from both -- it is a genuine mix, not a copy of one parent.
    ones, zeros = [1] * 8, [0] * 8
    took_from_ones = took_from_zeros = False
    for seed in range(32):
        child = _UNIFORM.apply([ones, zeros], np.random.default_rng(seed))
        if any(bit == 1 for bit in child):
            took_from_ones = True
        if any(bit == 0 for bit in child):
            took_from_zeros = True
    assert took_from_ones and took_from_zeros


def test_uniform_preserves_identical_parents() -> None:
    # Identical parents leave uniform crossover nothing to recombine: any selector yields
    # that same bitstring (a fresh copy).
    parent = [1, 0, 1, 1, 0, 0, 1, 0]
    for seed in range(16):
        child = _UNIFORM.apply([list(parent), list(parent)], np.random.default_rng(seed))
        assert child == parent


def test_uniform_is_rng_driven_not_fixed() -> None:
    # The per-bit selector comes from rng, so over seeds a fixed operator on fixed parents
    # produces more than one distinct child.
    first, second = _sample_parents(_UNIFORM)
    seen = {
        tuple(_UNIFORM.apply([first, second], np.random.default_rng(seed))) for seed in range(32)
    }
    assert len(seen) >= 2


# --- Bit-flip mutation: independent 1/n flips, no forced change -----------------


def test_bitflip_can_change_the_number_of_ones() -> None:
    # The knapsack acceptance criterion: bit-flip is able to add or drop selected items,
    # so the count of ones is not conserved (contrast with binary swap).
    parent = [1, 0, 1, 1, 0, 1, 0, 0]
    base_ones = sum(parent)
    counts = {
        sum(_BITFLIP.apply([list(parent)], np.random.default_rng(seed))) for seed in range(32)
    }
    assert any(count != base_ones for count in counts)


def test_bitflip_can_leave_the_parent_unchanged() -> None:
    # Pure 1/n per bit with no forced flip -> child == parent is a legal outcome. This is
    # the deliberate contrast with inversion, which always changes the genome.
    parent = [1, 0, 1, 1, 0, 1, 0, 0]
    assert any(
        _BITFLIP.apply([list(parent)], np.random.default_rng(seed)) == parent for seed in range(50)
    )


def test_bitflip_can_flip_more_than_one_bit() -> None:
    # Bits flip independently, so an application can flip several at once -- it is not a
    # fixed single-bit move. (All-zeros parent: the ones in the child are exactly the flips.)
    parent = [0] * 8
    assert any(
        sum(_BITFLIP.apply([list(parent)], np.random.default_rng(seed))) >= 2 for seed in range(50)
    )


def test_bitflip_flip_rate_is_about_one_over_n() -> None:
    # Expected flips per application = n * (1/n) = 1. Averaging over many seeds pins the
    # 1/n per-bit rate and rules out a much higher rate (e.g. 1/2 -> ~4 flips). Loose
    # bounds keep the check robust to sampling noise. All-zeros parent: ones == flips.
    parent = [0] * 8
    n_seeds = 300
    total_flips = sum(
        sum(_BITFLIP.apply([list(parent)], np.random.default_rng(seed))) for seed in range(n_seeds)
    )
    mean_flips = total_flips / n_seeds
    assert 0.5 <= mean_flips <= 1.5


def test_bitflip_is_rng_driven_not_fixed() -> None:
    parent = [1, 0, 1, 1, 0, 1, 0, 0]
    seen = {
        tuple(_BITFLIP.apply([list(parent)], np.random.default_rng(seed))) for seed in range(32)
    }
    assert len(seen) >= 2


# --- Single-point crossover: contiguous prefix/suffix split, one child from a pair --


def test_singlepoint_output_is_a_valid_singlepoint_child_across_seeds() -> None:
    first, second = _sample_parents(_SINGLEPOINT)
    for seed in range(32):
        child = _SINGLEPOINT.apply([first, second], np.random.default_rng(seed))
        # A prefix of one parent up to some cut k, a suffix of the other from k on.
        assert _is_singlepoint_child(child, first, second)


def test_singlepoint_head_bit_reaches_both_parents() -> None:
    # With an all-ones and an all-zeros parent the head parent is visible in the prefix:
    # head == ones -> child starts with 1, head == zeros -> child starts with 0. Over
    # seeds both must appear, proving the head bit is drawn (not a fixed donor) -- the
    # single-cut analogue of order crossover's donor bit.
    ones, zeros = [1] * 8, [0] * 8
    heads = {
        _SINGLEPOINT.apply([ones, zeros], np.random.default_rng(seed))[0] for seed in range(32)
    }
    assert heads == {0, 1}


def test_singlepoint_choice_is_rng_driven_not_fixed() -> None:
    # Both the head bit and the cut point come from rng, so a fixed operator on fixed
    # parents produces more than one distinct child over seeds.
    first, second = _sample_parents(_SINGLEPOINT)
    seen = {
        tuple(_SINGLEPOINT.apply([first, second], np.random.default_rng(seed)))
        for seed in range(32)
    }
    assert len(seen) >= 2


def test_singlepoint_preserves_identical_parents() -> None:
    # Identical parents leave single-point nothing to recombine: any cut and either head
    # yield that same bitstring (a fresh copy).
    parent = [1, 0, 1, 1, 0, 0, 1, 0]
    for seed in range(16):
        child = _SINGLEPOINT.apply([list(parent), list(parent)], np.random.default_rng(seed))
        assert child == parent


# --- Binary swap: one 1<->0 exchange, always conserves the number of ones -------


def test_swapbit_conserves_the_number_of_ones_across_seeds() -> None:
    # The knapsack acceptance criterion and the point of the "1<->0" variant: swap
    # trades a selected item for an unselected one, so the count of ones is invariant --
    # the deliberate contrast with bit-flip, which can change it.
    parent = [1, 0, 1, 1, 0, 1, 0, 0]
    base_ones = sum(parent)
    for seed in range(32):
        child = _SWAPBIT.apply([list(parent)], np.random.default_rng(seed))
        assert sum(child) == base_ones


def test_swapbit_exchanges_one_one_and_one_zero_across_seeds() -> None:
    # Exactly two positions change: a 1 turns into a 0 and a 0 into a 1, nothing else.
    parent = [1, 0, 1, 1, 0, 1, 0, 0]
    for seed in range(32):
        child = _SWAPBIT.apply([list(parent)], np.random.default_rng(seed))
        assert _swapped_bit_positions(parent, child) is not None


def test_swapbit_always_changes_a_mixed_parent() -> None:
    # A parent with at least one 1 and one 0 always has a pair to exchange, so the swap is
    # a real change -- no evaluation is spent on a no-op (contrast with bit-flip).
    parent = [1, 0, 1, 1, 0, 1, 0, 0]
    for seed in range(32):
        child = _SWAPBIT.apply([list(parent)], np.random.default_rng(seed))
        assert child != parent


def test_swapbit_returns_a_copy_for_an_all_ones_parent() -> None:
    # No zero to receive a one: there is no 1<->0 pair, so the child is a FRESH copy of the
    # parent, conserving the count trivially.
    parent = [1] * 8
    child = _SWAPBIT.apply([parent], np.random.default_rng(0))
    assert child == parent
    assert child is not parent  # a fresh list, never the aliased parent


def test_swapbit_returns_a_copy_for_an_all_zeros_parent() -> None:
    # Mirror edge: no one to give away, so the child is a fresh copy of the all-zeros parent.
    parent = [0] * 8
    child = _SWAPBIT.apply([parent], np.random.default_rng(0))
    assert child == parent
    assert child is not parent  # a fresh list, never the aliased parent


def test_swapbit_is_rng_driven_not_fixed() -> None:
    # Four ones and four zeros give many 1<->0 pairs, so over seeds the operator produces
    # more than one distinct child.
    parent = [1, 1, 1, 1, 0, 0, 0, 0]
    seen = {
        tuple(_SWAPBIT.apply([list(parent)], np.random.default_rng(seed))) for seed in range(32)
    }
    assert len(seen) >= 2
