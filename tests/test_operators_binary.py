"""Contract spec for the reduced-pool binary operators used by the knapsack CGA slice.

Two operators are pinned here -- Uniform Crossover and Bit-Flip Mutation -- because the
classic GA baseline on the 0/1 knapsack uses exactly this pair: uniform crossover for
recombination and bit-flip for perturbation. They implement the frozen
:class:`~aos_ga.core.operator.Operator` interface over the ``list[int]`` binary
representation (bitstrings of 0/1 item-selection decisions).

The concrete classes are not implemented yet: this file is the executable specification
of their behaviour. Expected public names (in ``aos_ga.operators.binary``):
``UniformCrossover`` (``operator_id="uniform"``, recombinative, arity 2) and
``BitFlipMutation`` (``operator_id="bitflip"``, perturbative, arity 1).

Frozen contract (specialises the operator interface for binary genomes):
- ``apply(parents, rng) -> child`` returns ONE child that is a fresh ``list[int]`` of the
  same length as its parents with every element in ``{0, 1}``, drawing randomness only
  from the injected ``rng`` and being deterministic for a fixed seed.
- ``len(parents)`` must equal ``arity`` or ``ValueError`` is raised.
- The child is always a fresh list; parents are never mutated.
- Uniform crossover draws one per-bit Bernoulli(1/2) parent selector and sets
  ``child[i] = parents[selector[i]][i]``: every bit is inherited from one of the two
  parents at that position. It returns ONE child -- the complementary child (the mirror
  selector) is never built, so no second evaluation is spent. Because the selector's
  complement is equiprobable, that single child is already a uniform draw from the pair,
  and no extra "which parent first" bit is needed (unlike order crossover). At positions
  where the parents agree the child inherits that shared bit; identical parents are
  reproduced verbatim.
- Bit-flip mutation flips each bit independently with probability ``1/n`` for
  ``n = len(parent)`` measured inside the operator. There is NO guarantee of at least one
  flip, so ``child == parent`` is a legal outcome -- the deliberate contrast with the
  permutation inversion operator, which always changes the genome. It can flip zero, one
  or several bits, and can therefore change the number of ones in the chromosome.

The rest of the binary pool (single-point crossover and binary swap for the full pool) is
out of the slice and specified when it is assembled.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.representation import Representation
from aos_ga.operators.binary import BitFlipMutation, UniformCrossover

_UNIFORM = UniformCrossover()
_BITFLIP = BitFlipMutation()
_SLICE_OPERATORS: list[Operator[list[int]]] = [_UNIFORM, _BITFLIP]


# --- Test oracle (verifier, not generator) -------------------------------------


def _is_uniform_child(child: list[int], first: list[int], second: list[int]) -> bool:
    """True if every bit of ``child`` was inherited from a parent at that position.

    The definition of uniform crossover: for each index the child's bit must equal one
    of the two parents' bits there. This checks the operator's output without reproducing
    its random selector mask.
    """
    return len(child) == len(first) == len(second) and all(
        child[i] in (first[i], second[i]) for i in range(len(child))
    )


# --- Oracle self-checks (the verifier above must itself be correct) ------------


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


# --- Sample parents ------------------------------------------------------------


def _sample_parents(operator: Operator[list[int]]) -> list[list[int]]:
    """Two equal-length bitstrings, trimmed to the operator's arity.

    They agree at indices 0, 3, 4, 7 and differ at 1, 2, 5, 6, so tests can tell an
    inherited shared bit from a recombined one.
    """
    parents = [[1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0]]
    return parents[: operator.arity]


# --- Metadata ------------------------------------------------------------------


def test_operators_are_operator_instances() -> None:
    for operator in _SLICE_OPERATORS:
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


# --- Arity enforcement ---------------------------------------------------------


def test_uniform_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _UNIFORM.apply([[1, 0, 1, 0]], np.random.default_rng(0))  # one parent, needs two


def test_bitflip_rejects_wrong_parent_count() -> None:
    with pytest.raises(ValueError):
        _BITFLIP.apply([[1, 0, 1], [0, 1, 0]], np.random.default_rng(0))  # two, needs one


# --- Shared contract: one fresh 0/1 child of equal length, parents untouched ----


@pytest.mark.parametrize("operator", _SLICE_OPERATORS, ids=lambda op: op.operator_id)
def test_apply_returns_a_fresh_binary_child(operator: Operator[list[int]]) -> None:
    parents = _sample_parents(operator)
    child = operator.apply(parents, np.random.default_rng(0))

    assert isinstance(child, list)
    assert all(bit in (0, 1) for bit in child)  # a 0/1 bitstring
    assert len(child) == len(parents[0])  # one child, same genome length
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
    # so the count of ones is not conserved (contrast with the deferred binary swap).
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
