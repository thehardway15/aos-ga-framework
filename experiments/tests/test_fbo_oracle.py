"""Contract spec for deriving the reference operator (``o*``) from per-operator medians.

The benchmark runs every operator on its own (30 repetitions) and aggregates each to a single
median final quality -- the upper reference point for the adaptive operator-selection strategies.
``derive_oracle`` is the pure, I/O-free step
that turns that per-operator median table into the reference operator set for one pool: the
operators that share the maximum median, plus their common median. It is deliberately isolated
from the GA runs and from CSV writing so the whole tie/maximizer/projection logic can be pinned
on a controlled double of known medians.

The concrete module is the executable target of this specification. Expected public names
(in ``experiments.baselines.fbo_oracle``):

- ``OperatorOracle`` -- a frozen dataclass with fields ``o_star: tuple[str, ...]`` (the
  maximizer set, in pool-membership order) and ``o_star_median: float`` (the single maximum
  median shared by every member), plus a derived property ``o_star_count -> int`` equal to
  ``len(o_star)``.
- ``derive_oracle(operator_medians, pool_members) -> OperatorOracle`` where
  ``operator_medians`` is a mapping ``operator_id -> median`` and ``pool_members`` is the
  ordered ids of the pool to derive over.

Frozen contract:
- Quality is already in the "more is better" convention (``g``; for minimisation ``g = -f`` is
  computed upstream), so ``derive_oracle`` MAXIMISES the median.
- ``o_star`` is every operator in ``pool_members`` whose median equals the maximum median taken
  over ``pool_members``, listed in ``pool_members`` order. ``o_star_median`` is that maximum,
  shared by all members. ``o_star_count`` equals ``len(o_star)``.
- Equality is exact float ``==`` -- no tolerance and no tie-break. Near-but-unequal medians
  do not tie; exactly equal ones do. There is no secondary key (mean, std, order): the function
  reads only the medians it is given.
- ``pool_members`` defines both the subset considered and the deterministic order. Keys of
  ``operator_medians`` outside ``pool_members`` are ignored -- this is how a single FULL median
  table projects onto both the FULL and the REDUCED oracle with no extra GA runs.
- Medians are native ``float`` values; the harness computes them once per operator
  (pool-independently) and casts the numpy medians to ``float`` before calling.
- Preconditions: empty ``pool_members`` raises ``ValueError``; a member absent from
  ``operator_medians`` raises ``ValueError``.
- Serialisation (joining ``o_star`` with ``;``, the CSV columns) and the configuration key
  (problem, instance, ``N``, ``G``, ``pool_variant``) live in the harness, not here.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from experiments.baselines.fbo_oracle import OperatorOracle, derive_oracle

# One median table, reused to show that a single pool-independent measurement projects onto
# both pools. ``singlepoint`` is the global best but is absent from the reduced pool, so FULL and
# REDUCED must resolve to different reference operators from THESE SAME medians.
_BINARY_MEDIANS: dict[str, float] = {
    "singlepoint": 100.0,
    "uniform": 90.0,
    "bitflip": 85.0,
    "swapbit": 70.0,
}
_BINARY_FULL: tuple[str, ...] = ("singlepoint", "uniform", "bitflip", "swapbit")
_BINARY_REDUCED: tuple[str, ...] = ("uniform", "bitflip")


# --- Core semantics: the maximizer set and its shared median -----------------------


def test_singleton_maximizer() -> None:
    # One strict maximum: the oracle is that single operator and its median.
    oracle = derive_oracle({"a": 3.0, "b": 1.0, "c": 2.0}, ("a", "b", "c"))
    assert oracle.o_star == ("a",)
    assert oracle.o_star_median == 3.0


def test_partial_tie_keeps_all_maximizers() -> None:
    # Two operators share the maximum, a third is lower: both maximizers are in the set, the
    # lower one is not, and the shared median is the common maximum. No tie-break narrows this to
    # one -- and no other statistic is consulted, only the medians given.
    oracle = derive_oracle({"a": 5.0, "b": 5.0, "c": 3.0}, ("a", "b", "c"))
    assert oracle.o_star == ("a", "b")
    assert oracle.o_star_median == 5.0


def test_full_tie_returns_the_whole_pool() -> None:
    # All medians equal: every operator is a maximizer, so the set is the whole membership.
    oracle = derive_oracle({"ox": 4.0, "cx": 4.0, "inversion": 4.0}, ("ox", "cx", "inversion"))
    assert oracle.o_star == ("ox", "cx", "inversion")
    assert oracle.o_star_median == 4.0


def test_o_star_follows_membership_order_not_value_or_insertion() -> None:
    # The set is ordered by ``pool_members``, giving a stable, reproducible artefact. Here the
    # mapping's insertion order ("c", "a", "b") differs from the membership order, and two
    # operators tie for the max: the result must follow membership ("a" before "c"), not the
    # dict order and not any value ranking.
    medians = {"c": 5.0, "a": 5.0, "b": 3.0}
    oracle = derive_oracle(medians, ("a", "b", "c"))
    assert oracle.o_star == ("a", "c")


# --- FULL -> REDUCED projection: one median table, two pools -----------------------


def test_same_medians_project_to_a_different_oracle_per_pool() -> None:
    # Core projection: the FULL and REDUCED oracle are derived from the SAME median table (the
    # measurement is pool-independent). With ``singlepoint`` the global best but outside REDUCED,
    # the two pools resolve to different reference operators and different reference qualities.
    full = derive_oracle(_BINARY_MEDIANS, _BINARY_FULL)
    reduced = derive_oracle(_BINARY_MEDIANS, _BINARY_REDUCED)

    assert full.o_star == ("singlepoint",)
    assert full.o_star_median == 100.0

    assert reduced.o_star == ("uniform",)
    assert reduced.o_star_median == 90.0


def test_operators_outside_membership_are_ignored() -> None:
    # ``pool_members`` selects which medians count. A high-valued operator absent from the pool
    # (here ``singlepoint`` at 100) does not leak into the result -- this is exactly what lets a
    # FULL median table serve the REDUCED oracle without a separate run.
    oracle = derive_oracle(_BINARY_MEDIANS, _BINARY_REDUCED)
    assert "singlepoint" not in oracle.o_star
    assert oracle.o_star_median == 90.0


# --- Exact equality, no tolerance --------------------------------------------------


def test_near_but_unequal_medians_do_not_tie() -> None:
    # Equality is exact ``==`` with no tolerance: a difference far below any meaningful scale
    # still breaks the tie, so only the strictly larger operator is the maximizer.
    medians = {"a": 1.0, "b": 1.0 + 1e-9}
    oracle = derive_oracle(medians, ("a", "b"))
    assert oracle.o_star == ("b",)
    assert oracle.o_star_median == 1.0 + 1e-9


def test_exactly_equal_discrete_medians_tie() -> None:
    # On discrete representations (knapsack values, TSP tour lengths carried as ``g``) equal
    # medians are real and their equality is natural, so both belong in the set.
    oracle = derive_oracle({"a": 301.0, "b": 301.0, "c": 278.0}, ("a", "b", "c"))
    assert oracle.o_star == ("a", "b")
    assert oracle.o_star_median == 301.0


# --- The result object -------------------------------------------------------------


@pytest.mark.parametrize(
    ("medians", "members", "expected_count"),
    [
        ({"a": 3.0, "b": 1.0}, ("a", "b"), 1),
        ({"a": 5.0, "b": 5.0, "c": 3.0}, ("a", "b", "c"), 2),
        ({"a": 4.0, "b": 4.0, "c": 4.0}, ("a", "b", "c"), 3),
    ],
    ids=["singleton", "pair-tie", "full-tie"],
)
def test_o_star_count_equals_len_of_o_star(
    medians: dict[str, float], members: tuple[str, ...], expected_count: int
) -> None:
    # ``o_star_count`` is derived from ``o_star`` (single source of truth), so it never drifts.
    oracle = derive_oracle(medians, members)
    assert oracle.o_star_count == len(oracle.o_star) == expected_count


def test_o_star_is_a_tuple() -> None:
    # A tuple, not a set/list: ordered and hashable, so the artefact is stable across writes.
    oracle = derive_oracle({"a": 2.0, "b": 1.0}, ("a", "b"))
    assert isinstance(oracle.o_star, tuple)


def test_oracle_is_immutable() -> None:
    # The oracle is a value object: once derived it cannot be mutated in place.
    oracle = derive_oracle({"a": 2.0, "b": 1.0}, ("a", "b"))
    field = "o_star_median"
    with pytest.raises(FrozenInstanceError):
        setattr(oracle, field, 0.0)


def test_returns_an_operator_oracle() -> None:
    assert isinstance(derive_oracle({"a": 1.0}, ("a",)), OperatorOracle)


# --- Preconditions fail loudly -----------------------------------------------------


def test_empty_pool_members_is_an_error() -> None:
    # An empty pool has no maximum to take, so it is a loud error rather than an empty oracle.
    with pytest.raises(ValueError):
        derive_oracle({"a": 1.0}, ())


def test_member_without_a_median_is_an_error() -> None:
    # Every pool member must have a measured median. A missing one is a programming error in the
    # harness, surfaced loudly rather than silently skipped.
    with pytest.raises(ValueError):
        derive_oracle({"a": 1.0}, ("a", "b"))
