"""Contract spec for instant-reward credit assignment.

Credit assignment turns the outcome of one operator application into the
non-negative reward an AOS strategy learns from. The instantaneous variant asks the
narrowest possible question: did the child improve on the material the operator was
given? Everything it needs -- the child's quality and the reference quality of its
parents -- is available at the reproduction event itself, which is why it is the
one scheme that fits the frozen variation seam (``observe(child_quality)`` plus the
``g_ref`` the step recorded when it drew the parents) without widening it.

The module works exclusively in the unified quality ``g`` (more is better), so a
minimization and a maximization problem earn identical credit for identical
progress -- the sign of the objective is settled once, in ``Problem.g``, and never
re-derived here.

Nothing is implemented yet: this file is the executable specification. Expected
public names (in ``aos_ga.credit.instant``): ``reference_quality``,
``instant_reward`` and ``is_zero_reward``, all pure functions -- credit assignment
holds no state, the strategy does.

Frozen contract (instant reward):
- ``reference_quality(parent_qualities) -> g_ref`` is ``max`` over the operator's
  parent set: the sole parent for a mutation, the better of the two for a crossover.
  One rule covers both arities, so the step that calls it never branches on arity.
  An empty parent set has no reference and is rejected with ``ValueError``.
- ``instant_reward(child_quality, g_ref) -> max(0.0, child_quality - g_ref)``. A
  child that fails to beat its parents earns exactly ``0.0``, never a negative
  reward: the strategies consume non-negative credit by contract. No numeric
  validation -- ``repair`` and ``g`` produce finite values in every problem family.
- ``is_zero_reward(reward) -> bool`` is exact equality with ``0.0``, no tolerance:
  the ``max(0.0, ...)`` clamp yields a literal zero, so a threshold would be
  fiction. Zero rewards are data, not a degenerate case to smooth away with an
  artificial epsilon -- a run may legitimately produce long zero-reward stretches,
  and reporting their rate (split by recombinative and perturbative operators, since
  the ``g_ref`` rule favours unary ones) is a result in its own right.
"""

from __future__ import annotations

import numpy as np
import pytest

from aos_ga.core.problem import Direction, quality
from aos_ga.credit.instant import instant_reward, is_zero_reward, reference_quality

# --- reference_quality: g_ref over the operator's parent set ---------------------


def test_reference_quality_of_one_parent_is_that_parents_quality() -> None:
    # The mutation case: the sole parent is the material the child must beat.
    assert reference_quality([4.5]) == 4.5


def test_reference_quality_of_two_parents_is_the_better_one() -> None:
    # The crossover case: the child is credited only for beating the stronger parent.
    assert reference_quality([2.0, 7.0]) == 7.0


def test_reference_quality_ignores_parent_order() -> None:
    assert reference_quality([7.0, 2.0]) == reference_quality([2.0, 7.0])


def test_reference_quality_of_equal_parents_is_that_value() -> None:
    assert reference_quality([3.25, 3.25]) == 3.25


def test_reference_quality_works_on_the_minimization_scale() -> None:
    # Under minimization g = -f, so every quality is negative and the better parent
    # is the one closest to zero; a plain max still picks it.
    assert reference_quality([-10.0, -8.0]) == -8.0


def test_reference_quality_accepts_a_tuple() -> None:
    assert reference_quality((2.0, 7.0)) == 7.0


def test_reference_quality_rejects_an_empty_parent_set() -> None:
    # No parents means no reference to improve on -- a caller bug, not a zero.
    with pytest.raises(ValueError):
        reference_quality([])


# --- instant_reward: improvement over the reference, clamped at zero -------------


@pytest.mark.parametrize(
    ("child_quality", "g_ref", "expected"),
    [
        (5.0, 3.0, 2.0),  # improvement is credited in full
        (3.0, 3.0, 0.0),  # a tie gains nothing
        (1.0, 3.0, 0.0),  # a regression is clamped, never negative
        (-8.0, -10.0, 2.0),  # minimization scale: negative g, positive progress
        (-12.0, -10.0, 0.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_reward_is_the_clamped_improvement(
    child_quality: float, g_ref: float, expected: float
) -> None:
    assert instant_reward(child_quality, g_ref) == pytest.approx(expected)


def test_no_improvement_yields_exactly_zero() -> None:
    # Exactly 0.0, not a small residue: ``is_zero_reward`` compares without tolerance.
    assert instant_reward(3.0, 3.0) == 0.0


def test_a_worse_child_yields_exactly_zero() -> None:
    assert instant_reward(-5.0, 12.0) == 0.0


def test_a_tiny_improvement_is_not_clamped_away() -> None:
    # Credit is not thresholded: a barely-better child still earns its difference.
    assert instant_reward(1.000000000001, 1.0) > 0.0


@pytest.mark.parametrize("seed", range(8))
def test_reward_is_never_negative(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for child_quality, g_ref in rng.normal(scale=100.0, size=(50, 2)):
        assert instant_reward(float(child_quality), float(g_ref)) >= 0.0


def test_reward_is_pure() -> None:
    # No state, no accumulation: the same pair always earns the same credit.
    assert instant_reward(5.0, 3.0) == instant_reward(5.0, 3.0)


def test_reward_is_the_same_for_both_optimization_directions() -> None:
    # A minimization run improving f from 10 to 8 and a maximization run improving f
    # from 8 to 10 are the same two units of progress once read through g, so they
    # must earn identical credit. This is what the unified quality is for: the credit
    # module never learns which direction it is serving.
    minimized = instant_reward(
        quality(8.0, Direction.MINIMIZE),
        reference_quality([quality(10.0, Direction.MINIMIZE)]),
    )
    maximized = instant_reward(
        quality(10.0, Direction.MAXIMIZE),
        reference_quality([quality(8.0, Direction.MAXIMIZE)]),
    )
    assert minimized == maximized == pytest.approx(2.0)


def test_reward_composes_with_the_reference_over_two_parents() -> None:
    # End to end for a crossover event: the weaker parent never sets the bar.
    assert instant_reward(6.0, reference_quality([2.0, 7.0])) == 0.0
    assert instant_reward(9.0, reference_quality([2.0, 7.0])) == pytest.approx(2.0)


# --- is_zero_reward: the per-step diagnostic flag --------------------------------


def test_zero_is_flagged() -> None:
    assert is_zero_reward(0.0) is True


def test_a_positive_reward_is_not_flagged() -> None:
    assert is_zero_reward(2.0) is False


def test_a_tiny_positive_reward_is_not_flagged() -> None:
    # Exact comparison, no tolerance: 1e-12 of progress is progress.
    assert is_zero_reward(1e-12) is False


@pytest.mark.parametrize(
    ("child_quality", "g_ref", "flagged"),
    [
        (5.0, 3.0, False),
        (3.0, 3.0, True),
        (1.0, 3.0, True),
    ],
)
def test_flag_reads_the_output_of_instant_reward(
    child_quality: float, g_ref: float, flagged: bool
) -> None:
    # The diagnostic runs on the reward, not on the raw qualities, so the run-level
    # zero-reward rate is derived from exactly the value the strategy was handed.
    assert is_zero_reward(instant_reward(child_quality, g_ref)) is flagged
