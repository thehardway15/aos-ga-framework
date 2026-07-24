"""Instantaneous-reward credit assignment.

Turns the outcome of one operator application into the non-negative reward an AOS
strategy learns from, by asking the narrowest possible question: did the child
improve on the material the operator was given? Everything that answer needs -- the
child's quality and the reference quality of its parents -- exists at the
reproduction event itself, which is what makes this the one credit scheme that fits
the frozen variation seam unchanged: the step records ``g_ref`` when it draws the
parents and receives ``child_quality`` in ``observe``. The population-wide schemes
(rewarded fitness improvement, rank-based) need the generation's spread or the whole
population and would have to widen that seam first.

Everything here works in the unified quality ``g`` (more is better), so a
minimization and a maximization problem earn identical credit for identical
progress: the sign of the objective is settled once, in ``Problem.g``, and never
re-derived downstream. The functions are pure and hold no state -- accumulating
history is the strategy's job, not credit assignment's.
"""

from __future__ import annotations

from collections.abc import Sequence


def reference_quality(parent_qualities: Sequence[float]) -> float:
    """Return ``g_ref``: the best quality among the operator's parents.

    One rule covers both arities -- the sole parent for a mutation, the better of
    the two for a crossover -- so the caller never branches on arity. Parents are
    already evaluated when the variation step draws them, so this costs nothing.
    The rule deliberately favours unary operators (a mutation must beat one parent,
    a crossover the stronger of two), which is why the zero-reward rate is later
    reported separately for recombinative and perturbative operators. An empty
    parent set has no reference to improve on and raises ``ValueError``.
    """
    if not parent_qualities:
        raise ValueError("parent_qualities must be non-empty")

    return max(parent_qualities)


def instant_reward(child_quality: float, g_ref: float) -> float:
    """Return the child's improvement over ``g_ref``, clamped at zero.

    ``max(0.0, child_quality - g_ref)``: a child that fails to beat its parents
    earns exactly ``0.0`` and never negative credit, which is the non-negativity the
    strategies rely on. Both arguments are ``g`` values, so the reward is
    direction-agnostic by construction. Nothing is validated numerically --
    ``Problem.repair`` and ``Problem.g`` yield finite values in every problem family.
    """
    return max(0.0, child_quality - g_ref)


def is_zero_reward(reward: float) -> bool:
    """Flag a reproduction event that earned no credit.

    Exact equality with ``0.0``, with no tolerance: the clamp in
    :func:`instant_reward` yields a literal zero, so a threshold would be fiction and
    would silently reclassify genuine improvements. Zero rewards are data, not a
    degenerate case to be smoothed away with an artificial epsilon -- a short run may
    legitimately produce long zero-reward stretches, and their rate is a reported
    result rather than a defect.
    """
    return reward == 0.0
