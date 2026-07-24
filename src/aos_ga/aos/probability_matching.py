"""Probability Matching: selection proportional to smoothed operator quality.

The first *learning* strategy of the family. It keeps one quality estimate per
operator, smooths it exponentially towards the rewards that operator earned, and
selects proportionally to those estimates while reserving a floor of exploration
probability for every arm:

    q_i <- q_i + alpha * (r - q_i)        (the selected arm only)
    p_i  = p_min + (1 - K * p_min) * q_i / sum_j q_j

Smoothing -- rather than accumulating -- is what makes the estimate answer "how much
does this operator deliver *when applied*" instead of conflating quality with how
often the arm happened to be drawn. The floor keeps every arm recoverable: at the
budgets studied here an operator that is unlucky in its first few applications would
otherwise be written off long before the run ends.

Parameters come from the a priori strategy table and are never re-tuned:
``alpha = 0.1``, ``p_min = 0.05``. The initial estimate is a project decision, since
the sources fix both parameters but not ``q_i(0)``, and this strategy -- unlike UCB
and DMAB -- gets no round-robin warm-up to derive it from data. It is pinned at
``1.0``: the selection formula is undefined for a zero sum, so a zero start is
excluded outright, and any equal positive start yields ``p_i = 1/K``, making the
strategy's first step an exact degeneration to Random selection. Because selection is
proportional to raw estimates, the strategy is sensitive to the scale of the rewards
a credit scheme produces -- a known property of the method, not a defect to correct
here.
"""

from __future__ import annotations

from collections.abc import Sequence

from numpy.random import Generator

from .strategy import AosStrategy


class ProbabilityMatching(AosStrategy):
    """Selects an operator with probability proportional to its quality estimate.

    The estimates are the only state: the selection distribution is derived from them
    on demand rather than stored, so the two can never drift apart. Selection spends
    exactly one draw of the injected generator, and learning happens solely in
    :meth:`update`.
    """

    def __init__(
        self, operator_ids: Sequence[str], *, alpha: float = 0.1, p_min: float = 0.05
    ) -> None:
        """Set up one quality estimate per arm, all equal to ``1.0``.

        ``alpha`` is the smoothing rate: small values react slowly to new observations,
        large ones follow the noise. ``p_min`` is the probability reserved for every
        arm regardless of its record, so the pool consumes ``K * p_min`` of the total
        mass and the rest is distributed proportionally -- which is why a floor the
        pool cannot afford is rejected rather than silently renormalized. Raises
        ``ValueError`` on an empty arm list, duplicate ids, ``alpha`` outside
        ``(0, 1]``, a negative ``p_min`` or ``K * p_min > 1``.
        """
        if not operator_ids:
            raise ValueError("operator_ids must contain at least one arm")
        if len(set(operator_ids)) != len(operator_ids):
            raise ValueError(f"operator_ids must be unique, got {tuple(operator_ids)}")
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if p_min < 0.0:
            raise ValueError(f"p_min must be non-negative, got {p_min}")
        if len(operator_ids) * p_min > 1.0:
            raise ValueError(
                f"p_min={p_min} reserves more than the whole probability mass "
                f"for {len(operator_ids)} arms"
            )

        self._operator_ids = tuple(operator_ids)
        self._alpha = alpha
        self._p_min = p_min
        self._quality = dict.fromkeys(self._operator_ids, 1.0)

    @property
    def operator_ids(self) -> tuple[str, ...]:
        """The arms in pool order; fixed for the lifetime of the strategy."""
        return self._operator_ids

    def select_operator(self, rng: Generator) -> str:
        """Draw one arm with probability ``p_i``, spending a single ``rng`` draw.

        Maps one uniform draw through the cumulative distribution taken in
        ``operator_ids`` order, so the operator is the first random decision of a
        reproduction event -- the same shape as the Random baseline, and the reason a
        seed replays the whole run. The walk is written out rather than delegated to
        ``rng.choice`` so the mapping from draw to arm is auditable here rather than in
        a library internal, and so it costs exactly one draw. It buys clarity, not
        cross-version stability: `Generator` streams are outside NumPy's compatibility
        guarantee either way, which is why the replication environment pins the version
        instead. The final arm closes the walk, covering the rounding case where the
        probabilities sum a hair below one. The estimates are left untouched.
        """
        probabilities = self._probabilities()
        threshold = float(rng.random())
        cumulative = 0.0
        for operator_id in self._operator_ids:
            cumulative += probabilities[operator_id]
            if threshold < cumulative:
                return operator_id
        return self._operator_ids[-1]

    def update(self, operator_id: str, reward: float) -> None:
        """Smooth the selected arm's estimate towards ``reward``.

        Touches that arm only -- the others carry no information about this
        reproduction event. A zero reward is an ordinary update that decays the arm by
        ``(1 - alpha)``, which is how an operator that stops paying off loses ground.
        Raises ``KeyError`` for an id that is not an arm and ``ValueError`` for a
        negative reward, both of which mean the caller is wired wrong: credit
        assignment guarantees non-negative rewards over this exact arm set.
        """
        if operator_id not in self._quality:
            raise KeyError(f"unknown operator_id {operator_id!r}, arms are {self._operator_ids}")
        if reward < 0.0:
            raise ValueError(f"reward must be non-negative, got {reward}")

        self._quality[operator_id] += self._alpha * (reward - self._quality[operator_id])

    def snapshot_state(self) -> dict[str, object]:
        """Return the current distribution and estimates, as fresh copies.

        Both mappings are rebuilt per call, so the dynamics log can never write back
        into the strategy. ``probabilities`` is derived here rather than read from a
        stored field, which keeps the estimates the single source of truth.
        """
        state: dict[str, object] = {
            "probabilities": self._probabilities(),
            "quality_estimates": dict(self._quality),
        }
        return state

    def _probabilities(self) -> dict[str, float]:
        """The selection distribution implied by the current estimates.

        Applies the matching formula, which sums to one by construction: the arms
        share ``K * p_min`` as a floor and split the remainder in proportion to their
        estimates, so no renormalization is needed -- and none is applied, because
        rescaling would quietly cancel the floor it is meant to preserve. Should the
        estimates ever sum to zero the formula is undefined, and selection falls back
        to the uniform distribution.
        """
        total = sum(self._quality.values())
        if total == 0.0:
            return dict.fromkeys(self._operator_ids, 1.0 / len(self._operator_ids))

        scale = 1.0 - len(self._operator_ids) * self._p_min
        return {
            operator_id: self._p_min + scale * (estimate / total)
            for operator_id, estimate in self._quality.items()
        }
