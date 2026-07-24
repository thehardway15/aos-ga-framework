"""Contract spec for the Probability Matching operator-selection strategy.

The first *learning* member of the strategy family: it keeps one quality estimate
per operator, smooths it exponentially towards the rewards that operator earned, and
selects proportionally to those estimates while reserving a floor of exploration
probability for every arm. It is the reference point against which the more
aggressive strategies (Adaptive Pursuit, UCB, DMAB) are judged -- deliberately slow
to react, which is exactly the property under test at budgets of 20 to 50
generations.

Parameters come from the a priori strategy table and are never re-tuned:
``alpha = 0.1``, ``p_min = 0.05``. The update and selection formulas are the
textbook ones:

    q_i <- q_i + alpha * (r - q_i)        (rewarded arm only)
    p_i  = p_min + (1 - K * p_min) * q_i / sum_j q_j

The initial estimate is a project decision rather than a quotation: the sources fix
``alpha`` and ``p_min`` but say nothing about ``q_i(0)``, and Probability Matching --
unlike UCB and DMAB -- gets no round-robin warm-up to derive it from data. It is
pinned at ``1.0``: the selection formula is undefined for a zero sum, so a zero start
is excluded outright, and any equal positive start yields ``p_i = 1/K``, which makes
the strategy's first step an exact degeneration to Random selection. The magnitude
matters only for how fast the estimates catch up with the reward scale.

Nothing is implemented yet: this file is the executable specification. Expected
public name (in ``aos_ga.aos.probability_matching``): ``ProbabilityMatching``, a
subclass of ``aos_ga.aos.strategy.AosStrategy``.

Frozen contract (Probability Matching):
- ``ProbabilityMatching(operator_ids, *, alpha=0.1, p_min=0.05)``. All estimates
  start at ``1.0``. Rejected with ``ValueError``: an empty arm list, duplicate ids,
  ``alpha`` outside ``(0, 1]``, a negative ``p_min`` and ``K * p_min > 1`` (which
  would make ``1 - K * p_min`` negative and hence some ``p_i`` negative).
- ``update`` smooths the SELECTED arm only, leaving every other estimate untouched.
  A zero reward is an ordinary update that decays the arm by ``(1 - alpha)``; a
  negative reward is a credit-module bug and raises ``ValueError``; an unknown id
  raises ``KeyError``.
- ``select_operator`` consumes exactly ONE ``rng.random()`` draw and maps it through
  the cumulative distribution taken in ``operator_ids`` order, so the operator is the
  first random decision of a reproduction event, as in ``RandomOperatorStep``. It
  does not touch the estimates.
- ``snapshot_state`` derives ``probabilities`` from the estimates on every call --
  the estimates are the single source of truth, there is no stored distribution --
  and exposes them alongside ``quality_estimates``. Should the estimates ever sum to
  zero, selection falls back to the uniform distribution instead of dividing by zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from aos_ga.aos.probability_matching import ProbabilityMatching
from aos_ga.aos.strategy import AosStrategy

_ARMS = ("ox", "cx", "inversion")
_ALPHA = 0.1
_P_MIN = 0.05


def _strategy(operator_ids: tuple[str, ...] = _ARMS, **kwargs: float) -> ProbabilityMatching:
    return ProbabilityMatching(operator_ids, **kwargs)


def _boosted() -> ProbabilityMatching:
    """A strategy whose first arm has been rewarded into a clear lead."""
    strategy = _strategy()
    for _ in range(10):
        strategy.update("ox", 5.0)
    return strategy


def _entry(strategy: AosStrategy, key: str) -> dict[str, float]:
    """Read one per-arm mapping out of a snapshot, typed."""
    entry = strategy.snapshot_state()[key]
    assert isinstance(entry, dict)
    return {str(arm): float(value) for arm, value in entry.items()}


def _probabilities(strategy: AosStrategy) -> dict[str, float]:
    return _entry(strategy, "probabilities")


def _estimates(strategy: AosStrategy) -> dict[str, float]:
    return _entry(strategy, "quality_estimates")


# --- construction and parameter validation ---------------------------------------


def test_probability_matching_is_an_aos_strategy() -> None:
    assert isinstance(_strategy(), AosStrategy)


def test_operator_ids_are_kept_in_pool_order() -> None:
    assert _strategy().operator_ids == _ARMS


def test_rejects_an_empty_arm_list() -> None:
    with pytest.raises(ValueError):
        _strategy(())


def test_rejects_duplicate_operator_ids() -> None:
    # Two arms sharing an id would silently merge their credit.
    with pytest.raises(ValueError):
        _strategy(("ox", "cx", "ox"))


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_rejects_an_alpha_outside_the_unit_interval(alpha: float) -> None:
    # alpha = 0 would freeze the estimates; above 1 the smoothing overshoots.
    with pytest.raises(ValueError):
        _strategy(alpha=alpha)


def test_accepts_alpha_of_one() -> None:
    # The boundary is legal and degenerates to "forget everything but the last reward".
    _strategy(alpha=1.0)  # no raise


def test_rejects_a_negative_p_min() -> None:
    with pytest.raises(ValueError):
        _strategy(p_min=-0.01)


def test_accepts_a_zero_p_min() -> None:
    # Legal: pure proportional matching, no reserved exploration.
    _strategy(p_min=0.0)  # no raise


def test_rejects_a_p_min_the_pool_cannot_afford() -> None:
    # K * p_min > 1 makes (1 - K * p_min) negative, hence negative probabilities.
    with pytest.raises(ValueError):
        _strategy(p_min=0.4)  # K = 3


def test_accepts_a_p_min_that_exactly_exhausts_the_budget() -> None:
    # K * p_min == 1 leaves nothing to distribute: every arm sits at p_min.
    strategy = _strategy(p_min=1.0 / len(_ARMS))
    assert _probabilities(strategy) == pytest.approx(dict.fromkeys(_ARMS, 1.0 / len(_ARMS)))


# --- the start: an exact degeneration to Random selection ------------------------


@pytest.mark.parametrize("arm_count", [2, 3, 4, 6])
def test_initial_probabilities_are_uniform(arm_count: int) -> None:
    # Equal estimates give p_i = 1/K for any pool size, so at t = 0 Probability
    # Matching IS Random selection -- the lower reference point is its starting state.
    arms = tuple(f"op{index}" for index in range(arm_count))
    assert _probabilities(_strategy(arms)) == pytest.approx(dict.fromkeys(arms, 1.0 / arm_count))


def test_initial_estimates_are_equal_and_positive() -> None:
    # Positive is not cosmetic: the selection formula divides by their sum.
    estimates = _estimates(_strategy())
    assert set(estimates) == set(_ARMS)
    assert len(set(estimates.values())) == 1
    assert all(value > 0.0 for value in estimates.values())


def test_initial_selection_frequencies_match_a_uniform_draw() -> None:
    strategy = _strategy(("ox", "cx"))
    rng = np.random.default_rng(0)
    draws = [strategy.select_operator(rng) for _ in range(1000)]
    assert 400 <= draws.count("ox") <= 600


# --- update: exponential smoothing on the selected arm only ----------------------


def test_update_smooths_the_estimate_of_the_rewarded_arm() -> None:
    # q <- 1.0 + 0.1 * (3.0 - 1.0) = 1.2, computed by hand from the frozen formula.
    strategy = _strategy()
    strategy.update("ox", 3.0)
    assert _estimates(strategy)["ox"] == pytest.approx(1.0 + _ALPHA * (3.0 - 1.0))


def test_update_leaves_the_other_estimates_untouched() -> None:
    # Only the arm that was actually applied earns or loses credit; the rest carry no
    # information about this reproduction event.
    strategy = _strategy()
    strategy.update("ox", 3.0)
    estimates = _estimates(strategy)
    assert estimates["cx"] == pytest.approx(1.0)
    assert estimates["inversion"] == pytest.approx(1.0)


def test_a_zero_reward_decays_the_arm_by_one_minus_alpha() -> None:
    # Zero rewards are frequent under instant reward and are an ordinary update: they
    # push the estimate down rather than being ignored.
    strategy = _strategy()
    strategy.update("ox", 0.0)
    assert _estimates(strategy)["ox"] == pytest.approx(1.0 - _ALPHA)


def test_repeated_updates_converge_towards_the_reward() -> None:
    strategy = _strategy()
    for _ in range(200):
        strategy.update("ox", 5.0)
    assert _estimates(strategy)["ox"] == pytest.approx(5.0, abs=1e-6)


def test_alpha_of_one_replaces_the_estimate_outright() -> None:
    strategy = _strategy(alpha=1.0)
    strategy.update("ox", 7.0)
    assert _estimates(strategy)["ox"] == pytest.approx(7.0)


def test_update_rejects_a_negative_reward() -> None:
    # Credit assignment guarantees non-negative rewards; a negative one is a bug in
    # the caller, not a signal to learn from.
    with pytest.raises(ValueError):
        _strategy().update("ox", -0.5)


def test_update_rejects_an_unknown_arm() -> None:
    with pytest.raises(KeyError):
        _strategy().update("sbx", 1.0)


# --- probabilities: the matching formula, floored by p_min -----------------------


def test_probabilities_follow_the_matching_formula() -> None:
    # Hand-computed from q = (1.2, 1.0, 1.0), sum 3.2, p_min = 0.05, K = 3:
    # p_ox = 0.05 + 0.85 * 1.2 / 3.2, the others 0.05 + 0.85 * 1.0 / 3.2.
    strategy = _strategy()
    strategy.update("ox", 3.0)
    scale = 1.0 - len(_ARMS) * _P_MIN
    assert _probabilities(strategy) == pytest.approx(
        {
            "ox": _P_MIN + scale * 1.2 / 3.2,
            "cx": _P_MIN + scale * 1.0 / 3.2,
            "inversion": _P_MIN + scale * 1.0 / 3.2,
        }
    )


def test_probabilities_are_derived_from_the_estimates_not_stored() -> None:
    # The estimates are the single source of truth: recomputing the formula from the
    # snapshot's own quality_estimates must reproduce its probabilities exactly.
    strategy = _boosted()
    strategy.update("cx", 0.0)
    estimates = _estimates(strategy)
    total = sum(estimates.values())
    scale = 1.0 - len(_ARMS) * _P_MIN
    assert _probabilities(strategy) == pytest.approx(
        {arm: _P_MIN + scale * value / total for arm, value in estimates.items()}
    )


@pytest.mark.parametrize("seed", range(6))
def test_probabilities_sum_to_one_after_an_arbitrary_reward_history(seed: int) -> None:
    rng = np.random.default_rng(seed)
    strategy = _strategy(("ox", "pmx", "cx", "swap"))
    for _ in range(80):
        arm = strategy.select_operator(rng)
        strategy.update(arm, float(rng.exponential(scale=2.0)))
    assert sum(_probabilities(strategy).values()) == pytest.approx(1.0)


def test_no_arm_ever_falls_below_p_min() -> None:
    # The floor is what keeps a briefly unlucky operator recoverable at short budgets.
    strategy = _strategy()
    for _ in range(500):
        strategy.update("ox", 0.0)
    assert min(_probabilities(strategy).values()) >= _P_MIN


def test_a_rewarded_arm_gains_probability_and_the_others_lose_it() -> None:
    strategy = _strategy()
    before = _probabilities(strategy)
    strategy.update("ox", 4.0)
    after = _probabilities(strategy)
    assert after["ox"] > before["ox"]
    assert after["cx"] < before["cx"]
    assert after["inversion"] < before["inversion"]


def test_equal_rewards_on_every_arm_keep_the_distribution_uniform() -> None:
    strategy = _strategy()
    for arm in _ARMS:
        strategy.update(arm, 2.0)
    assert _probabilities(strategy) == pytest.approx(dict.fromkeys(_ARMS, 1.0 / len(_ARMS)))


def test_a_zero_p_min_gives_pure_proportional_matching() -> None:
    strategy = _strategy(p_min=0.0)
    strategy.update("ox", 3.0)
    assert _probabilities(strategy) == pytest.approx(
        {"ox": 1.2 / 3.2, "cx": 1.0 / 3.2, "inversion": 1.0 / 3.2}
    )


def test_estimates_summing_to_zero_fall_back_to_a_uniform_distribution() -> None:
    # Reachable only at alpha = 1 with a zero reward on every arm, but the guard has
    # to exist: the matching formula divides by the sum of the estimates.
    strategy = _strategy(alpha=1.0)
    for arm in _ARMS:
        strategy.update(arm, 0.0)
    assert _probabilities(strategy) == pytest.approx(dict.fromkeys(_ARMS, 1.0 / len(_ARMS)))


# --- selection: one draw, proportional, state-preserving -------------------------


def test_select_operator_returns_one_of_the_arms() -> None:
    strategy = _boosted()
    rng = np.random.default_rng(0)
    assert all(strategy.select_operator(rng) in _ARMS for _ in range(100))


def _arm_for_threshold(strategy: ProbabilityMatching, threshold: float) -> str:
    """The arm a cumulative walk in ``operator_ids`` order reaches at ``threshold``."""
    probabilities = _probabilities(strategy)
    cumulative = 0.0
    for arm in strategy.operator_ids:
        cumulative += probabilities[arm]
        if threshold < cumulative:
            return arm
    return strategy.operator_ids[-1]


@pytest.mark.parametrize("seed", range(12))
def test_selection_consumes_exactly_one_uniform_draw(seed: int) -> None:
    # One ``rng.random()``, mapped through the cumulative distribution in pool order:
    # the operator is the first random decision of the event, as in the Random step,
    # which is what makes a run replayable from its seed.
    strategy = _boosted()
    threshold = float(np.random.default_rng(seed).random())
    selected = strategy.select_operator(np.random.default_rng(seed))
    assert selected == _arm_for_threshold(strategy, threshold)


def test_selection_frequencies_track_the_probabilities() -> None:
    strategy = _boosted()
    expected = _probabilities(strategy)["ox"]
    rng = np.random.default_rng(0)
    draws = [strategy.select_operator(rng) for _ in range(2000)]
    assert abs(draws.count("ox") / 2000 - expected) < 0.05


def test_every_arm_stays_reachable_after_a_long_losing_streak() -> None:
    # p_min is not decorative: the crushed arm must still be drawn, or a short run
    # could never recover from an unlucky start.
    strategy = _strategy()
    for _ in range(500):
        strategy.update("cx", 0.0)
    rng = np.random.default_rng(0)
    draws = [strategy.select_operator(rng) for _ in range(2000)]
    assert draws.count("cx") > 0


def test_selection_is_deterministic_for_a_fixed_seed() -> None:
    def sequence() -> list[str]:
        strategy = _boosted()
        rng = np.random.default_rng(7)
        return [strategy.select_operator(rng) for _ in range(30)]

    assert sequence() == sequence()


def test_select_operator_does_not_touch_the_estimates() -> None:
    # Learning happens in ``update``; the draw is a pure read of the current state.
    strategy = _boosted()
    before = strategy.snapshot_state()
    strategy.select_operator(np.random.default_rng(0))
    assert strategy.snapshot_state() == before


# --- snapshot_state --------------------------------------------------------------


def test_snapshot_exposes_probabilities_and_quality_estimates() -> None:
    snapshot = _strategy().snapshot_state()
    assert set(snapshot) >= {"probabilities", "quality_estimates"}


def test_snapshot_covers_every_arm() -> None:
    strategy = _boosted()
    assert set(_probabilities(strategy)) == set(_ARMS)
    assert set(_estimates(strategy)) == set(_ARMS)


def test_snapshot_returns_fresh_dictionaries() -> None:
    strategy = _boosted()
    first, second = strategy.snapshot_state(), strategy.snapshot_state()
    assert first is not second
    assert first["probabilities"] is not second["probabilities"]
    assert first["quality_estimates"] is not second["quality_estimates"]


def test_mutating_a_snapshot_cannot_reach_the_strategy() -> None:
    strategy = _strategy()
    estimates = strategy.snapshot_state()["quality_estimates"]
    assert isinstance(estimates, dict)
    estimates["ox"] = 99.0
    assert _estimates(strategy)["ox"] == pytest.approx(1.0)


def test_snapshot_is_pure() -> None:
    strategy = _boosted()
    assert strategy.snapshot_state() == strategy.snapshot_state()
