"""Contract spec for the shared adaptive-operator-selection strategy interface.

A strategy answers exactly one question -- which operator to apply next -- and
consumes exactly one signal -- the non-negative reward the credit-assignment module
derived from the resulting child. It is the interface onto which Probability
Matching, Adaptive Pursuit, UCB and DMAB all collapse, and to which Random
selection degenerates (uniform probabilities, no learning). Everything else stays
outside: the genome, the population, the problem and the operators themselves are
never visible to a strategy, which is what makes one strategy usable across the
permutation, binary and real-valued problem families without a single branch.

The class is not implemented yet: this file is the executable specification.
Expected public name (in ``aos_ga.aos.strategy``): ``AosStrategy``.

Frozen contract (strategy seam):
- ``AosStrategy`` is a plain ``ABC``, deliberately NOT generic over the genome type
  (the contrast with ``VariationStep[Genome]`` is intentional): it trades only in
  ``operator_id: str`` and ``reward: float``, so it is independent of the problem
  class by construction, not by convention.
- ``operator_ids -> tuple[str, ...]`` names the arms in pool order and is stable for
  the lifetime of the strategy. Exposing it lets the variation step verify at
  construction time that the strategy's arms and the injected pool agree, instead of
  failing deep inside a sweep on an unknown id.
- ``select_operator(rng) -> operator_id`` returns one id drawn from
  ``operator_ids``. All randomness comes from the injected ``Generator``, so one seed
  reproduces the whole selection sequence, exactly as in ``RandomOperatorStep``. A
  deterministic rule (an argmax over indices) is legal and simply ignores ``rng``.
  The call does NOT advance the learning statistics -- counters such as the AOS step
  ``t`` and the per-arm usage ``n_i`` are the business of ``update``. This is safe
  because the skeleton alternates strictly: it produces a child, evaluates it and
  only then observes it, one reward per selection.
- ``update(operator_id, reward) -> None`` is called exactly once per
  ``select_operator``, after the child has been evaluated. ``reward >= 0`` is
  guaranteed by the credit module; zero is an ordinary, meaningful value (no
  artificial epsilon), and a run may legitimately produce long zero-reward
  sequences. An id outside ``operator_ids`` is a wiring bug and must fail loudly with
  ``KeyError``, never be absorbed silently. Rejecting a negative reward is left to
  the concrete strategies, which is where a real guard can be exercised.
- ``snapshot_state() -> dict[str, object]`` is pure -- it neither mutates the
  strategy nor depends on anything but its current state -- and returns a FRESH
  dictionary on every call, so a dynamics log can never write back into the
  strategy. The key ``probabilities`` (``{operator_id: p_i}``) is guaranteed for
  every strategy: any selection rule induces a distribution over the arms, degenerate
  (a point mass on the argmax) for a deterministic one. Strategy-specific fields --
  quality estimates, index values, usage counts, reset counters -- are added on top
  by each strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from aos_ga.aos.strategy import AosStrategy

# --- Test doubles ---------------------------------------------------------------


class _WeightedStrategy(AosStrategy):
    """Stochastic double: draws an arm from fixed, non-uniform selection weights.

    Stands in for a learning strategy without learning anything -- the weights never
    move, so every assertion about ``select_operator`` and ``snapshot_state`` reads
    the interface rather than some adaptation rule. It records the updates it was
    handed so the reward path can be pinned, and rejects an unknown arm.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = dict(weights)
        self.updates: list[tuple[str, float]] = []

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return tuple(self._weights)

    def select_operator(self, rng: Generator) -> str:
        threshold = float(rng.random())
        cumulative = 0.0
        for operator_id, probability in self._weights.items():
            cumulative += probability
            if threshold < cumulative:
                return operator_id
        return self.operator_ids[-1]

    def update(self, operator_id: str, reward: float) -> None:
        if operator_id not in self._weights:
            raise KeyError(operator_id)
        self.updates.append((operator_id, reward))

    def snapshot_state(self) -> dict[str, object]:
        return {"probabilities": dict(self._weights), "update_count": len(self.updates)}


class _ArgmaxStrategy(AosStrategy):
    """Deterministic double: always the highest-scoring arm, ``rng`` unused.

    The shape UCB and DMAB will take -- selection is an argmax over per-arm indices,
    so the induced distribution is a point mass. Included to pin that a deterministic
    rule satisfies the same contract, ``probabilities`` entry included.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = dict(scores)

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return tuple(self._scores)

    def select_operator(self, rng: Generator) -> str:
        return max(self._scores, key=lambda operator_id: self._scores[operator_id])

    def update(self, operator_id: str, reward: float) -> None:
        if operator_id not in self._scores:
            raise KeyError(operator_id)
        self._scores[operator_id] = reward

    def snapshot_state(self) -> dict[str, object]:
        best = max(self._scores, key=lambda operator_id: self._scores[operator_id])
        return {
            "probabilities": {arm: float(arm == best) for arm in self._scores},
            "index_values": dict(self._scores),
        }


class _UniformStrategy(AosStrategy):
    """Non-learning double: uniform draw, ``update`` a no-op -- Random on the seam.

    Documents that the lower reference point is expressible on this interface as the
    degenerate case ``p_i = 1/K`` with nothing to learn. Whether the standalone
    ``RandomOperatorStep`` is later rewired onto the strategy seam is a separate
    decision; what matters here is that the interface does not exclude it.
    """

    def __init__(self, operator_ids: tuple[str, ...]) -> None:
        self._operator_ids = operator_ids

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return self._operator_ids

    def select_operator(self, rng: Generator) -> str:
        return self._operator_ids[int(rng.integers(len(self._operator_ids)))]

    def update(self, operator_id: str, reward: float) -> None:
        return None

    def snapshot_state(self) -> dict[str, object]:
        uniform = 1.0 / len(self._operator_ids)
        return {"probabilities": dict.fromkeys(self._operator_ids, uniform)}


_ARMS = ("ox", "inversion", "pmx")


def _weighted() -> _WeightedStrategy:
    return _WeightedStrategy({"ox": 0.5, "inversion": 0.3, "pmx": 0.2})


def _probabilities(strategy: AosStrategy) -> dict[str, float]:
    """Read the guaranteed ``probabilities`` entry out of a snapshot, typed."""
    probabilities = strategy.snapshot_state()["probabilities"]
    assert isinstance(probabilities, dict)
    return {str(arm): float(value) for arm, value in probabilities.items()}


# --- abstractness and shape of the interface -------------------------------------


def test_aos_strategy_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        AosStrategy()  # type: ignore[abstract]


def test_all_four_members_are_abstract() -> None:
    # The whole interface, nothing more: an arm list, a selection, a reward sink and
    # a log-ready snapshot.
    assert AosStrategy.__abstractmethods__ == frozenset(
        {"operator_ids", "select_operator", "update", "snapshot_state"}
    )


def test_subclass_missing_a_member_is_abstract() -> None:
    class _NoSnapshot(AosStrategy):
        @property
        def operator_ids(self) -> tuple[str, ...]:
            return _ARMS

        def select_operator(self, rng: Generator) -> str:
            return _ARMS[0]

        def update(self, operator_id: str, reward: float) -> None:
            return None

    with pytest.raises(TypeError):
        _NoSnapshot()  # type: ignore[abstract]


def test_aos_strategy_is_not_generic_over_the_genome_type() -> None:
    # Deliberate contrast with ``VariationStep[Genome]``: a strategy trades only in
    # operator ids and rewards, so it is independent of the problem class.
    with pytest.raises(TypeError):
        AosStrategy[list[int]]  # type: ignore[misc]


# --- operator_ids: the arms, in pool order, stable -------------------------------


def test_operator_ids_lists_the_arms_in_pool_order() -> None:
    assert _weighted().operator_ids == _ARMS


def test_operator_ids_is_an_immutable_tuple_of_strings() -> None:
    operator_ids = _weighted().operator_ids
    assert isinstance(operator_ids, tuple)
    assert all(isinstance(arm, str) for arm in operator_ids)


def test_operator_ids_is_stable_across_updates() -> None:
    # The arm set is fixed at construction; learning moves the estimates, never the
    # membership -- this is what lets the variation step validate the pool once.
    strategy = _weighted()
    before = strategy.operator_ids
    strategy.update("ox", 1.5)
    assert strategy.operator_ids == before


# --- select_operator: an arm, drawn from the injected generator ------------------


def test_select_operator_returns_an_id_from_operator_ids() -> None:
    strategy = _weighted()
    rng = np.random.default_rng(0)
    assert all(strategy.select_operator(rng) in _ARMS for _ in range(200))


def _selection_sequence(seed: int, draws: int = 20) -> list[str]:
    """Replay ``draws`` selections of a fresh strategy from one seeded generator."""
    rng = np.random.default_rng(seed)
    strategy = _weighted()
    return [strategy.select_operator(rng) for _ in range(draws)]


@pytest.mark.parametrize("seed", range(8))
def test_select_operator_is_reproducible_from_the_seed(seed: int) -> None:
    # Every draw comes from the injected generator, so one seed replays the whole
    # selection sequence -- the run stays reproducible end to end.
    assert _selection_sequence(seed) == _selection_sequence(seed)


def test_select_operator_sequences_diverge_across_seeds() -> None:
    # The draw really consumes the generator: a different seed gives a different
    # sequence, so the selection is not silently constant.
    assert _selection_sequence(1) != _selection_sequence(2)


def test_select_operator_leaves_the_learning_state_untouched() -> None:
    # Counters (the AOS step ``t``, per-arm usage ``n_i``) advance in ``update``, not
    # in the draw; the skeleton's strict produce-evaluate-observe alternation
    # guarantees one update per selection, so nothing is lost by the split.
    strategy = _weighted()
    before = strategy.snapshot_state()
    strategy.select_operator(np.random.default_rng(0))
    assert strategy.snapshot_state() == before


def test_a_deterministic_strategy_satisfies_the_contract() -> None:
    # Ignoring ``rng`` is legal: UCB and DMAB select by argmax over their indices.
    strategy = _ArgmaxStrategy({"ox": 0.1, "inversion": 0.9, "pmx": 0.4})
    assert {strategy.select_operator(np.random.default_rng(seed)) for seed in range(8)} == {
        "inversion"
    }


# --- update: the reward sink -----------------------------------------------------


def test_update_is_a_command_not_a_query() -> None:
    # Nothing comes back from an update: the effect of a reward is read out through
    # ``snapshot_state``, which keeps the strategy the single source of its state.
    strategy = _weighted()
    strategy.update("ox", 2.0)
    assert strategy.snapshot_state()["update_count"] == 1


def test_update_carries_the_reward_to_the_named_arm() -> None:
    strategy = _weighted()
    strategy.update("inversion", 0.75)
    assert strategy.updates == [("inversion", 0.75)]


def test_zero_reward_is_an_ordinary_update() -> None:
    # Instant reward is ``max(0, g(child) - g_ref)``, so zeros are frequent and are
    # data, not a degenerate case to be smoothed away with an artificial epsilon.
    strategy = _weighted()
    strategy.update("ox", 0.0)
    assert strategy.updates == [("ox", 0.0)]


def test_update_rejects_an_unknown_operator_id() -> None:
    # An id outside the arm set means the strategy and the pool disagree; that is a
    # wiring bug and must surface immediately rather than be absorbed.
    with pytest.raises(KeyError):
        _weighted().update("not-in-the-pool", 1.0)


# --- snapshot_state: pure, fresh, and a distribution over the arms ---------------


def test_snapshot_state_exposes_a_probability_for_every_arm() -> None:
    assert set(_probabilities(_weighted())) == set(_ARMS)


@pytest.mark.parametrize(
    "strategy",
    [
        _weighted(),
        _ArgmaxStrategy({"ox": 0.1, "inversion": 0.9, "pmx": 0.4}),
        _UniformStrategy(_ARMS),
    ],
)
def test_snapshot_probabilities_sum_to_one(strategy: AosStrategy) -> None:
    # Guaranteed for every strategy: a selection rule always induces a distribution,
    # a point mass in the deterministic case.
    assert sum(_probabilities(strategy).values()) == pytest.approx(1.0)


def test_a_deterministic_strategy_reports_a_point_mass() -> None:
    strategy = _ArgmaxStrategy({"ox": 0.1, "inversion": 0.9, "pmx": 0.4})
    assert _probabilities(strategy) == {"ox": 0.0, "inversion": 1.0, "pmx": 0.0}


def test_a_non_learning_strategy_reports_uniform_probabilities() -> None:
    # Random selection is the degenerate member of this interface: ``p_i = 1/K`` with
    # an ``update`` that has nothing to learn.
    assert _probabilities(_UniformStrategy(_ARMS)) == pytest.approx(
        dict.fromkeys(_ARMS, 1.0 / len(_ARMS))
    )


def test_snapshot_state_is_pure() -> None:
    strategy = _weighted()
    assert strategy.snapshot_state() == strategy.snapshot_state()


def test_snapshot_state_returns_a_fresh_dictionary_each_call() -> None:
    strategy = _weighted()
    first, second = strategy.snapshot_state(), strategy.snapshot_state()
    assert first is not second
    assert first["probabilities"] is not second["probabilities"]


def test_mutating_a_snapshot_cannot_reach_the_strategy() -> None:
    # The dynamics log receives a copy: whatever it does to the dictionary, the
    # strategy keeps selecting from its own state.
    strategy = _weighted()
    snapshot = strategy.snapshot_state()
    probabilities = snapshot["probabilities"]
    assert isinstance(probabilities, dict)
    probabilities["ox"] = 99.0
    snapshot["probabilities"] = "corrupted"
    assert _probabilities(strategy)["ox"] == pytest.approx(0.5)


def test_snapshot_state_may_carry_strategy_specific_fields() -> None:
    # ``probabilities`` is the common denominator, not the whole payload: PM adds its
    # quality estimates, UCB and DMAB their index values and usage counts.
    assert "index_values" in _ArgmaxStrategy({"ox": 1.0, "inversion": 0.0}).snapshot_state()
    assert "update_count" in _weighted().snapshot_state()
