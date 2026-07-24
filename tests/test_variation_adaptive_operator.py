"""Contract spec for the adaptive-operator variation step.

The third and last shape of the variation layer: ``SingleOperatorStep`` applies one
fixed operator (the FBO ceiling), ``RandomOperatorStep`` draws uniformly from the
pool (the Random floor), and ``AdaptiveOperatorStep`` lets a strategy choose -- and
pays it back with credit. It is the piece that turns the frozen seam into an AOS
loop without touching either side of it: the engine still produces one child per
call and observes its quality afterwards, and the strategy still trades only in
operator ids and rewards.

Everything the two halves need to meet lives in this step. ``produce`` asks the
strategy for an arm, records which arm it applied and the reference quality
``g_ref`` of the parents it drew; ``observe`` turns the child's quality into an
instant reward against that ``g_ref`` and hands it to the strategy. That pending
pair is the whole reason this class exists: ``observe(child_quality)`` alone cannot
tell which operator earned the outcome.

Nothing is implemented yet: this file is the executable specification. Expected
public name (in ``aos_ga.variation.adaptive_operator``): ``AdaptiveOperatorStep``, a
subclass of ``aos_ga.core.variation.VariationStep``.

Frozen contract (adaptive-operator step):
- ``AdaptiveOperatorStep(pool, strategy)`` indexes the pool by ``operator_id`` and
  checks up front that the pool and ``strategy.operator_ids`` name the same arm set
  -- order may differ, membership may not. A mismatch, an empty pool or two
  operators sharing an id all raise ``ValueError``: credit routed to the wrong arm
  would corrupt a whole sweep silently.
- ``produce(select_parent, rng)`` draws in a fixed order: the strategy selects the
  arm FIRST (the same shape as the Random baseline, so the operator is the first
  random decision of the event), then exactly ``operator.arity`` parents are drawn,
  then ``g_ref = max(parent quality)`` is recorded together with the arm, and finally
  the operator's child is returned as-is -- the step adds no copy of its own, since
  the frozen ``Operator.apply`` already returns a fresh, non-aliased child.
- ``observe(child_quality)`` calls ``strategy.update`` exactly once, with the arm
  ``produce`` actually applied and the instant reward ``max(0, child_quality -
  g_ref)``, then clears the pending pair.
- Calling ``observe`` with nothing pending, or ``produce`` twice without an
  intervening ``observe``, raises ``RuntimeError``. The engine alternates strictly,
  so either case means the step is being driven outside its contract -- and silently
  overwriting the pending pair would drop one event's credit without a trace.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.random import Generator

from aos_ga.aos.probability_matching import ProbabilityMatching
from aos_ga.aos.strategy import AosStrategy
from aos_ga.core.engine import run
from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation
from aos_ga.core.variation import Parent, VariationStep
from aos_ga.operators.permutation import OrderCrossover, SegmentInversion
from aos_ga.variation.adaptive_operator import AdaptiveOperatorStep

# --- Test doubles ----------------------------------------------------------------


class _MarkerMutation(Operator[list[int]]):
    """Arity-1 double that emits a fixed marker naming the arm that fired."""

    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, operator_id: str, marker: list[int] | None = None) -> None:
        self.operator_id = operator_id
        self._marker = list(marker) if marker is not None else [1]

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        if len(parents) != self.arity:
            raise ValueError(f"expected {self.arity} parents, got {len(parents)}")
        return list(self._marker)


class _MarkerCrossover(Operator[list[int]]):
    """Arity-2 counterpart of :class:`_MarkerMutation`."""

    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def __init__(self, operator_id: str, marker: list[int] | None = None) -> None:
        self.operator_id = operator_id
        self._marker = list(marker) if marker is not None else [2]

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        if len(parents) != self.arity:
            raise ValueError(f"expected {self.arity} parents, got {len(parents)}")
        return list(self._marker)


class _SentinelOperator(Operator[list[int]]):
    """Arity-1 double returning the SAME child object every call.

    A probe for the no-extra-copy rule: if ``produce`` hands back this exact object,
    the step passed the operator's child straight through.
    """

    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, operator_id: str = "sentinel") -> None:
        self.operator_id = operator_id
        self.child: list[int] = [42]

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        return self.child


class _RecordingStrategy(AosStrategy):
    """Strategy double: selects from a script and records every call it received.

    Learning is irrelevant here -- what matters is which arm it named, what reward
    came back and in what order the step interleaved the two with its parent draws.
    """

    def __init__(
        self,
        operator_ids: Sequence[str],
        *,
        script: Sequence[str] | None = None,
        order_log: list[str] | None = None,
    ) -> None:
        self._operator_ids = tuple(operator_ids)
        self._script = tuple(script) if script else self._operator_ids[:1]
        self._order_log = order_log
        self.selections: list[str] = []
        self.updates: list[tuple[str, float]] = []

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return self._operator_ids

    def select_operator(self, rng: Generator) -> str:
        if self._order_log is not None:
            self._order_log.append("select")
        chosen = self._script[len(self.selections) % len(self._script)]
        self.selections.append(chosen)
        return chosen

    def update(self, operator_id: str, reward: float) -> None:
        self.updates.append((operator_id, reward))

    def snapshot_state(self) -> dict[str, object]:
        uniform = 1.0 / len(self._operator_ids)
        return {"probabilities": dict.fromkeys(self._operator_ids, uniform)}


class _ParentSource:
    """A ``select_parent`` double yielding scripted (genome, quality) pairs, cycling."""

    def __init__(
        self,
        parents: Sequence[tuple[list[int], float]],
        *,
        order_log: list[str] | None = None,
    ) -> None:
        self._parents = list(parents)
        self._order_log = order_log
        self.calls = 0

    def __call__(self) -> Parent[list[int]]:
        genome, quality = self._parents[self.calls % len(self._parents)]
        if self._order_log is not None:
            self._order_log.append("parent")
        parent = Parent(index=self.calls, genome=genome, quality=quality)
        self.calls += 1
        return parent


class _PermutationSortProblem(Problem[list[int]]):
    """Permutation problem double: minimize displacement from the identity tour.

    ``f = sum |g[i] - i|`` over a permutation of ``0..n-1``, a minimization problem
    with a known optimum, so the adaptive step runs end-to-end on the real skeleton
    with real permutation operators and a real strategy.
    """

    direction = Direction.MINIMIZE
    representation = Representation.PERMUTATION

    def __init__(self, dimension: int = 8) -> None:
        self.name = "permutation-sort-double"
        self._dimension = dimension
        self.eval_count = 0

    def evaluate(self, individual: list[int]) -> float:
        self.eval_count += 1
        return float(sum(abs(city - position) for position, city in enumerate(individual)))

    def initialize(self, rng: Generator) -> list[int]:
        return [int(city) for city in rng.permutation(self._dimension)]


_FIRST = [0, 1, 2]
_SECOND = [3, 4, 5]


def _mixed_pool() -> list[Operator[list[int]]]:
    """A two-arm pool of mixed arity: a unary and a binary marker operator."""
    return [_MarkerMutation("mut", [1]), _MarkerCrossover("cross", [2])]


def _real_pool() -> list[Operator[list[int]]]:
    return [OrderCrossover(), SegmentInversion()]


def _step(
    pool: list[Operator[list[int]]] | None = None,
    *,
    script: Sequence[str] | None = None,
    order_log: list[str] | None = None,
) -> tuple[AdaptiveOperatorStep[list[int]], _RecordingStrategy]:
    operators = pool if pool is not None else _mixed_pool()
    strategy = _RecordingStrategy(
        [operator.operator_id for operator in operators], script=script, order_log=order_log
    )
    return AdaptiveOperatorStep(operators, strategy), strategy


# --- construction: pool and strategy must name the same arms ---------------------


def test_adaptive_operator_step_is_a_variation_step() -> None:
    step, _ = _step()
    assert isinstance(step, VariationStep)


def test_arm_order_may_differ_between_pool_and_strategy() -> None:
    # Membership is the contract, order is not: the strategy iterates its own order.
    pool = _mixed_pool()
    AdaptiveOperatorStep(pool, _RecordingStrategy(("cross", "mut")))  # no raise


def test_rejects_an_empty_pool() -> None:
    with pytest.raises(ValueError):
        AdaptiveOperatorStep([], _RecordingStrategy(("mut",)))


def test_rejects_a_strategy_with_an_arm_missing_from_the_pool() -> None:
    # The strategy could name an operator that does not exist -- a KeyError waiting to
    # happen thousands of runs into a sweep.
    with pytest.raises(ValueError):
        AdaptiveOperatorStep(_mixed_pool(), _RecordingStrategy(("mut", "cross", "insert")))


def test_rejects_a_pool_operator_the_strategy_cannot_select() -> None:
    # The mirror case: an arm that can never be chosen and never earns credit.
    with pytest.raises(ValueError):
        AdaptiveOperatorStep(_mixed_pool(), _RecordingStrategy(("mut",)))


def test_rejects_two_pool_operators_sharing_an_id() -> None:
    # Credit would be routed to whichever instance the index happened to keep.
    pool: list[Operator[list[int]]] = [_MarkerMutation("mut", [1]), _MarkerMutation("mut", [9])]
    with pytest.raises(ValueError):
        AdaptiveOperatorStep(pool, _RecordingStrategy(("mut",)))


# --- produce: the strategy chooses first, then the parents are drawn -------------


def test_the_applied_operator_is_the_one_the_strategy_named() -> None:
    step, _ = _step(script=["cross"])
    child = step.produce(_ParentSource([(_FIRST, 1.0), (_SECOND, 2.0)]), np.random.default_rng(0))
    assert child == [2]  # the binary marker


def test_selection_precedes_every_parent_draw() -> None:
    # The arm is the first decision of the event, exactly as in the Random baseline:
    # the parents drawn depend on the operator's arity, never the other way round.
    order_log: list[str] = []
    step, _ = _step(script=["cross"], order_log=order_log)
    step.produce(
        _ParentSource([(_FIRST, 1.0), (_SECOND, 2.0)], order_log=order_log),
        np.random.default_rng(0),
    )
    assert order_log == ["select", "parent", "parent"]


@pytest.mark.parametrize(("arm", "expected_parents"), [("mut", 1), ("cross", 2)])
def test_parent_draws_match_the_selected_operators_arity(arm: str, expected_parents: int) -> None:
    step, _ = _step(script=[arm])
    source = _ParentSource([(_FIRST, 1.0), (_SECOND, 2.0)])
    step.produce(source, np.random.default_rng(0))
    assert source.calls == expected_parents


def test_returns_the_operators_child_without_an_extra_copy() -> None:
    sentinel = _SentinelOperator()
    pool: list[Operator[list[int]]] = [sentinel]
    step = AdaptiveOperatorStep(pool, _RecordingStrategy(("sentinel",)))
    child = step.produce(_ParentSource([(_FIRST, 1.0)]), np.random.default_rng(0))
    assert child is sentinel.child


@pytest.mark.parametrize("seed", range(8))
def test_parents_are_never_mutated_by_the_step(seed: int) -> None:
    first, second = [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]
    snapshot = [list(first), list(second)]
    pool = _real_pool()
    step = AdaptiveOperatorStep(pool, _RecordingStrategy(("ox", "inversion"), script=["ox"]))
    step.produce(_ParentSource([(first, 1.0), (second, 2.0)]), np.random.default_rng(seed))
    assert [first, second] == snapshot


@pytest.mark.parametrize("seed", range(8))
def test_child_is_never_an_aliased_parent(seed: int) -> None:
    first, second = [0, 1, 2, 3], [3, 2, 1, 0]
    step = AdaptiveOperatorStep(
        _real_pool(), _RecordingStrategy(("ox", "inversion"), script=["ox"])
    )
    child = step.produce(_ParentSource([(first, 1.0), (second, 2.0)]), np.random.default_rng(seed))
    assert child is not first
    assert child is not second


# --- observe: instant reward against the parents' g_ref --------------------------


def test_reward_is_the_improvement_over_the_better_parent() -> None:
    # g_ref = max(2.0, 7.0) = 7.0, so a child at 9.0 earns 2.0 -- the weaker parent
    # never sets the bar for a crossover.
    step, strategy = _step(script=["cross"])
    step.produce(_ParentSource([(_FIRST, 2.0), (_SECOND, 7.0)]), np.random.default_rng(0))
    step.observe(9.0)
    assert strategy.updates == [("cross", 2.0)]


def test_reward_uses_the_sole_parent_for_a_unary_operator() -> None:
    step, strategy = _step(script=["mut"])
    step.produce(_ParentSource([(_FIRST, 4.0)]), np.random.default_rng(0))
    step.observe(4.5)
    assert len(strategy.updates) == 1
    operator_id, reward = strategy.updates[0]
    assert operator_id == "mut"
    assert reward == pytest.approx(0.5)


def test_a_child_worse_than_its_parents_earns_zero() -> None:
    step, strategy = _step(script=["cross"])
    step.produce(_ParentSource([(_FIRST, 2.0), (_SECOND, 7.0)]), np.random.default_rng(0))
    step.observe(3.0)
    assert strategy.updates == [("cross", 0.0)]


def test_parent_order_does_not_change_the_reference() -> None:
    step, strategy = _step(script=["cross"])
    step.produce(_ParentSource([(_SECOND, 7.0), (_FIRST, 2.0)]), np.random.default_rng(0))
    step.observe(9.0)
    assert strategy.updates == [("cross", 2.0)]


def test_credit_goes_to_the_arm_that_was_applied_not_the_next_one() -> None:
    # The script hands out a different arm on the second selection; the reward for the
    # first child must still name the first arm. This is what the pending pair is for.
    step, strategy = _step(script=["cross", "mut"])
    step.produce(_ParentSource([(_FIRST, 1.0), (_SECOND, 3.0)]), np.random.default_rng(0))
    step.observe(5.0)
    step.produce(_ParentSource([(_FIRST, 1.0)]), np.random.default_rng(0))
    step.observe(1.5)
    assert strategy.updates == [("cross", 2.0), ("mut", 0.5)]


def test_update_is_called_exactly_once_per_child() -> None:
    step, strategy = _step(script=["mut"])
    for _ in range(5):
        step.produce(_ParentSource([(_FIRST, 1.0)]), np.random.default_rng(0))
        step.observe(2.0)
    assert len(strategy.updates) == 5
    assert len(strategy.selections) == 5


# --- driving the step outside its contract ---------------------------------------


def test_observe_without_a_pending_child_is_an_error() -> None:
    step, _ = _step()
    with pytest.raises(RuntimeError):
        step.observe(1.0)


def test_observing_twice_for_one_child_is_an_error() -> None:
    step, _ = _step(script=["mut"])
    step.produce(_ParentSource([(_FIRST, 1.0)]), np.random.default_rng(0))
    step.observe(2.0)
    with pytest.raises(RuntimeError):
        step.observe(2.0)


def test_producing_twice_without_observing_is_an_error() -> None:
    # Overwriting the pending pair would drop one event's credit with no trace.
    step, _ = _step(script=["mut"])
    source = _ParentSource([(_FIRST, 1.0)])
    step.produce(source, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        step.produce(source, np.random.default_rng(0))


def test_the_step_is_reusable_after_a_completed_event() -> None:
    step, strategy = _step(script=["mut"])
    step.produce(_ParentSource([(_FIRST, 1.0)]), np.random.default_rng(0))
    step.observe(2.0)
    step.produce(_ParentSource([(_FIRST, 1.0)]), np.random.default_rng(0))  # no raise
    step.observe(3.0)
    assert strategy.updates == [("mut", 1.0), ("mut", 2.0)]


# --- integration on the skeleton --------------------------------------------------


def test_run_credits_the_strategy_once_per_reproduction_event() -> None:
    # The invariant the whole strategy contract rests on: the engine alternates
    # produce and observe strictly, so a counter kept in ``update`` is a faithful
    # count of the AOS steps taken.
    problem = _PermutationSortProblem(dimension=8)
    strategy = _RecordingStrategy(("ox", "inversion"), script=["ox", "inversion"])
    step = AdaptiveOperatorStep(_real_pool(), strategy)

    result = run(problem, step, np.random.default_rng(0), population_size=12, generations=10)

    assert len(strategy.updates) == result.reproduction_events
    assert len(strategy.selections) == result.reproduction_events
    assert all(reward >= 0.0 for _, reward in strategy.updates)


def test_run_with_probability_matching_improves_and_stays_legal() -> None:
    problem = _PermutationSortProblem(dimension=8)
    strategy = ProbabilityMatching(("ox", "inversion"))
    step = AdaptiveOperatorStep(_real_pool(), strategy)

    result = run(problem, step, np.random.default_rng(0), population_size=12, generations=10)

    history = result.best_quality_history
    assert history[-1] > history[0]  # the run made progress
    assert all(earlier <= later for earlier, later in zip(history, history[1:], strict=False))
    assert sorted(result.best) == list(range(8))  # elitism kept a legal permutation
    assert problem.eval_count == result.evaluations  # one evaluation per child
    probabilities = strategy.snapshot_state()["probabilities"]
    assert isinstance(probabilities, dict)
    assert sum(float(value) for value in probabilities.values()) == pytest.approx(1.0)


def test_run_is_reproducible_with_an_adaptive_step() -> None:
    def run_once() -> object:
        return run(
            _PermutationSortProblem(dimension=8),
            AdaptiveOperatorStep(_real_pool(), ProbabilityMatching(("ox", "inversion"))),
            np.random.default_rng(3),
            population_size=10,
            generations=8,
        )

    assert run_once() == run_once()


def test_the_strategy_actually_learns_during_a_run() -> None:
    # Not a claim about which operator wins -- only that credit moved the estimates
    # off their uniform start, i.e. the observe -> update path is wired end to end.
    strategy = ProbabilityMatching(("ox", "inversion"))
    step = AdaptiveOperatorStep(_real_pool(), strategy)

    run(
        _PermutationSortProblem(dimension=8),
        step,
        np.random.default_rng(0),
        population_size=12,
        generations=10,
    )

    estimates = strategy.snapshot_state()["quality_estimates"]
    assert isinstance(estimates, dict)
    assert len(set(estimates.values())) > 1
