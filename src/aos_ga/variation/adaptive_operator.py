"""The adaptive-operator variation step: a strategy picks the arm, credit pays it back.

The third and last shape of the variation layer.
:class:`~aos_ga.variation.single_operator.SingleOperatorStep` applies one fixed
operator (the fixed-best-operator ceiling),
:class:`~aos_ga.variation.random_operator.RandomOperatorStep` draws uniformly from the
pool (the Random floor), and ``AdaptiveOperatorStep`` lets an
:class:`~aos_ga.aos.strategy.AosStrategy` choose -- then hands it the credit that
choice earned. It turns the frozen seam into a full adaptive-operator-selection loop
without touching either side of it: the engine still produces one child per call and
observes its quality afterwards, and the strategy still trades only in operator ids
and rewards, never seeing a genome or a problem.

Everything the two halves need in order to meet lives here. ``produce`` asks the
strategy for an arm and records both that arm and the reference quality ``g_ref`` of
the parents it drew; ``observe`` turns the child's quality into an instant reward
against that ``g_ref`` and passes it on. Carrying that pair between the two calls is
the whole reason this class exists -- ``observe(child_quality)`` alone cannot say
which operator earned the outcome, and the engine cannot say it either, because the
choice was made in here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from numpy.random import Generator

from ..aos.strategy import AosStrategy
from ..core.operator import Operator
from ..core.representation import Genome
from ..core.variation import Parent, VariationStep
from ..credit.instant import instant_reward, reference_quality


class AdaptiveOperatorStep(VariationStep[Genome]):
    """Applies the operator an AOS strategy selects, and credits it with the result.

    Generic over the genome type and blind to the representation: the pool is built
    elsewhere, so the same class carries a permutation, binary or real-valued pool.
    Exactly one reproduction event may be in flight at a time -- the step holds the
    selected arm and its ``g_ref`` between ``produce`` and ``observe``, which is only
    safe because the engine alternates the two strictly.
    """

    def __init__(self, pool: Sequence[Operator[Genome]], strategy: AosStrategy) -> None:
        """Index the pool by ``operator_id`` and check it against the strategy's arms.

        The pool and ``strategy.operator_ids`` must name the same set of arms; their
        order need not agree, since the two sides only ever meet through ids. All
        three rejections raise ``ValueError`` and all three protect the same thing --
        that credit reaches the operator that actually produced the child: an empty
        pool leaves nothing to select, two operators sharing an id would make the
        routing ambiguous, and a mismatched arm set means either a strategy naming an
        operator that does not exist or an operator that can never be chosen and never
        rewarded. Catching this here rather than mid-sweep matters: a silently
        misrouted reward corrupts a whole experiment without failing.
        """
        if not pool:
            raise ValueError("AdaptiveOperatorStep requires a non-empty pool of operators")

        operators = {operator.operator_id: operator for operator in pool}
        if len(operators) != len(pool):
            raise ValueError(
                f"pool operator ids must be unique, got {[op.operator_id for op in pool]}"
            )
        if set(operators) != set(strategy.operator_ids):
            # Rendered through ``str`` so a caller who passed operators where arm ids were
            # expected still gets this message, rather than a sorting error on top of it.
            raise ValueError(
                f"pool arms {sorted(map(str, operators))} do not match strategy arms "
                f"{sorted(map(str, strategy.operator_ids))}"
            )

        self.pool = pool
        self.strategy = strategy
        self._operators = operators
        self._pending: tuple[str, float] | None = None

    def produce(self, select_parent: Callable[[], Parent[Genome]], rng: Generator) -> Genome:
        """Build one unevaluated child with the operator the strategy selects.

        Draws in a fixed order: the strategy selects the arm first -- the same shape
        as the Random baseline, so the operator is the first random decision of the
        event and the whole step replays from a seed -- then exactly
        ``operator.arity`` parents are drawn through ``select_parent``, and the
        operator is applied to their genomes unconditionally (no ``p_c``/``p_m`` coin
        gates it). The reference quality ``g_ref`` comes from the parents' recorded
        ``quality``, so no parent is re-evaluated. The arm and ``g_ref`` are held for
        :meth:`observe`; the child is returned as the operator built it, since the
        frozen ``Operator.apply`` already yields a fresh, non-aliased genome.

        Raises ``RuntimeError`` if the previous child was never observed: the pending
        pair has one slot, and overwriting it would drop that child's credit silently.
        """
        if self._pending is not None:
            raise RuntimeError("produce() called before observe() for the previous child")

        operator_id = self.strategy.select_operator(rng)
        operator = self._operators[operator_id]
        parents = [select_parent() for _ in range(operator.arity)]
        self._pending = (operator_id, reference_quality([parent.quality for parent in parents]))
        return operator.apply([parent.genome for parent in parents], rng)

    def observe(self, child_quality: float) -> None:
        """Credit the operator that produced this child, then close the event.

        Converts the child's quality into an instant reward against the ``g_ref``
        recorded in :meth:`produce` and passes it to the strategy exactly once. The
        pending pair is cleared before the strategy is called, so a failure inside
        ``update`` surfaces on its own rather than leaving the step wedged and turning
        the next ``produce`` into a misleading second error.

        Raises ``RuntimeError`` when no child is pending -- there is no event this
        quality could belong to.
        """
        if self._pending is None:
            raise RuntimeError("observe() called before produce()")

        operator_id, g_ref = self._pending
        self._pending = None
        self.strategy.update(operator_id, instant_reward(child_quality, g_ref))
