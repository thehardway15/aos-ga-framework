"""Shared interface for the adaptive operator-selection strategies.

A strategy answers exactly one question -- which operator to apply next -- and
consumes exactly one signal -- the non-negative reward that the credit-assignment
module derived from the resulting child. Probability Matching, Adaptive Pursuit,
UCB and DMAB all implement this single contract, and Random selection is its
degenerate member (uniform probabilities, nothing to learn). A strategy trades only
in ``operator_id`` and ``reward``: it never sees a genome, a population or a
problem, which is what lets one implementation run over a permutation, binary or
real-valued pool without a single branch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.random import Generator


class AosStrategy(ABC):
    """Selects the next operator and learns from the reward that choice earned.

    Deliberately NOT generic over the genome type -- the contrast with
    :class:`~aos_ga.core.variation.VariationStep` is the point: an operator id and a
    float are the whole vocabulary, so independence from the problem class is
    structural rather than a matter of discipline. Pairing a strategy with a concrete
    pool is the variation step's job: it reads ``operator_ids`` to check that the two
    name the same arms, calls :meth:`select_operator` to pick one, and calls
    :meth:`update` once the resulting child has been evaluated.
    """

    @property
    @abstractmethod
    def operator_ids(self) -> tuple[str, ...]:
        """The arms this strategy selects among, in pool order.

        Fixed for the lifetime of the strategy: learning moves the estimates, never
        the membership. Exposing it lets the variation step verify at construction
        time that the strategy and the injected pool agree, instead of failing deep
        inside a sweep on an operator id that was never an arm.
        """

    @abstractmethod
    def select_operator(self, rng: Generator) -> str:
        """Return the ``operator_id`` of the arm to apply next.

        Draws all randomness from the injected ``rng``, so one seed replays the whole
        selection sequence; a deterministic rule (an argmax over per-arm indices) is
        equally legal and simply ignores it. The call does NOT advance the learning
        statistics -- the AOS step counter and the per-arm usage counts belong to
        :meth:`update`. The split is safe because the skeleton alternates strictly:
        it produces a child, evaluates it and only then observes it, so every
        selection is followed by exactly one update.
        """

    @abstractmethod
    def update(self, operator_id: str, reward: float) -> None:
        """Credit ``operator_id`` with ``reward`` -- called once per selection.

        Invoked after the child that operator produced has been evaluated. The
        credit-assignment module guarantees ``reward >= 0``; zero is an ordinary
        value (there is no artificial epsilon) and a long run of zero rewards is
        data, not a degenerate case to smooth away. An id outside ``operator_ids``
        means the strategy and the pool disagree -- a wiring bug that must fail
        loudly with ``KeyError`` rather than be absorbed silently.
        """

    @abstractmethod
    def snapshot_state(self) -> dict[str, object]:
        """Return a log-ready view of the current selection state.

        Pure: it neither mutates the strategy nor depends on anything but its current
        state, and it returns a FRESH dictionary on every call, so a dynamics log can
        never write back into the strategy. The key ``probabilities``
        (``{operator_id: p_i}``) is guaranteed for every strategy, because any
        selection rule induces a distribution over the arms -- a degenerate one, a
        point mass on the argmax, for a deterministic rule. Strategy-specific fields
        (quality estimates, index values, usage counts, reset counters) are added on
        top by each implementation.
        """
