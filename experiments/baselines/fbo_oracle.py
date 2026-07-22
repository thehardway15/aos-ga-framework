"""Derive the reference operator (``o*``) for one pool from per-operator medians.

Each operator is measured on its own (its median final quality over the repetition seeds);
this step turns that per-operator median table into the pool's reference operator set -- the
operators sharing the maximum median -- together with that shared median. It is the upper
reference point for the adaptive operator-selection strategies. The derivation is pure and
membership-driven: the same median table projects onto any pool by passing that pool's
membership, so a full-pool measurement yields both the full and the reduced reference set with
no extra runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorOracle:
    """The reference operator set for one pool and its shared quality.

    ``o_star`` holds the operators that tie for the best median, in membership order;
    ``o_star_median`` is the single maximum they share; ``o_star_count`` is their number.
    """

    o_star: tuple[str, ...]
    o_star_median: float

    @property
    def o_star_count(self) -> int:
        return len(self.o_star)


def derive_oracle(
    operator_medians: Mapping[str, float], pool_members: Sequence[str]
) -> OperatorOracle:
    """Return the operators in ``pool_members`` whose median is maximal, with that median.

    ``operator_medians`` maps each ``operator_id`` to its median quality ("more is better");
    ``pool_members`` gives the ids to consider, in the order the result preserves. Only these
    members count, so a superset table projects cleanly onto a subpool. Ties are kept in full --
    every operator equal to the maximum by exact ``==``, with no tolerance and no tie-break.
    Raises ``ValueError`` if ``pool_members`` is empty or names an operator with no median.
    """
    if not pool_members:
        raise ValueError("Pool members cannot be empty.")

    missing = [op for op in pool_members if op not in operator_medians]
    if missing:
        raise ValueError(f"Missing medians for operators: {missing}")

    m_max = max(operator_medians[op] for op in pool_members)
    o_star = tuple(op for op in pool_members if operator_medians[op] == m_max)

    return OperatorOracle(o_star=o_star, o_star_median=m_max)
