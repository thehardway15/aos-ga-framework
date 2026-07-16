"""Exact 0/1 knapsack solver used to fix small reference optima offline.

Dynamic programming over item prefixes and capacity: for a small integer instance
it returns the exact optimal value and one optimal 0/1 selection. Run once,
deterministically, to pin the reference optima of the nine Pisinger knapsack
instances, which -- unlike the TSP instances -- ship without a published optimum.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def knapsack_dp(
    values: Sequence[int], weights: Sequence[int], capacity: int
) -> tuple[int, list[int]]:
    """Exact optimal 0/1 knapsack value and selection via dynamic programming.

    ``values`` and ``weights`` are equal-length non-negative integer sequences and
    ``capacity`` a non-negative bound. Returns ``(optimal_value, selection)``: the
    maximum total value of a subset whose weight stays within ``capacity``, and one
    optimal 0/1 selection (``selection[i] == 1`` iff item ``i`` is chosen, with
    ``len(selection) == len(values)``). O(n * capacity) time and memory; deterministic.
    On value ties the reconstruction leaves the item out, so the selection is canonical.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    n = len(values)
    W = capacity
    # dp[i, c] = best value achievable from the first i items within capacity c.
    dp = np.zeros((n + 1, W + 1), dtype=np.int64)
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        prev = dp[i - 1]
        # take[c] = value if item i-1 is chosen: shift prev right by w (reserve the
        # weight) and add v; capacities below w cannot fit it, so leave them at -1.
        take = np.full(W + 1, -1, dtype=np.int64)
        if w <= W:
            take[w:] = prev[: W + 1 - w] + v
        dp[i] = np.maximum(prev, take)  # per capacity: better of {skip, take}

    # Walk the layers back: a value change from dp[i-1] to dp[i] means item i-1 was
    # taken; spend its weight and continue. Equality (a tie) leaves the item out.
    selection = [0] * n
    c = W
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selection[i - 1] = 1
            c -= weights[i - 1]
    return int(dp[n][W]), selection
