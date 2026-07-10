"""Exact Held-Karp TSP solver used to fix small reference optima offline.

Dynamic programming over city subsets: for a small symmetric distance matrix it
returns the exact optimal tour length and one optimal tour. Run once,
deterministically, to pin the reference optimum of the eil22 instance, which has
no published optimum in TSPLIB's symmetric-TSP section.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_LARGE = np.iinfo(np.int64).max // 2  # finite stand-in for "infinity"


def held_karp(distance: NDArray[np.int64]) -> tuple[int, list[int]]:
    """Exact optimal TSP tour via Held-Karp dynamic programming.

    ``distance`` is an n x n symmetric integer matrix with a zero diagonal.
    Returns ``(length, tour)``: the optimal closed-cycle length and one optimal
    tour, 0-indexed and starting at city 0 (``len(tour) == n``). Deterministic.
    """
    n = int(distance.shape[0])
    if n == 1:
        return 0, [0]

    full = (1 << n) - 1
    # dp[mask, j] = best path over subset `mask` ending at city j; parent tracks predecessors.
    dp = np.full((1 << n, n), _LARGE, dtype=np.int64)
    parent = np.full((1 << n, n), -1, dtype=np.int32)

    # Base case: the direct edge 0 -> j visits only {0, j} and ends at j.
    for j in range(1, n):
        mask = (1 << 0) | (1 << j)
        dp[mask, j] = distance[0, j]
        parent[mask, j] = 0

    # Grow every subset by one city. For a fixed mask, trans[j, k] is the cost of
    # reaching k with j as the previous city; the column minimum picks the best j.
    for mask in range(1 << n):
        if not (mask & 1):  # every valid subset contains the start city 0
            continue
        row = dp[mask]
        trans = row[:, None] + distance  # (n, n)
        candidate = trans.min(axis=0)  # best cost of reaching each k
        pred = trans.argmin(axis=0)  # via which previous city j
        for k in range(1, n):
            if mask & (1 << k):  # k already visited
                continue
            new_mask = mask | (1 << k)
            if candidate[k] < dp[new_mask, k]:
                dp[new_mask, k] = candidate[k]
                parent[new_mask, k] = pred[k]

    # Close the cycle: add the return edge j -> 0 and keep the best ending city.
    best_cost = _LARGE
    best_end = -1
    for j in range(1, n):
        cost = dp[full, j] + distance[j, 0]
        if cost < best_cost:
            best_cost = cost
            best_end = j

    # Walk predecessors back to 0, then reverse so the tour starts at city 0.
    tour = [best_end]
    mask = full
    j = best_end
    while j != 0:
        pj = parent[mask, j]
        tour.append(pj)
        mask ^= 1 << j
        j = pj
    tour.reverse()
    return int(best_cost), [int(c) for c in tour]
