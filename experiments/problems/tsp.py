"""Travelling-salesman test problem (permutation representation).

Wraps a TSPLIB instance as a :class:`~aos_ga.core.problem.Problem`: it precomputes
the integer ``EUC_2D`` distance matrix and scores a tour by its closed-cycle length,
a minimization objective. Genomes are city permutations (``list[int]``).
"""

import numpy as np
from numpy.random import Generator

from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation

from ..datasets.tsplib import TSPInstance


class TSPProblem(Problem[list[int]]):
    """A TSPLIB instance as a minimization problem over city permutations.

    The n x n integer distance matrix is precomputed once from the instance
    coordinates; the fitness of a tour is the length of the closed cycle that visits
    every city exactly once and returns to the start.
    """

    direction = Direction.MINIMIZE
    representation = Representation.PERMUTATION

    def __init__(self, instance: TSPInstance):
        """Store metadata and precompute the ``EUC_2D`` distance matrix (nint rounding)."""
        self.name = instance.instance_id
        self.dimension = instance.dimension

        pts = np.array(instance.coordinates)
        diff = pts[:, None, :] - pts[None, :, :]
        self.distances = np.floor(np.sqrt((diff**2).sum(axis=2)) + 0.5).astype(np.int64)

    def evaluate(self, tour: list[int]) -> float:
        """Closed-cycle length of ``tour``: the sum of its edges plus the return to the start."""
        return float(
            sum(self.distances[tour[i], tour[i + 1]] for i in range(self.dimension - 1))
            + self.distances[tour[-1], tour[0]]
        )

    def initialize(self, rng: Generator) -> list[int]:
        """Sample a uniformly random tour using only ``rng`` (no global state)."""
        perm: list[int] = rng.permutation(self.dimension).tolist()
        return perm
