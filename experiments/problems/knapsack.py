"""Zero/one knapsack test problem (binary representation).

Wraps a Pisinger knapsack instance as a :class:`~aos_ga.core.problem.Problem`: a
genome is a vector of item-selection bits (``list[int]`` of 0/1) and its fitness is the
total value of the chosen items minus a big-M penalty for exceeding the capacity, a
maximization objective. Feasibility is enforced by the penalty rather than by repair,
so infeasible genomes stay in the search but score below the empty knapsack.
"""

import numpy as np
from numpy.random import Generator

from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation

from ..datasets.knapsack import KnapsackInstance


class KnapsackProblem(Problem[list[int]]):
    """A Pisinger knapsack instance as a maximization problem over 0/1 selection vectors.

    The item values, weights and the 50% capacity are read from the instance; the big-M
    penalty coefficient ``rho = sum(values) + 1`` is computed once, per instance. Because
    every weight is a positive integer, the smallest possible overflow is one unit, so
    ``rho`` guarantees that any infeasible solution scores below the empty knapsack
    (``f < 0``) regardless of the instance or its value--weight correlation.
    """

    direction = Direction.MAXIMIZE
    representation = Representation.BINARY

    def __init__(self, instance: KnapsackInstance):
        """Store metadata and precompute the big-M penalty ``rho = sum(values) + 1``."""
        self.name = instance.instance_id
        self.dimension = instance.n
        self.correlation_type = instance.correlation_type
        self.values = np.array(instance.values, dtype=np.int64)
        self.weights = np.array(instance.weights, dtype=np.int64)
        self.capacity = instance.capacity
        self.penalty = sum(instance.values) + 1

    def evaluate(self, solution: list[int]) -> float:
        """Total value of the selected items, less a big-M penalty for any overflow.

        The selection bits act as a mask, so the dot products ``values @ x`` and
        ``weights @ x`` are the total value and weight of the chosen items. Fitness is that
        value minus ``penalty * max(0, weight - capacity)``: a feasible selection (weight up
        to and including the capacity) scores its plain value, while an infeasible one is
        driven below zero by the penalty. Total and finite; never raises on infeasibility.
        """
        x = np.array(solution, dtype=np.int64)
        total_value = int(self.values @ x)
        total_weight = int(self.weights @ x)
        overflow = max(0, total_weight - self.capacity)
        return float(total_value - self.penalty * overflow)

    def initialize(self, rng: Generator) -> list[int]:
        """Sample a uniformly random bitstring, each item chosen with probability 1/2.

        Draws every bit independently from ``rng`` alone (no global state), so a fixed
        seed reproduces the individual. The draw does not enforce feasibility -- with the
        50% capacity rule about half of a random population starts infeasible and is
        ranked out by the big-M penalty.
        """
        return [int(b) for b in rng.integers(0, 2, size=self.dimension)]
