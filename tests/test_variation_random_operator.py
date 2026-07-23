"""Contract spec for the random-operator variation step.

``RandomOperatorStep`` draws one operator uniformly at random from an injected pool
on every reproduction event and makes that single draw the entire variation: no
crossover-then-mutation pipeline and no application probabilities. It is the lower
reference point (Random selection, ``p_i = 1/K``) for the adaptive
operator-selection strategies, and the thinnest slice of the AOS layer -- the
pattern "a variation step selects an operator from a pool" onto which the full
strategies later collapse. Because it assigns no credit, ``observe`` stays the
inherited no-op: it needs neither an ``update`` nor a dynamics snapshot. It is
generic over the genome type and reuses the existing ``Operator`` implementations by
composition, so the same class carries a permutation, binary or real-valued pool
without hardcoding a representation.

The class is not implemented yet: this file is the executable specification.
Expected public name (in ``aos_ga.variation.random_operator``):
``RandomOperatorStep``, a subclass of ``aos_ga.core.variation.VariationStep``.

Frozen contract (random-operator step):
- ``RandomOperatorStep(pool)`` stores a pool of one or more operators
  (``Sequence[Operator[Genome]]``). An empty pool is rejected with ``ValueError``:
  there is no arm to draw. A single-operator pool is legal and degenerates to
  always applying that one operator. Operators of mixed arity (a unary and a binary
  one) may share a pool; the step never validates the pool's representation --
  homogeneity is the pool builder's guarantee.
- ``produce(select_parent, rng) -> child`` builds exactly ONE unevaluated child:
    1. draw the operator index ``i = rng.integers(len(pool))`` uniformly -- this is
       the FIRST use of ``rng`` in the call, so the selection is reproducible from
       the seed and independent of the parents (decision: ``rng.integers``);
    2. draw exactly ``pool[i].arity`` parents by calling ``select_parent`` that many
       times (one for a unary operator, two for a binary one);
    3. apply the drawn operator to the parents' genomes -- unconditionally, there is
       no ``p_c``/``p_m`` coin -- and return its single child.
  All randomness is drawn from the single injected ``rng`` in this fixed order, so a
  fixed seed reproduces the child. The step adds no copy of its own: the frozen
  ``Operator.apply`` already returns a fresh, non-aliased child and never mutates its
  parents. Legalization against a problem's constraints stays the skeleton's job
  (``Problem.repair``); the step is given a ready-built pool and knows no bounds.
- ``observe`` is the inherited no-op: Random assigns no operator credit, so it is
  excluded from the adaptive reward machinery.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.random import Generator

from aos_ga.core.engine import run
from aos_ga.core.operator import Operator, OperatorKind
from aos_ga.core.problem import Direction, Problem
from aos_ga.core.representation import Representation
from aos_ga.core.variation import Parent, VariationStep
from aos_ga.operators.permutation import OrderCrossover, SegmentInversion
from aos_ga.variation.random_operator import RandomOperatorStep

# --- Test doubles --------------------------------------------------------------


class _MarkerMutation(Operator[list[int]]):
    """Arity-1 perturbation double: returns a fixed marker naming which arm fired.

    Ignores the parent's contents and emits its ``marker`` as the child, so a test
    can read the returned genome to tell exactly which pool member was drawn.
    Records every input it saw so its firing can be checked independently.
    """

    operator_id = "marker-mutation"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self, marker: list[int]) -> None:
        self._marker = list(marker)
        self.inputs: list[list[int]] = []

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        if len(parents) != self.arity:
            raise ValueError(f"expected {self.arity} parents, got {len(parents)}")
        self.inputs.append(list(parents[0]))
        return list(self._marker)


class _MarkerCrossover(Operator[list[int]]):
    """Arity-2 recombination double: returns a fixed marker naming which arm fired.

    The binary counterpart of :class:`_MarkerMutation`; records the genome pairs it
    saw so a test can pin that it drew two parents when it was the one selected.
    """

    operator_id = "marker-crossover"
    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def __init__(self, marker: list[int]) -> None:
        self._marker = list(marker)
        self.calls: list[list[list[int]]] = []

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        if len(parents) != self.arity:
            raise ValueError(f"expected {self.arity} parents, got {len(parents)}")
        self.calls.append([list(parent) for parent in parents])
        return list(self._marker)


class _SentinelOperator(Operator[list[int]]):
    """Arity-1 double whose ``apply`` returns the SAME child object on every call.

    A white-box probe for the no-extra-copy rule: if ``produce`` returns this exact
    object, the step passed the operator's child straight through; if it returns a
    different object, the step wrapped it in a copy of its own.
    """

    operator_id = "sentinel"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self) -> None:
        self.child: list[int] = [42]

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        return self.child


class _ParentSource:
    """A ``select_parent`` double: yields the given genomes in order, cycling.

    Wraps each genome in a ``Parent`` and counts calls, so a test can pin how many
    parents the step drew and check aliasing against the exact genome objects. Draws
    no randomness, so the step's first ``rng`` use is its operator-selection draw.
    """

    def __init__(self, genomes: list[list[int]]) -> None:
        self._genomes = genomes
        self.calls = 0

    def __call__(self) -> Parent[list[int]]:
        genome = self._genomes[self.calls % len(self._genomes)]
        parent = Parent(index=self.calls, genome=genome, quality=float(self.calls))
        self.calls += 1
        return parent


class _PermutationSortProblem(Problem[list[int]]):
    """Permutation problem double: minimize displacement from the identity tour.

    ``f = sum |g[i] - i|`` over a permutation of ``0..n-1`` -- a minimization problem
    with a known optimum (0 at the identity), so a real random-operator step runs
    end-to-end on the skeleton over a pool of permutation operators. ``evaluate``
    tallies calls so a test can pin the evaluation budget. OX and inversion keep
    offspring legal permutations, so the inherited identity ``repair`` suffices.
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


def _mixed_real_pool() -> list[Operator[list[int]]]:
    """A permutation pool of mixed arity: OX (binary) and inversion (unary)."""
    return [OrderCrossover(), SegmentInversion()]


_FIRST = [0, 1, 2]
_SECOND = [3, 4, 5]
_SOLO = [0, 1, 2]


# --- construction: pool size and mixed arity ------------------------------------


def test_random_operator_step_is_a_variation_step() -> None:
    assert isinstance(RandomOperatorStep([_MarkerMutation([0])]), VariationStep)


def test_accepts_a_single_element_pool() -> None:
    RandomOperatorStep([_MarkerMutation([0])])  # no raise


def test_accepts_a_mixed_arity_pool() -> None:
    # A unary and a binary operator may share a pool; the step never validates arity.
    pool: list[Operator[list[int]]] = [_MarkerMutation([0]), _MarkerCrossover([1])]
    RandomOperatorStep(pool)  # no raise


def test_empty_pool_is_rejected() -> None:
    with pytest.raises(ValueError):
        RandomOperatorStep([])


# --- selection: uniform, drawn first, reproducible ------------------------------


@pytest.mark.parametrize("pool_size", [2, 3, 5])
@pytest.mark.parametrize("seed", range(12))
def test_selected_operator_is_the_first_rng_draw(pool_size: int, seed: int) -> None:
    # The operator index is ``rng.integers(len(pool))`` drawn before anything else,
    # so it matches a parallel generator that only draws that one integer.
    pool: list[Operator[list[int]]] = [_MarkerMutation([index]) for index in range(pool_size)]
    step = RandomOperatorStep(pool)
    expected = int(np.random.default_rng(seed).integers(pool_size))

    child = step.produce(_ParentSource([list(_SOLO)]), np.random.default_rng(seed))

    assert child == [expected]


def test_every_operator_in_the_pool_is_reachable() -> None:
    pool_size = 4
    pool: list[Operator[list[int]]] = [_MarkerMutation([index]) for index in range(pool_size)]
    step = RandomOperatorStep(pool)
    rng = np.random.default_rng(0)
    source = _ParentSource([list(_SOLO)])

    selected = {step.produce(source, rng)[0] for _ in range(300)}

    assert selected == set(range(pool_size))


def test_selection_is_uniform_across_the_pool() -> None:
    # 1000 seeded draws of a two-arm pool: each arm within a generous band of 500.
    pool: list[Operator[list[int]]] = [_MarkerMutation([0]), _MarkerMutation([1])]
    step = RandomOperatorStep(pool)
    rng = np.random.default_rng(0)
    source = _ParentSource([list(_SOLO)])

    draws = [step.produce(source, rng)[0] for _ in range(1000)]

    assert 400 <= draws.count(0) <= 600


def test_single_element_pool_always_selects_that_operator() -> None:
    step = RandomOperatorStep([_MarkerMutation([9])])
    for seed in range(8):
        child = step.produce(_ParentSource([list(_SOLO)]), np.random.default_rng(seed))
        assert child == [9]


# --- produce: parent draws equal the DRAWN operator's arity ---------------------


@pytest.mark.parametrize("seed", range(16))
def test_parent_draw_count_matches_the_drawn_operator_arity(seed: int) -> None:
    # index 0 is unary (marker [0]), index 1 is binary (marker [1]); the number of
    # ``select_parent`` calls tracks whichever operator the draw selected.
    pool: list[Operator[list[int]]] = [_MarkerMutation([0]), _MarkerCrossover([1])]
    step = RandomOperatorStep(pool)
    source = _ParentSource([_FIRST, _SECOND])

    child = step.produce(source, np.random.default_rng(seed))

    expected_arity = 1 if child == [0] else 2
    assert source.calls == expected_arity


def test_draws_one_parent_for_a_unary_only_pool() -> None:
    step = RandomOperatorStep([_MarkerMutation([0])])
    source = _ParentSource([_FIRST, _SECOND])
    step.produce(source, np.random.default_rng(0))
    assert source.calls == 1


def test_draws_two_parents_for_a_binary_only_pool() -> None:
    step = RandomOperatorStep([_MarkerCrossover([0])])
    source = _ParentSource([_FIRST, _SECOND])
    step.produce(source, np.random.default_rng(0))
    assert source.calls == 2


# --- produce: no extra copy, freshness, parent integrity ------------------------


def test_returns_the_operators_child_without_an_extra_copy() -> None:
    operator = _SentinelOperator()
    step = RandomOperatorStep([operator])
    child = step.produce(_ParentSource([list(_FIRST)]), np.random.default_rng(0))
    assert child is operator.child  # passed straight through, the step adds no copy


@pytest.mark.parametrize("seed", range(8))
def test_child_is_never_an_aliased_parent(seed: int) -> None:
    first, second = [0, 1, 2, 3], [3, 2, 1, 0]
    step = RandomOperatorStep(_mixed_real_pool())
    child = step.produce(_ParentSource([first, second]), np.random.default_rng(seed))
    assert child is not first
    assert child is not second


@pytest.mark.parametrize("seed", range(8))
def test_parents_are_never_mutated_by_the_step(seed: int) -> None:
    first, second = [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]
    snapshot = [list(first), list(second)]
    step = RandomOperatorStep(_mixed_real_pool())
    step.produce(_ParentSource([first, second]), np.random.default_rng(seed))
    assert [first, second] == snapshot


# --- produce: permutation legality and determinism (real operators) -------------


@pytest.mark.parametrize("seed", range(32))
def test_child_stays_a_permutation_across_seeds(seed: int) -> None:
    genomes = [[0, 1, 2, 3, 4, 5], [3, 5, 1, 0, 4, 2]]  # same element multiset
    step = RandomOperatorStep(_mixed_real_pool())
    child = step.produce(
        _ParentSource([list(genomes[0]), list(genomes[1])]),
        np.random.default_rng(seed),
    )
    assert sorted(child) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("seed", range(16))
def test_produce_is_deterministic_for_a_fixed_seed(seed: int) -> None:
    genomes = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]
    step = RandomOperatorStep(_mixed_real_pool())
    first = step.produce(
        _ParentSource([list(genomes[0]), list(genomes[1])]), np.random.default_rng(seed)
    )
    second = step.produce(
        _ParentSource([list(genomes[0]), list(genomes[1])]), np.random.default_rng(seed)
    )
    assert first == second


# --- observe: no-op (this baseline assigns no credit) ---------------------------


def test_observe_is_a_noop() -> None:
    # The inherited hook must be callable and side-effect-free: Random keeps no
    # statistics, so it stays out of the adaptive reward machinery.
    RandomOperatorStep([_MarkerMutation([0])]).observe(3.14)


# --- integration on the skeleton: budget, history and reproducibility -----------


def test_run_pins_the_budget_and_history_with_a_pool() -> None:
    problem = _PermutationSortProblem(dimension=8)
    step = RandomOperatorStep(_mixed_real_pool())
    population_size, generations = 12, 10

    result = run(
        problem,
        step,
        np.random.default_rng(0),
        population_size=population_size,
        generations=generations,
    )

    expected_children = (population_size - 1) * generations
    assert result.reproduction_events == expected_children
    assert result.evaluations == population_size + expected_children
    assert problem.eval_count == result.evaluations  # exactly one evaluation per child
    assert len(result.best_quality_history) == generations + 1
    history = result.best_quality_history
    assert all(earlier <= later for earlier, later in zip(history, history[1:], strict=False))
    assert sorted(result.best) == list(range(8))  # the incumbent is a legal permutation


def test_run_is_reproducible_with_a_random_operator_step() -> None:
    def run_once() -> object:
        return run(
            _PermutationSortProblem(dimension=8),
            RandomOperatorStep(_mixed_real_pool()),
            np.random.default_rng(3),
            population_size=10,
            generations=8,
        )

    assert run_once() == run_once()
