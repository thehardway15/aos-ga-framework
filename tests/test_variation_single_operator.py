"""Contract spec for the single-operator variation step.

``SingleOperatorStep`` makes one fixed operator the entire variation: no
crossover-then-mutation pipeline and no application probabilities -- the operator
is applied on every reproduction event. It is the step used to run each operator
in isolation as an upper reference point for the adaptive operator-selection
strategies, and it is the primitive the adaptive variation step reduces to when a
fixed operator is chosen from the pool. It is generic over the genome type and
reuses the existing ``Operator`` implementations by composition, so the same class
carries a permutation, binary or real-valued operator without hardcoding a
representation.

The class is not implemented yet: this file is the executable specification.
Expected public name (in ``aos_ga.variation.single_operator``):
``SingleOperatorStep``, a subclass of ``aos_ga.core.variation.VariationStep``.

Frozen contract (single-operator step):
- ``SingleOperatorStep(operator)`` stores exactly one operator. Unlike
  ``CanonicalPipeline`` it validates nothing about the operator: there is no slot
  to mis-fill (a single operator of arity 1 or 2 is the whole variation), and the
  ``Operator`` contract already fixes ``arity in {1, 2}``. Construction accepts a
  unary (perturbative) or a binary (recombinative) operator alike; ``kind`` is
  irrelevant here.
- ``produce(select_parent, rng) -> child`` builds exactly ONE unevaluated child:
    1. draw exactly ``operator.arity`` parents by calling ``select_parent`` that
       many times (one draw for a unary operator, two for a binary one);
    2. apply the operator to the drawn parents' genomes -- unconditionally, there
       is no ``p_c``/``p_m`` coin, so the operator fires on every call;
    3. return the operator's single child.
  All randomness is drawn from the single injected ``rng`` in this fixed order, so
  a fixed seed reproduces the child. The step adds no copy of its own: the frozen
  ``Operator.apply`` already returns a fresh, non-aliased child and never mutates
  its parents (including the degenerate cases where an operator copies its parent).
  Legalization against a problem's constraints stays the skeleton's job
  (``Problem.repair``).
- A crossover whose textbook form yields a pair collapses that pair to one child
  inside ``apply``, so the step still returns exactly one child and costs exactly
  one evaluation per child -- the same budget as the shared pool and the canonical
  pipeline.
- ``observe`` is the inherited no-op: this baseline assigns no operator credit, so
  it is excluded from the adaptive reward machinery.
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
from aos_ga.variation.single_operator import SingleOperatorStep

# --- Test doubles --------------------------------------------------------------


class _RecordingCrossover(Operator[list[int]]):
    """Arity-2 recombination double: records the genomes it saw, returns a marker.

    Ignores the parents' contents and emits a fixed marker child, so a test can
    tell whether the operator fired (``calls`` non-empty), on which parent genomes,
    and distinguish its output from a copied parent.
    """

    operator_id = "recording-crossover"
    representation = Representation.PERMUTATION
    arity = 2
    kind = OperatorKind.RECOMBINATIVE

    def __init__(self, child: list[int] | None = None) -> None:
        self._child = list(child) if child is not None else [7, 8, 9]
        self.calls: list[list[list[int]]] = []

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        if len(parents) != self.arity:
            raise ValueError(f"expected {self.arity} parents, got {len(parents)}")
        self.calls.append([list(parent) for parent in parents])
        return list(self._child)


class _RecordingMutation(Operator[list[int]]):
    """Arity-1 perturbation double: records its input, returns a fresh copy of it.

    Identity-content so its firing is asserted via ``inputs`` (not the output),
    independent of any transformation; the fresh copy honours the no-parent-mutation
    rule.
    """

    operator_id = "recording-mutation"
    representation = Representation.PERMUTATION
    arity = 1
    kind = OperatorKind.PERTURBATIVE

    def __init__(self) -> None:
        self.inputs: list[list[int]] = []

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        if len(parents) != self.arity:
            raise ValueError(f"expected {self.arity} parents, got {len(parents)}")
        self.inputs.append(list(parents[0]))
        return list(parents[0])


class _ParentSource:
    """A ``select_parent`` double: yields the given genomes in order, cycling.

    Wraps each genome in a ``Parent`` and counts calls, so a test can pin how many
    parents the step drew and check aliasing against the exact genome objects.
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

    ``f = sum |g[i] - i|`` over a permutation of ``0..n-1`` -- a minimization
    problem with a known optimum (0 at the identity), so a real single-operator step
    runs end-to-end on the skeleton for both a crossover and a mutation. ``evaluate``
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


_MARKER = [7, 8, 9]
_FIRST = [0, 1, 2]
_SECOND = [3, 4, 5]


# --- construction: accepts either arity, validates nothing ----------------------


def test_single_operator_step_is_a_variation_step() -> None:
    assert isinstance(SingleOperatorStep(_RecordingMutation()), VariationStep)


def test_accepts_a_unary_operator() -> None:
    # A perturbative operator (arity 1) is a legal whole variation.
    SingleOperatorStep(_RecordingMutation())  # no raise


def test_accepts_a_binary_operator() -> None:
    # A recombinative operator (arity 2) is an equally legal whole variation --
    # there is no arity slot to mis-fill, so nothing is rejected on construction.
    SingleOperatorStep(_RecordingCrossover())  # no raise


# --- produce: parent draws equal the operator's arity ---------------------------


def test_produce_draws_exactly_two_parents_for_a_binary_operator() -> None:
    step = SingleOperatorStep(_RecordingCrossover())
    source = _ParentSource([_FIRST, _SECOND])
    step.produce(source, np.random.default_rng(0))
    assert source.calls == 2


def test_produce_draws_exactly_one_parent_for_a_unary_operator() -> None:
    step = SingleOperatorStep(_RecordingMutation())
    source = _ParentSource([_FIRST, _SECOND])
    step.produce(source, np.random.default_rng(0))
    assert source.calls == 1


# --- produce: the operator always fires (no p_c/p_m coin) -----------------------


@pytest.mark.parametrize("seed", range(8))
def test_binary_operator_fires_on_every_call_regardless_of_rng(seed: int) -> None:
    crossover = _RecordingCrossover(_MARKER)
    step = SingleOperatorStep(crossover)
    child = step.produce(_ParentSource([_FIRST, _SECOND]), np.random.default_rng(seed))
    # No probability gates it: the operator is applied to both parents every time.
    assert crossover.calls == [[_FIRST, _SECOND]]
    assert child == _MARKER


@pytest.mark.parametrize("seed", range(8))
def test_unary_operator_fires_on_every_call_regardless_of_rng(seed: int) -> None:
    mutation = _RecordingMutation()
    step = SingleOperatorStep(mutation)
    child = step.produce(_ParentSource([_FIRST, _SECOND]), np.random.default_rng(seed))
    assert mutation.inputs == [_FIRST]  # applied to the single drawn parent
    assert child == _FIRST  # recording mutation returns an identity-content copy


# --- produce: freshness, non-aliasing and parent integrity (real operators) -----


@pytest.mark.parametrize("operator", [OrderCrossover(), SegmentInversion()])
def test_child_is_never_an_aliased_parent(operator: Operator[list[int]]) -> None:
    first, second = [0, 1, 2, 3], [3, 2, 1, 0]
    step = SingleOperatorStep(operator)
    child = step.produce(_ParentSource([first, second]), np.random.default_rng(0))
    assert child is not first
    assert child is not second


@pytest.mark.parametrize("operator", [OrderCrossover(), SegmentInversion()])
def test_parents_are_never_mutated_by_the_step(operator: Operator[list[int]]) -> None:
    first, second = [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]
    snapshot = [list(first), list(second)]
    step = SingleOperatorStep(operator)
    step.produce(_ParentSource([first, second]), np.random.default_rng(0))
    assert [first, second] == snapshot


# --- produce: permutation legality and determinism (real operators) -------------


@pytest.mark.parametrize("operator", [OrderCrossover(), SegmentInversion()])
def test_child_stays_a_permutation_across_seeds(operator: Operator[list[int]]) -> None:
    genomes = [[0, 1, 2, 3, 4, 5], [3, 5, 1, 0, 4, 2]]  # same element multiset
    step = SingleOperatorStep(operator)
    for seed in range(32):
        child = step.produce(
            _ParentSource([list(genomes[0]), list(genomes[1])]),
            np.random.default_rng(seed),
        )
        assert sorted(child) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("operator", [OrderCrossover(), SegmentInversion()])
def test_produce_is_deterministic_for_a_fixed_seed(operator: Operator[list[int]]) -> None:
    genomes = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]
    step = SingleOperatorStep(operator)
    for seed in range(16):
        first = step.produce(
            _ParentSource([list(genomes[0]), list(genomes[1])]), np.random.default_rng(seed)
        )
        second = step.produce(
            _ParentSource([list(genomes[0]), list(genomes[1])]), np.random.default_rng(seed)
        )
        assert first == second


# --- observe: no-op (this baseline assigns no credit) ---------------------------


def test_observe_is_a_noop() -> None:
    # The inherited hook must be callable and side-effect-free: it does nothing and
    # returns nothing, so the step stays out of the adaptive reward machinery.
    SingleOperatorStep(_RecordingMutation()).observe(3.14)


# --- integration on the skeleton: budget, history and reproducibility -----------


@pytest.mark.parametrize("operator", [OrderCrossover(), SegmentInversion()])
def test_run_pins_the_budget_and_history_on_a_permutation_problem(
    operator: Operator[list[int]],
) -> None:
    problem = _PermutationSortProblem(dimension=8)
    step = SingleOperatorStep(operator)
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


@pytest.mark.parametrize("operator", [OrderCrossover(), SegmentInversion()])
def test_run_is_reproducible_with_a_single_operator_step(
    operator: Operator[list[int]],
) -> None:
    def run_once() -> object:
        return run(
            _PermutationSortProblem(dimension=8),
            SingleOperatorStep(operator),
            np.random.default_rng(3),
            population_size=10,
            generations=8,
        )

    assert run_once() == run_once()
