"""Contract spec for the canonical two-operator variation step (CGA baseline).

``CanonicalPipeline`` is the first concrete ``VariationStep``: the classic genetic
algorithm's variation model, a fixed pair of complementary operators applied to
one offspring -- crossover with probability ``p_c``, then mutation with
probability ``p_m``. It is generic over the genome type and parameterized by its
operators and probabilities in the constructor, so the same class assembles the
TSP baseline (Order Crossover ``p_c=0.9`` then Segment Inversion ``p_m=0.1``) and,
later, the binary and real-valued baselines -- it never hardcodes a
representation and reuses the existing ``Operator`` implementations by composition.

The class is not implemented yet: this file is the executable specification.
Expected public name (in ``aos_ga.variation.canonical``): ``CanonicalPipeline``,
a subclass of ``aos_ga.core.variation.VariationStep``.

Frozen contract (canonical pipeline):
- ``CanonicalPipeline(crossover, p_c, mutation, p_m)`` stores the two operators and
  their per-individual application probabilities. It validates on construction and
  raises ``ValueError`` when ``mutation.arity != 1`` (the pipeline applies mutation
  to exactly one child), ``crossover.arity < 2`` (recombination needs two parents,
  and the crossover-skipped path copies the first), or either probability falls
  outside ``[0.0, 1.0]``. The operators' ``kind`` is not validated -- it is
  AOS-logging metadata, not a structural precondition.
- ``produce(select_parent, rng) -> child`` builds exactly ONE unevaluated child:
    1. draw exactly ``crossover.arity`` parents by calling ``select_parent`` that
       many times, unconditionally and before the ``p_c`` coin (parent selection is
       independent of the coin);
    2. if ``rng.random() < p_c`` apply ``crossover`` to the parents' genomes,
       otherwise copy the FIRST drawn parent's genome (a fresh, non-aliased list);
    3. if ``rng.random() < p_m`` apply ``mutation`` to that child, otherwise leave
       it unchanged;
    4. return the child.
  All randomness is drawn from the single injected ``rng`` in this fixed order, so
  a fixed seed reproduces the child. The returned child is always a fresh genome,
  never an aliased parent, and no parent is ever mutated. Legalization against a
  problem's constraints stays the skeleton's job (``Problem.repair``).
- Probability boundaries follow the ``rng.random() < p`` convention (``random()``
  in ``[0, 1)``): ``p == 0.0`` never fires the step, ``p == 1.0`` always fires it.
- ``observe`` is the inherited no-op: the canonical GA assigns no operator credit,
  so it is excluded from the AOS reward machinery.
- Budget: the step returns exactly one child regardless of how many operators fire
  (zero, one or two), so on the skeleton it costs exactly one evaluation per child
  and ``N-1`` children per generation -- the same budget as the shared-pool model.
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
from aos_ga.variation.canonical import CanonicalPipeline

# --- Test doubles --------------------------------------------------------------


class _RecordingCrossover(Operator[list[int]]):
    """Arity-2 recombination double: records the genomes it saw, returns a marker.

    Ignores the parents' contents and emits a fixed marker child, so a test can
    tell whether crossover fired (``calls`` non-empty), on which parent genomes,
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


class _FixedArityCrossover(Operator[list[int]]):
    """Recombination double with a configurable arity.

    Proves construction rejects any crossover that is not at least binary; it is
    never actually applied.
    """

    operator_id = "fixed-arity-crossover"
    representation = Representation.PERMUTATION
    kind = OperatorKind.RECOMBINATIVE

    def __init__(self, arity: int) -> None:
        self.arity = arity

    def apply(self, parents: Sequence[list[int]], rng: Generator) -> list[int]:
        raise AssertionError("a crossover with arity < 2 must be rejected before it is applied")


class _ParentSource:
    """A ``select_parent`` double: yields the given genomes in order, cycling.

    Wraps each genome in a ``Parent`` and counts calls, so a test can pin how many
    parents the pipeline drew and check aliasing against the exact genome objects.
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
    problem with a known optimum (0 at the identity), so a real ``CanonicalPipeline``
    of OX + inversion runs end-to-end on the skeleton. ``evaluate`` tallies calls so
    a test can pin the evaluation budget. OX and inversion keep offspring legal
    permutations, so the inherited identity ``repair`` suffices.
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


def _marker_pipeline(
    p_c: float, p_m: float
) -> tuple[CanonicalPipeline[list[int]], _RecordingCrossover, _RecordingMutation]:
    """A pipeline over recording doubles, returned with its two operators."""
    crossover = _RecordingCrossover(_MARKER)
    mutation = _RecordingMutation()
    return CanonicalPipeline(crossover, p_c, mutation, p_m), crossover, mutation


# --- construction: type, arity and probability validation ----------------------


def test_canonical_pipeline_is_a_variation_step() -> None:
    pipeline, _, _ = _marker_pipeline(0.9, 0.1)
    assert isinstance(pipeline, VariationStep)


def test_rejects_mutation_whose_arity_is_not_one() -> None:
    # An arity-2 operator in the mutation slot cannot mutate a single child.
    with pytest.raises(ValueError):
        CanonicalPipeline(_RecordingCrossover(), 0.9, _RecordingCrossover(), 0.1)


@pytest.mark.parametrize("arity", [0, 1])
def test_rejects_crossover_whose_arity_is_below_two(arity: int) -> None:
    # Recombination needs two parents; a nullary or unary crossover is rejected.
    with pytest.raises(ValueError):
        CanonicalPipeline(_FixedArityCrossover(arity), 0.9, _RecordingMutation(), 0.1)


@pytest.mark.parametrize(
    ("p_c", "p_m"),
    [(-0.1, 0.1), (1.5, 0.1), (0.9, -0.1), (0.9, 2.0)],
)
def test_rejects_probabilities_outside_the_unit_interval(p_c: float, p_m: float) -> None:
    with pytest.raises(ValueError):
        CanonicalPipeline(_RecordingCrossover(), p_c, _RecordingMutation(), p_m)


@pytest.mark.parametrize(("p_c", "p_m"), [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
def test_accepts_boundary_probabilities(p_c: float, p_m: float) -> None:
    CanonicalPipeline(_RecordingCrossover(), p_c, _RecordingMutation(), p_m)  # no raise


# --- produce: parent draws (arity of crossover, independent of the p_c coin) ----


@pytest.mark.parametrize("p_c", [0.0, 1.0])
def test_produce_draws_exactly_crossover_arity_parents(p_c: float) -> None:
    pipeline, crossover, _ = _marker_pipeline(p_c, 0.0)
    source = _ParentSource([_FIRST, _SECOND])
    pipeline.produce(source, np.random.default_rng(0))
    # Two parents are drawn whether or not crossover fires.
    assert source.calls == crossover.arity == 2


# --- produce: the crossover step -----------------------------------------------


def test_crossover_fires_on_both_parents_when_p_c_is_one() -> None:
    pipeline, crossover, mutation = _marker_pipeline(1.0, 0.0)
    child = pipeline.produce(_ParentSource([_FIRST, _SECOND]), np.random.default_rng(0))
    assert crossover.calls == [[_FIRST, _SECOND]]  # applied to both parents' genomes
    assert child == _MARKER
    assert mutation.inputs == []


def test_crossover_is_skipped_and_first_parent_copied_when_p_c_is_zero() -> None:
    pipeline, crossover, _ = _marker_pipeline(0.0, 0.0)
    source = _ParentSource([_FIRST, _SECOND])
    child = pipeline.produce(source, np.random.default_rng(0))
    assert crossover.calls == []  # crossover never applied
    assert child == _FIRST  # a copy of the first drawn parent
    assert child is not _FIRST  # ... but a fresh, non-aliased genome
    assert source.calls == 2  # both parents still drawn


# --- produce: the mutation step and pipeline order ------------------------------


def test_mutation_fires_on_the_recombined_child_when_p_m_is_one() -> None:
    pipeline, _, mutation = _marker_pipeline(1.0, 1.0)
    child = pipeline.produce(_ParentSource([_FIRST, _SECOND]), np.random.default_rng(0))
    # Mutation consumed crossover's output -> crossover runs before mutation.
    assert mutation.inputs == [_MARKER]
    assert child == _MARKER  # recording mutation returns an identity-content copy


def test_mutation_is_skipped_when_p_m_is_zero() -> None:
    pipeline, _, mutation = _marker_pipeline(1.0, 0.0)
    child = pipeline.produce(_ParentSource([_FIRST, _SECOND]), np.random.default_rng(0))
    assert mutation.inputs == []
    assert child == _MARKER


def test_mutation_applies_to_the_copied_parent_when_crossover_is_skipped() -> None:
    pipeline, crossover, mutation = _marker_pipeline(0.0, 1.0)
    child = pipeline.produce(_ParentSource([_FIRST, _SECOND]), np.random.default_rng(0))
    assert crossover.calls == []
    assert mutation.inputs == [_FIRST]  # mutation ran on the copy of the first parent
    assert child == _FIRST


# --- produce: freshness, non-aliasing and parent integrity ----------------------


@pytest.mark.parametrize(("p_c", "p_m"), [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])
def test_child_is_never_an_aliased_parent(p_c: float, p_m: float) -> None:
    first, second = [0, 1, 2, 3], [3, 2, 1, 0]
    pipeline = CanonicalPipeline(OrderCrossover(), p_c, SegmentInversion(), p_m)
    child = pipeline.produce(_ParentSource([first, second]), np.random.default_rng(0))
    assert child is not first
    assert child is not second


def test_parents_are_never_mutated_by_the_pipeline() -> None:
    first, second = [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]
    snapshot = [list(first), list(second)]
    pipeline = CanonicalPipeline(OrderCrossover(), 1.0, SegmentInversion(), 1.0)
    pipeline.produce(_ParentSource([first, second]), np.random.default_rng(0))
    assert [first, second] == snapshot


# --- produce: permutation legality and determinism (real operators) -------------


@pytest.mark.parametrize(("p_c", "p_m"), [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])
def test_child_stays_a_permutation_across_seeds(p_c: float, p_m: float) -> None:
    genomes = [[0, 1, 2, 3, 4, 5], [3, 5, 1, 0, 4, 2]]  # same element multiset
    pipeline = CanonicalPipeline(OrderCrossover(), p_c, SegmentInversion(), p_m)
    for seed in range(32):
        child = pipeline.produce(
            _ParentSource([list(genomes[0]), list(genomes[1])]),
            np.random.default_rng(seed),
        )
        assert sorted(child) == [0, 1, 2, 3, 4, 5]


def test_produce_is_deterministic_for_a_fixed_seed() -> None:
    pipeline = CanonicalPipeline(OrderCrossover(), 0.9, SegmentInversion(), 0.1)
    genomes = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]
    for seed in range(16):
        first = pipeline.produce(
            _ParentSource([list(genomes[0]), list(genomes[1])]), np.random.default_rng(seed)
        )
        second = pipeline.produce(
            _ParentSource([list(genomes[0]), list(genomes[1])]), np.random.default_rng(seed)
        )
        assert first == second


# --- observe: no-op (the canonical GA assigns no credit) ------------------------


def test_observe_is_a_noop() -> None:
    # The canonical GA never overrides observe; the inherited hook must be callable
    # and side-effect-free -- it does nothing and returns nothing.
    pipeline, _, _ = _marker_pipeline(0.9, 0.1)
    pipeline.observe(3.14)


# --- integration on the skeleton: budget, history and reproducibility -----------


def test_run_pins_the_budget_and_history_on_a_permutation_problem() -> None:
    problem = _PermutationSortProblem(dimension=8)
    pipeline = CanonicalPipeline(OrderCrossover(), 0.9, SegmentInversion(), 0.1)
    population_size, generations = 12, 10

    result = run(
        problem,
        pipeline,
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


def test_run_is_reproducible_with_the_canonical_pipeline() -> None:
    def run_once() -> object:
        return run(
            _PermutationSortProblem(dimension=8),
            CanonicalPipeline(OrderCrossover(), 0.9, SegmentInversion(), 0.1),
            np.random.default_rng(3),
            population_size=10,
            generations=8,
        )

    assert run_once() == run_once()
