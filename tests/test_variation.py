"""Contract spec for the interchangeable variation-step interface.

The GA skeleton produces every offspring by delegating to a :class:`VariationStep`:
one call builds exactly ONE child from the current population, drawing parents
through an injected tournament service and randomness only from an injected
``Generator``. The skeleton then evaluates that child exactly once. How many
operators a step applies *inside* one child is the step's business, never the
skeleton's -- this is the seam that lets the classic GA (a ``p_c`` crossover then
a ``p_m`` mutation) and future AOS (one pooled operator + credit) share one loop.

The names below are not implemented yet: this file is the executable specification
of the contract. Expected public names (in ``aos_ga.core.variation``):
``Parent`` (a frozen record) and ``VariationStep`` (generic over the genome type,
like ``Operator`` and ``Problem``).

Frozen contract (variation seam):
- ``Parent`` is an immutable record of one tournament winner: its ``index`` in the
  population (a ``parent_id`` for AOS logs), its ``genome`` and its quality
  ``g`` (``quality``). Carrying ``g`` lets a future AOS step read ``g_ref =
  max(parent.quality ...)`` for free -- the parents are already evaluated.
- ``VariationStep.produce(select_parent, rng) -> child`` returns exactly ONE
  child genome (unevaluated). The step calls ``select_parent`` once per parent it
  needs (arity is the step's concern) and draws randomness only from ``rng`` (no
  global state). Legalization against a problem's constraints is the caller's job
  (``Problem.repair``), not the step's.
- ``VariationStep.observe(child_quality)`` is the post-evaluation hook: the
  skeleton calls it with the child's ``g`` right after evaluating the child. The
  base implementation is a no-op (the classic GA ignores it); a future AOS step
  overrides it to turn ``child_quality`` and the ``g_ref`` it recorded in
  ``produce`` into a reward and update its selection statistics. Freezing the call
  site now keeps AOS a plug-in, not a rewrite -- the credit semantics stay out of
  scope here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.random import Generator

from aos_ga.core.variation import Parent, VariationStep


class _CopyOneParent(VariationStep[list[int]]):
    """Minimal step double: draw one parent, return a fresh copy of its genome.

    Enough to exercise the interface -- it uses ``select_parent`` and returns a
    single fresh child -- without depending on any real operator mechanics.
    """

    def produce(self, select_parent: Callable[[], Parent[list[int]]], rng: Generator) -> list[int]:
        return list(select_parent().genome)


# --- Parent: an immutable (index, genome, quality) record ----------------------


def test_parent_exposes_index_genome_and_quality() -> None:
    parent = Parent(index=4, genome=[1, 2, 3], quality=2.5)
    assert parent.index == 4
    assert parent.genome == [1, 2, 3]
    assert parent.quality == 2.5


def test_parent_is_frozen() -> None:
    # Immutable so a step can never corrupt the population it reads from.
    parent = Parent(index=0, genome=[0], quality=0.0)
    with pytest.raises(FrozenInstanceError):
        parent.index = 1  # type: ignore[misc]


# --- VariationStep: abstractness -----------------------------------------------


def test_variation_step_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        VariationStep()  # type: ignore[abstract]


def test_subclass_missing_produce_is_abstract() -> None:
    class _NoProduce(VariationStep[list[int]]):
        pass

    with pytest.raises(TypeError):
        _NoProduce()  # type: ignore[abstract]


# --- VariationStep.produce: one child from the injected parent source ----------


def test_produce_reads_parents_only_from_the_injected_source() -> None:
    def source() -> Parent[list[int]]:
        return Parent(index=2, genome=[7, 8, 9], quality=1.0)

    child = _CopyOneParent().produce(source, np.random.default_rng(0))
    assert child == [7, 8, 9]


def test_produce_returns_a_fresh_child_not_an_aliased_parent() -> None:
    parent_genome = [7, 8, 9]

    def source() -> Parent[list[int]]:
        return Parent(index=0, genome=parent_genome, quality=1.0)

    child = _CopyOneParent().produce(source, np.random.default_rng(0))
    assert child is not parent_genome


# --- VariationStep.observe: no-op by default (AOS plug-in point) ----------------


def test_observe_default_is_a_noop() -> None:
    # The classic GA never overrides observe; the base hook must be callable and
    # side-effect-free -- it does nothing and returns nothing.
    _CopyOneParent().observe(3.14)
