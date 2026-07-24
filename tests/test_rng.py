"""Contract tests for the framework's deterministic random-number management.

These pin the public API of :mod:`aos_ga.rng`: a per-run ``numpy.random.Generator``
for framework-controlled stochasticity, and a helper that derives reproducible seed
streams from a ``SeedSequence``. Expected public names: ``run_generator``,
``spawn_seeds``.

The absence of any global random state is part of the contract, not an incidental
property, so it is asserted rather than assumed: neither function may seed or read
Python's ``random`` module or NumPy's legacy global state. That is what lets
independent runs share a process without contaminating each other.
"""

from __future__ import annotations

import pickle
import random

import numpy as np

from aos_ga.rng import run_generator, spawn_seeds


def _sample(generator: np.random.Generator, size: int = 64) -> list[int]:
    """Draw a fixed-length integer sample for comparing two generators."""
    return [int(value) for value in generator.integers(0, 2**31 - 1, size=size)]


def _numpy_global_state() -> bytes:
    """A comparable snapshot of NumPy's global (legacy) RNG state."""
    return pickle.dumps(np.random.get_state())


# --- run_generator -------------------------------------------------------------


def test_run_generator_returns_a_numpy_generator() -> None:
    assert isinstance(run_generator(123), np.random.Generator)


def test_run_generator_is_deterministic_for_the_same_seed() -> None:
    assert _sample(run_generator(7)) == _sample(run_generator(7))


def test_run_generator_differs_for_different_seeds() -> None:
    assert _sample(run_generator(7)) != _sample(run_generator(8))


def test_run_generator_accepts_a_seed_sequence() -> None:
    # The ``Seed`` alias allows passing a SeedSequence; two with equal entropy
    # must yield the same stream.
    first = _sample(run_generator(np.random.SeedSequence(99)))
    second = _sample(run_generator(np.random.SeedSequence(99)))
    assert first == second


def test_run_generator_does_not_touch_global_random() -> None:
    # The generator is the only source of framework randomness; building and
    # drawing from it must leave Python's global ``random`` state untouched.
    before = random.getstate()
    _sample(run_generator(7))
    assert random.getstate() == before


def test_run_generator_does_not_touch_global_numpy_state() -> None:
    # The other half of the same guarantee: NumPy's legacy global state is never
    # seeded or advanced, so a run cannot be perturbed by anything outside it.
    before = _numpy_global_state()
    _sample(run_generator(7))
    assert _numpy_global_state() == before


# --- spawn_seeds ---------------------------------------------------------------


def test_spawn_seeds_returns_the_requested_count() -> None:
    assert len(spawn_seeds(np.random.SeedSequence(1), 30)) == 30


def test_spawn_seeds_are_distinct() -> None:
    seeds = spawn_seeds(np.random.SeedSequence(1), 30)
    assert len(set(seeds)) == len(seeds)


def test_spawn_seeds_are_within_the_uint32_range() -> None:
    assert all(0 <= seed < 2**32 for seed in spawn_seeds(np.random.SeedSequence(1), 30))


def test_spawn_seeds_is_deterministic_for_equal_entropy() -> None:
    # Determinism is defined with respect to the seed, not the object: spawning
    # twice from the same SeedSequence instance advances its child counter, so a
    # fresh SeedSequence is built each time.
    first = spawn_seeds(np.random.SeedSequence(123), 8)
    second = spawn_seeds(np.random.SeedSequence(123), 8)
    assert first == second


def test_spawn_seeds_from_sibling_branches_are_disjoint() -> None:
    # The mechanism the study relies on to keep instance and repetition seeds
    # from overlapping: distinct branches of one root spawn disjoint streams.
    branch_a, branch_b = np.random.SeedSequence(2024).spawn(2)
    assert set(spawn_seeds(branch_a, 16)).isdisjoint(spawn_seeds(branch_b, 16))


def test_spawn_seeds_does_not_touch_global_random() -> None:
    before = random.getstate()
    spawn_seeds(np.random.SeedSequence(1), 8)
    assert random.getstate() == before
