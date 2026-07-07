"""Contract tests for the framework's deterministic random-number management.

These pin the public API of :mod:`aos_ga.rng`: a per-run ``numpy.random.Generator``
for framework-controlled stochasticity, a helper that derives reproducible seed
streams, the single seam that seeds Python's global ``random`` module for DEAP's
built-in operators, and the per-run initializer that wires the two together.

The functions these tests import are not implemented yet: this file is the
executable specification of the contract. Expected public names: ``run_generator``,
``spawn_seeds``, ``seed_global_random``, ``init_run_randomness``.
"""

from __future__ import annotations

import pickle
import random
from collections.abc import Iterator

import numpy as np
import pytest

from aos_ga.rng import (
    init_run_randomness,
    run_generator,
    seed_global_random,
    spawn_seeds,
)


@pytest.fixture(autouse=True)
def _isolate_global_random() -> Iterator[None]:
    """Save and restore the global ``random`` state so tests stay hermetic.

    ``seed_global_random`` and ``init_run_randomness`` mutate process-global
    state by design; this keeps that side effect from leaking between tests.
    """
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


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


# --- seed_global_random --------------------------------------------------------


def test_seed_global_random_is_reproducible() -> None:
    seed_global_random(5)
    first = [random.random() for _ in range(20)]
    seed_global_random(5)
    second = [random.random() for _ in range(20)]
    assert first == second


def test_seed_global_random_differs_for_different_seeds() -> None:
    seed_global_random(5)
    first = [random.random() for _ in range(20)]
    seed_global_random(6)
    second = [random.random() for _ in range(20)]
    assert first != second


def test_seed_global_random_accepts_a_seed_sequence() -> None:
    seed_global_random(np.random.SeedSequence(5))
    first = [random.random() for _ in range(20)]
    seed_global_random(np.random.SeedSequence(5))
    second = [random.random() for _ in range(20)]
    assert first == second


def test_seed_global_random_does_not_touch_global_numpy_state() -> None:
    # seed_global_random is the DEAP boundary: it seeds Python's global random
    # module only. Framework code draws from injected Generators, never from
    # NumPy's global state, so seeding must leave that state untouched.
    before = _numpy_global_state()
    seed_global_random(5)
    assert _numpy_global_state() == before


# --- init_run_randomness -------------------------------------------------------


def test_init_run_randomness_returns_a_generator() -> None:
    assert isinstance(init_run_randomness(42), np.random.Generator)


def test_init_run_randomness_is_deterministic_end_to_end() -> None:
    # The same run seed must reproduce both the numpy stream and the global
    # ``random`` stream it seeds as a side effect.
    generator = init_run_randomness(42)
    numpy_draws = _sample(generator)
    global_draws = [random.random() for _ in range(20)]

    generator_again = init_run_randomness(42)
    assert _sample(generator_again) == numpy_draws
    assert [random.random() for _ in range(20)] == global_draws


def test_init_run_randomness_differs_for_different_run_seeds() -> None:
    numpy_draws = _sample(init_run_randomness(42))
    global_draws = [random.random() for _ in range(20)]

    other_numpy_draws = _sample(init_run_randomness(43))
    other_global_draws = [random.random() for _ in range(20)]

    assert numpy_draws != other_numpy_draws
    assert global_draws != other_global_draws


def test_init_run_randomness_uses_disjoint_substreams() -> None:
    # The numpy generator and the global ``random`` seed are spawned from two
    # disjoint substreams, so the generator's stream is not echoed by the global
    # one for the same run seed.
    generator = init_run_randomness(7)
    numpy_draws = _sample(generator)
    global_draws = [random.random() for _ in range(64)]
    assert [float(value) for value in numpy_draws] != global_draws
