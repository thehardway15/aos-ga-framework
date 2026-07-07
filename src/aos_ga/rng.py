"""Deterministic random-number management.

A fresh ``numpy.random.Generator`` per run backs all framework-controlled
stochasticity (population initialization, adaptive operator-selection decisions,
credit assignment). Seeds are derived through ``numpy.random.SeedSequence``, and
distinct branches of one master sequence stay disjoint so their streams never
overlap -- this is how the instance seeds and the repetition seeds are kept from
colliding. The module holds no study-specific constants; callers pass their own
seeds.

Determinism boundary (DEAP). DEAP's built-in operators draw from Python's global
``random`` module and expose no rng-injection parameter, so full reproducibility
requires seeding that module once per run (see :func:`seed_global_random`). This
is the single, deliberate use of global random state. Framework code otherwise
draws only from the injected ``Generator`` and never seeds or reads NumPy's
global state. Because the ``random`` module is process-global, independent runs
must be isolated per process rather than per thread.
"""

import random

import numpy as np
from numpy.random import Generator, SeedSequence


def run_generator(seed: int | SeedSequence) -> Generator:
    """Return a fresh ``Generator`` seeded from ``seed``.

    The single source of framework-controlled randomness for one run. Pure: it
    touches no global state. An ``int`` is normalized to ``SeedSequence(int)``.
    """
    if isinstance(seed, int):
        seed = np.random.SeedSequence(seed)
    return np.random.default_rng(seed)


def spawn_seeds(seed_sequence: SeedSequence, count: int) -> list[int]:
    """Derive ``count`` independent ``uint32`` seeds from ``seed_sequence``.

    Spawns ``count`` child sequences and materializes one ``uint32`` from each,
    yielding serializable integer seeds -- the same derivation used for both the
    repetition seeds and the instance seeds. Deterministic with respect to the
    sequence's entropy: build a fresh ``SeedSequence`` to reproduce a list, since
    spawning advances a sequence's internal child counter.
    """
    children = seed_sequence.spawn(count)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def seed_global_random(seed: int | SeedSequence) -> None:
    """Seed Python's process-global ``random`` module -- the DEAP boundary.

    DEAP's built-in operators draw from this module, so it is seeded once per run
    to make them reproducible. Only the global ``random`` module is touched;
    NumPy's global state is never seeded. Call once at the start of each run,
    inside each worker process.
    """
    if isinstance(seed, int):
        seed = np.random.SeedSequence(seed)
    random.seed(int(seed.generate_state(1, dtype=np.uint32)[0]))


def init_run_randomness(seed: int) -> Generator:
    """Initialize all randomness for one run from its integer ``seed``.

    Spawns two disjoint substreams from ``seed``: one seeds the global ``random``
    module (DEAP operators), the other backs the returned ``Generator`` used by
    framework code. Seeding the global ``random`` module is a documented side
    effect.
    """
    numpy_ss, random_ss = np.random.SeedSequence(seed).spawn(2)
    seed_global_random(random_ss)
    return run_generator(numpy_ss)
