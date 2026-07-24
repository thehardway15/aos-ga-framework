"""Deterministic random-number management.

A fresh ``numpy.random.Generator`` per run backs all framework-controlled
stochasticity (population initialization, adaptive operator-selection decisions,
credit assignment). Seeds are derived through ``numpy.random.SeedSequence``, and
distinct branches of one master sequence stay disjoint so their streams never
overlap -- this is how the instance seeds and the repetition seeds are kept from
colliding. The module holds no study-specific constants; callers pass their own
seeds.

There is no global random state anywhere in the framework: every draw comes from
a ``Generator`` handed down from the run's seed, so a run is reproducible without
process isolation and two runs can never contaminate each other. That is a
property of the engine being written here rather than taken from a library --
the established EC frameworks drive their operators from Python's process-global
``random`` module, which would have forced a documented exception at exactly this
point.

Reproducibility is bit-exact for a fixed NumPy version, not across versions:
``Generator`` streams are explicitly outside NumPy's stream-compatibility
guarantee (NEP 19), which covers only the legacy ``RandomState``. The replication
environment therefore pins NumPy exactly; see ``replication/README.md``.
"""

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
