"""Deterministic random-number management.

One injected ``numpy.random.Generator`` per run, with seeds derived from
:data:`aos_ga.constants.MASTER_SEED` through ``numpy.random.SeedSequence``. The
instance-generation branch and the repetition branch are kept disjoint so that
their streams never overlap. This module also documents the boundaries of
determinism, where DEAP relies on global random state and how that state is
isolated.
"""
