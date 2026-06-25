"""Project-wide constants.

These values are fixed for the whole project so that every run is reproducible
from a single master seed; see :mod:`aos_ga.rng` for how the derived random
streams are spawned.
"""

# Master seed from which every reproducible random stream is derived. The
# instance-generation branch and the repetition branch are spawned from this
# seed on disjoint ``numpy.random.SeedSequence`` paths.
MASTER_SEED = 20260101
