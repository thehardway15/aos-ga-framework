"""Solution-encoding vocabulary shared across the framework.

The genome families the engine operates on and the generic ``Genome`` type that
both the problem interface and the (problem-agnostic) operator interface are
parameterized by. Foundation vocabulary: problem and operator both depend on it,
and neither depends on the other.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

# The genome (individual) type a problem defines and its operators transform.
Genome = TypeVar("Genome")


class Representation(Enum):
    """Genome family a problem operates on (selects the operator pool)."""

    PERMUTATION = "permutation"
    BINARY = "binary"
    REAL = "real"
