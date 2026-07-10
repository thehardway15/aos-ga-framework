"""Contract tests for the TSPLIB parsers.

These pin the pure parsing API of :mod:`experiments.datasets.tsplib`:
``parse_tsplib`` (bytes of a ``.tsp``/``.vrp`` file -> ``TSPInstance``) and
``parse_opt_tour`` (bytes of a ``.opt.tour`` file -> 0-indexed permutation). They
exercise the real format quirks seen in the dataset: ``KEY : VALUE`` vs
``KEY: VALUE`` headers, integer vs float coordinates, and extra VRP sections that
must be ignored.

The module these tests import does not exist yet: this file is the executable
specification. Expected public names: ``TSPInstance``, ``parse_tsplib``,
``parse_opt_tour``, ``UnsupportedEdgeWeightError``, ``TSPLIBParseError``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from experiments.datasets.tsplib import (
    TSPInstance,
    TSPLIBParseError,
    UnsupportedEdgeWeightError,
    parse_opt_tour,
    parse_tsplib,
)


def _tsp_bytes(
    name: str,
    dimension: int,
    coords: Sequence[tuple[float, float]],
    *,
    edge_weight_type: str = "EUC_2D",
    colon: str = " : ",
    extra_sections: str = "",
) -> bytes:
    """Build the bytes of a minimal TSPLIB file in the on-disk shape."""
    header = (
        f"NAME{colon}{name}\n"
        f"TYPE{colon}TSP\n"
        f"DIMENSION{colon}{dimension}\n"
        f"EDGE_WEIGHT_TYPE{colon}{edge_weight_type}\n"
        "NODE_COORD_SECTION\n"
    )
    body = "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(coords))
    return (header + body + extra_sections + "EOF\n").encode("utf-8")


def _tour_bytes(order_1indexed: Sequence[int]) -> bytes:
    """Build the bytes of a minimal TSPLIB ``.opt.tour`` file (1-indexed, ``-1`` terminated)."""
    lines = "\n".join(str(c) for c in order_1indexed)
    return (
        f"NAME : t.opt.tour\nTYPE : TOUR\nDIMENSION : {len(order_1indexed)}\n"
        f"TOUR_SECTION\n{lines}\n-1\nEOF\n"
    ).encode()


# --- parse_tsplib --------------------------------------------------------------


def test_parses_a_basic_euc2d_instance() -> None:
    inst = parse_tsplib(_tsp_bytes("demo", 3, [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]))
    assert isinstance(inst, TSPInstance)
    assert inst.instance_id == "demo"
    assert inst.dimension == 3
    assert inst.edge_weight_type == "EUC_2D"
    assert inst.coordinates == ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))


def test_tolerates_spaced_and_tight_colons() -> None:
    coords = [(1.0, 2.0), (3.0, 4.0)]
    spaced = parse_tsplib(_tsp_bytes("a", 2, coords, colon=" : "))
    tight = parse_tsplib(_tsp_bytes("a", 2, coords, colon=": "))
    assert spaced.coordinates == tight.coordinates == ((1.0, 2.0), (3.0, 4.0))


def test_parses_float_coordinates() -> None:
    inst = parse_tsplib(_tsp_bytes("b", 2, [(565.0, 575.0), (25.0, 185.0)]))
    assert inst.coordinates == ((565.0, 575.0), (25.0, 185.0))


def test_coordinates_are_zero_indexed() -> None:
    # TSPLIB lists node id 1 first; it must land at index 0.
    inst = parse_tsplib(_tsp_bytes("c", 2, [(7.0, 8.0), (9.0, 10.0)]))
    assert inst.coordinates[0] == (7.0, 8.0)


def test_ignores_unknown_vrp_sections() -> None:
    extra = "DEMAND_SECTION\n1 0\n2 5\nDEPOT_SECTION\n 1\n -1\n"
    inst = parse_tsplib(_tsp_bytes("e22", 2, [(1.0, 2.0), (3.0, 4.0)], extra_sections=extra))
    assert inst.dimension == 2
    assert inst.coordinates == ((1.0, 2.0), (3.0, 4.0))


def test_rejects_non_euc2d_edge_weight() -> None:
    raw = _tsp_bytes("geo", 2, [(1.0, 2.0), (3.0, 4.0)], edge_weight_type="GEO")
    with pytest.raises(UnsupportedEdgeWeightError):
        parse_tsplib(raw)


def test_rejects_dimension_mismatch() -> None:
    raw = _tsp_bytes("bad", 5, [(1.0, 2.0), (3.0, 4.0)])  # claims 5, provides 2
    with pytest.raises(TSPLIBParseError):
        parse_tsplib(raw)


# --- parse_opt_tour ------------------------------------------------------------


def test_parses_a_tour_to_zero_indexed() -> None:
    assert parse_opt_tour(_tour_bytes([1, 3, 2])) == [0, 2, 1]


def test_parsed_tour_is_a_permutation() -> None:
    tour = parse_opt_tour(_tour_bytes([1, 2, 3, 4]))
    assert sorted(tour) == [0, 1, 2, 3]


def test_rejects_a_non_permutation_tour() -> None:
    with pytest.raises(TSPLIBParseError):
        parse_opt_tour(_tour_bytes([1, 1, 2]))  # duplicate city
