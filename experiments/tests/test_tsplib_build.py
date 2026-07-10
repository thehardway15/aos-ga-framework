"""Contract tests for the TSPLIB dataset builder helpers.

These pin the pure helpers of :mod:`experiments.datasets.tsplib_build`:
``normalize_vrp_to_tsp`` (VRP bytes -> canonical TSP bytes, dropping VRP-only
sections while keeping coordinates verbatim) and ``format_opt_tour`` (0-indexed
tour -> TSPLIB ``TOUR`` bytes). The orchestration (``build_dataset``) and the
real-data integrity checks are covered separately.

The module these tests import does not exist yet: this file is the executable
specification. Expected public names: ``normalize_vrp_to_tsp``, ``format_opt_tour``.
"""

from __future__ import annotations

from collections.abc import Sequence

from experiments.datasets.tsplib import parse_opt_tour, parse_tsplib
from experiments.datasets.tsplib_build import format_opt_tour, normalize_vrp_to_tsp


def _vrp_bytes(
    name: str,
    coords: Sequence[tuple[int, int]],
    demands: Sequence[int],
    capacity: int,
) -> bytes:
    """Build the bytes of a minimal TSPLIB CVRP file (with the sections a TSP must drop)."""
    header = (
        f"NAME : {name}\nCOMMENT : test instance\nTYPE : CVRP\n"
        f"DIMENSION : {len(coords)}\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : {capacity}\n"
        "NODE_COORD_SECTION\n"
    )
    coord_lines = "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(coords))
    demand_lines = "DEMAND_SECTION\n" + "".join(f"{i + 1} {d}\n" for i, d in enumerate(demands))
    depot = "DEPOT_SECTION\n 1\n -1\n"
    return (header + coord_lines + demand_lines + depot + "EOF\n").encode()


_VRP_COORDS: list[tuple[int, int]] = [(145, 215), (151, 264), (159, 261)]
_VRP = _vrp_bytes("eil_demo", _VRP_COORDS, [0, 100, 200], 6000)


# --- normalize_vrp_to_tsp ------------------------------------------------------


def test_normalize_preserves_the_instance_semantically() -> None:
    inst = parse_tsplib(normalize_vrp_to_tsp(_VRP))
    assert inst.instance_id == "eil_demo"
    assert inst.dimension == 3
    assert inst.edge_weight_type == "EUC_2D"
    assert inst.coordinates == ((145.0, 215.0), (151.0, 264.0), (159.0, 261.0))


def test_normalize_matches_direct_vrp_parse() -> None:
    # parse_tsplib already ignores VRP sections, so normalization must not change the coords.
    assert parse_tsplib(normalize_vrp_to_tsp(_VRP)).coordinates == parse_tsplib(_VRP).coordinates


def test_normalize_drops_vrp_only_sections() -> None:
    tsp = normalize_vrp_to_tsp(_VRP).decode()
    assert "CVRP" not in tsp
    assert "DEMAND_SECTION" not in tsp
    assert "DEPOT_SECTION" not in tsp
    assert "CAPACITY" not in tsp


def test_normalize_keeps_integer_coordinates() -> None:
    # Original integer coordinates must survive verbatim (no float widening to "145.0").
    tsp = normalize_vrp_to_tsp(_VRP).decode()
    assert "145 215" in tsp


# --- format_opt_tour -----------------------------------------------------------


def test_format_opt_tour_round_trips() -> None:
    tour = [0, 2, 1]
    assert parse_opt_tour(format_opt_tour(tour, name="demo", optimal=42)) == tour


def test_format_opt_tour_records_the_optimal_length() -> None:
    raw = format_opt_tour([0, 1, 2], name="demo", optimal=278).decode()
    assert "278" in raw
