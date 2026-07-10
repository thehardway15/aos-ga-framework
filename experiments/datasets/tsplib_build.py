"""Builder for the frozen TSPLIB dataset.

Run once to complete ``data/tsplib/``: normalizes ``eil22.vrp`` into a canonical
``eil22.tsp``, computes its optimum with Held-Karp, writes ``eil22.opt.tour``, and
emits a checksummed ``manifest.json`` for all three instances. The optimal length
of each instance is derived by evaluating its optimal tour, and the known TSPLIB
optima (eil51, berlin52) are asserted as a consistency check. Deterministic and
idempotent; run with ``python -m experiments.datasets.tsplib_build``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..problems.tsp import TSPProblem
from .exact_tsp import held_karp
from .tsplib import (
    TSPLIB_DIR,
    _compute_sha256,
    parse_opt_tour,
    parse_tsplib,
)

# Instances that already ship in data/tsplib/ with a known published TSP optimum.
_KNOWN_OPTIMA = {"eil51": 426, "berlin52": 7542}

_SOURCES = {
    "eil22": "TSPLIB VRP set (Eilon et al.); TSP optimum computed via Held-Karp",
    "eil51": "TSPLIB (Reinelt 1991)",
    "berlin52": "TSPLIB (Reinelt 1991)",
}

_INSTANCES = ("eil22", "eil51", "berlin52")


def normalize_vrp_to_tsp(vrp_raw: bytes) -> bytes:
    """Turn a TSPLIB CVRP file into a canonical symmetric-TSP file.

    Retains ``NAME``/``DIMENSION``/``EDGE_WEIGHT_TYPE`` and the ``NODE_COORD_SECTION``
    verbatim (so integer coordinates are preserved), rewrites ``TYPE`` to ``TSP``, and
    drops the VRP-only fields (``CAPACITY``, ``DEMAND_SECTION``, ``DEPOT_SECTION``).
    """
    out: list[str] = []
    for line in vrp_raw.decode("utf-8").splitlines():
        if line.startswith("DEMAND_SECTION"):
            break  # everything from here on (demands, depot) is VRP-only
        if line.startswith("TYPE"):
            out.append("TYPE : TSP")
        elif line.startswith("CAPACITY"):
            continue
        else:
            out.append(line)
    out.append("EOF")
    return ("\n".join(out) + "\n").encode("utf-8")


def format_opt_tour(tour: list[int], *, name: str, optimal: int) -> bytes:
    """Render a 0-indexed tour as the bytes of a TSPLIB ``.opt.tour`` file (1-indexed)."""
    lines = [
        f"NAME : {name}.opt.tour",
        f"COMMENT : Optimal tour ({optimal})",
        "TYPE : TOUR",
        f"DIMENSION : {len(tour)}",
        "TOUR_SECTION",
        *[str(city + 1) for city in tour],
        "-1",
        "EOF",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize like the rest of the datasets: sorted keys, 4-space indent, trailing newline."""
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=4) + "\n").encode(
        "utf-8"
    )


def build_dataset(*, data_dir: Path = TSPLIB_DIR) -> None:
    """Generate ``eil22.tsp``/``eil22.opt.tour`` and write ``manifest.json`` for all instances."""
    # 1. Normalize the eil22 VRP source into a canonical TSP file.
    vrp_raw = (data_dir / "sources" / "eil22.vrp").read_bytes()
    (data_dir / "eil22.tsp").write_bytes(normalize_vrp_to_tsp(vrp_raw))

    # 2. Solve eil22 exactly and freeze its optimal tour.
    eil22 = TSPProblem(parse_tsplib((data_dir / "eil22.tsp").read_bytes()))
    optimal, tour = held_karp(eil22.distances)
    (data_dir / "eil22.opt.tour").write_bytes(format_opt_tour(tour, name="eil22", optimal=optimal))

    # 3. Build the manifest: derive each optimum from its tour; cross-check known optima.
    entries: list[dict[str, Any]] = []
    for instance_id in _INSTANCES:
        tsp_raw = (data_dir / f"{instance_id}.tsp").read_bytes()
        tour_raw = (data_dir / f"{instance_id}.opt.tour").read_bytes()
        problem = TSPProblem(parse_tsplib(tsp_raw))
        optimal_length = int(problem.evaluate(parse_opt_tour(tour_raw)))

        expected = _KNOWN_OPTIMA.get(instance_id)
        if expected is not None and optimal_length != expected:
            raise ValueError(
                f"{instance_id}: computed optimum {optimal_length} != known optimum {expected}"
            )

        entries.append(
            {
                "instance_id": instance_id,
                "dimension": problem.dimension,
                "edge_weight_type": "EUC_2D",
                "optimal_length": optimal_length,
                "checksum": _compute_sha256(tsp_raw),
                "opt_tour_checksum": _compute_sha256(tour_raw),
                "source": _SOURCES[instance_id],
            }
        )

    (data_dir / "manifest.json").write_bytes(
        _canonical_json({"schema_version": 1, "instances": entries})
    )


if __name__ == "__main__":
    build_dataset()
