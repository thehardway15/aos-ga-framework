"""Contract tests for the TSPLIB dataset loader.

These pin ``load_manifest`` / ``load_instance`` / ``load_optimal_tour`` against a
synthetic dataset built in a temp directory in the on-disk shape: a canonical
JSON manifest plus ``.tsp`` and ``.opt.tour`` files, checksum-verified. They never
touch the real ``data/tsplib/`` tree, so they pin the loader independently of the
frozen artifacts.

The names these tests import do not exist yet: this file is the executable
specification. Expected public names: ``TSPManifestEntry``, ``load_manifest``,
``load_instance``, ``load_optimal_tour``, ``ChecksumError``, ``ManifestError``,
``UnknownInstanceError``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from experiments.datasets.tsplib import (
    ChecksumError,
    ManifestError,
    TSPInstance,
    TSPManifestEntry,
    UnknownInstanceError,
    load_instance,
    load_manifest,
    load_optimal_tour,
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _tsp_bytes(name: str, coords: Sequence[tuple[float, float]]) -> bytes:
    header = (
        f"NAME : {name}\nTYPE : TSP\nDIMENSION : {len(coords)}\n"
        "EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
    )
    body = "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(coords))
    return (header + body + "EOF\n").encode()


def _tour_bytes(order_1indexed: Sequence[int]) -> bytes:
    lines = "\n".join(str(c) for c in order_1indexed)
    return (
        f"NAME : t.opt.tour\nTYPE : TOUR\nDIMENSION : {len(order_1indexed)}\n"
        f"TOUR_SECTION\n{lines}\n-1\nEOF\n"
    ).encode()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=4) + "\n").encode()


_DEMO: dict[str, dict[str, Any]] = {
    "demo_a": {"coords": [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], "tour": [1, 2, 3], "optimal": 34},
    "demo_b": {
        "coords": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        "tour": [1, 2, 3, 4],
        "optimal": 40,
    },
}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write the demo instances, their optimal tours, and a manifest into a temp dir."""
    entries: list[dict[str, Any]] = []
    for iid, data in _DEMO.items():
        tsp_raw = _tsp_bytes(iid, data["coords"])
        tour_raw = _tour_bytes(data["tour"])
        (tmp_path / f"{iid}.tsp").write_bytes(tsp_raw)
        (tmp_path / f"{iid}.opt.tour").write_bytes(tour_raw)
        entries.append(
            {
                "instance_id": iid,
                "dimension": len(data["coords"]),
                "edge_weight_type": "EUC_2D",
                "optimal_length": data["optimal"],
                "checksum": _sha256(tsp_raw),
                "opt_tour_checksum": _sha256(tour_raw),
                "source": "test",
            }
        )
    manifest = {"schema_version": 1, "instances": entries}
    (tmp_path / "manifest.json").write_bytes(_canonical_bytes(manifest))
    return tmp_path


# --- load_manifest -------------------------------------------------------------


def test_load_manifest_returns_one_entry_per_instance(dataset_dir: Path) -> None:
    entries = load_manifest(dataset_dir)
    assert {e.instance_id for e in entries} == set(_DEMO)
    assert all(isinstance(e, TSPManifestEntry) for e in entries)


def test_load_manifest_extracts_fields(dataset_dir: Path) -> None:
    by_id = {e.instance_id: e for e in load_manifest(dataset_dir)}
    a = by_id["demo_a"]
    assert a.dimension == 3
    assert a.edge_weight_type == "EUC_2D"
    assert a.optimal_length == 34
    assert a.checksum.startswith("sha256:")
    assert a.opt_tour_checksum.startswith("sha256:")
    assert a.source == "test"


def test_load_manifest_missing_instances_key_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b'{"schema_version": 1}\n')
    with pytest.raises(ManifestError):
        load_manifest(tmp_path)


def test_load_manifest_bad_schema_version_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_bytes(b'{"schema_version": 2, "instances": []}\n')
    with pytest.raises(ManifestError):
        load_manifest(tmp_path)


# --- load_instance -------------------------------------------------------------


def test_load_instance_returns_populated_instance(dataset_dir: Path) -> None:
    inst = load_instance("demo_a", data_dir=dataset_dir)
    assert isinstance(inst, TSPInstance)
    assert inst.instance_id == "demo_a"
    assert inst.dimension == 3
    assert inst.coordinates == ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))


def test_load_instance_verifies_checksum_by_default(dataset_dir: Path) -> None:
    load_instance("demo_a", data_dir=dataset_dir)  # untouched file must not raise


def test_load_instance_raises_on_checksum_mismatch(dataset_dir: Path) -> None:
    target = dataset_dir / "demo_a.tsp"
    target.write_bytes(target.read_bytes().replace(b"10.0", b"11.0"))
    with pytest.raises(ChecksumError):
        load_instance("demo_a", data_dir=dataset_dir)


def test_load_instance_verify_false_skips_check(dataset_dir: Path) -> None:
    target = dataset_dir / "demo_a.tsp"
    target.write_bytes(target.read_bytes().replace(b"10.0", b"11.0"))
    inst = load_instance("demo_a", data_dir=dataset_dir, verify=False)
    assert inst.instance_id == "demo_a"


def test_load_instance_unknown_id_raises(dataset_dir: Path) -> None:
    with pytest.raises(UnknownInstanceError):
        load_instance("missing", data_dir=dataset_dir)


def test_load_instance_missing_file_raises(dataset_dir: Path) -> None:
    (dataset_dir / "demo_a.tsp").unlink()
    with pytest.raises(FileNotFoundError):
        load_instance("demo_a", data_dir=dataset_dir)


# --- load_optimal_tour ---------------------------------------------------------


def test_load_optimal_tour_returns_zero_indexed_permutation(dataset_dir: Path) -> None:
    tour = load_optimal_tour("demo_a", data_dir=dataset_dir)
    assert tour == [0, 1, 2]  # [1, 2, 3] -> 0-indexed
    assert sorted(tour) == [0, 1, 2]


def test_load_optimal_tour_raises_on_checksum_mismatch(dataset_dir: Path) -> None:
    target = dataset_dir / "demo_a.opt.tour"
    target.write_bytes(target.read_bytes().replace(b"DIMENSION : 3", b"DIMENSION : 9"))
    with pytest.raises(ChecksumError):
        load_optimal_tour("demo_a", data_dir=dataset_dir)
