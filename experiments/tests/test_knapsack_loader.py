"""Contract tests for the knapsack dataset loader.

These exercise ``experiments.datasets.knapsack`` against a synthetic dataset
built in a temporary directory in the exact on-disk shape produced by the R1
generator: canonical JSON with sorted keys, 4-space indent, a trailing newline
and UTF-8 bytes (no newline translation). They never import R1 and never touch
the real ``data/knapsack/`` tree, so they pin the loader's behaviour
independently of the frozen artifact.

The module these tests import does not exist yet: this file is the executable
specification of the loader's public API. Expected public names:
``KnapsackInstance``, ``ManifestEntry``, ``load_manifest``, ``load_instance``,
``ChecksumError``, ``UnknownInstanceError``, ``ManifestError``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.datasets.knapsack import (
    ChecksumError,
    KnapsackInstance,
    ManifestEntry,
    ManifestError,
    UnknownInstanceError,
    load_instance,
    load_manifest,
)


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize like the R1 generator: sorted keys, 4-space indent, trailing newline."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=4) + "\n"
    return text.encode("utf-8")


def _instance_payload(
    instance_id: str,
    *,
    n: int,
    correlation_type: str,
    seed: int,
    values: list[int],
    weights: list[int],
    R: int = 1000,
) -> dict[str, Any]:
    """Build one instance mapping in the R1 file schema.

    The capacity follows the "50% knapsack" rule, ``W = floor(0.5 * sum(weights))``,
    so the fixture is internally consistent with the control field the loader
    surfaces.
    """
    return {
        "R": R,
        "capacity": sum(weights) // 2,
        "correlation_type": correlation_type,
        "metadata": {"instance_id": instance_id, "seed": seed},
        "n": n,
        "schema_version": 1,
        "values": values,
        "weights": weights,
    }


# Two fixed demo instances: small, but structurally identical to real files.
_DEMO: dict[str, dict[str, Any]] = {
    "kp_demo_a": _instance_payload(
        "kp_demo_a",
        n=4,
        correlation_type="uncorrelated",
        seed=111,
        values=[5, 9, 2, 7],
        weights=[10, 20, 30, 41],
    ),
    "kp_demo_b": _instance_payload(
        "kp_demo_b",
        n=3,
        correlation_type="strongly",
        seed=222,
        values=[200, 300, 400],
        weights=[100, 200, 300],
    ),
}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write the demo instances and an R1-style manifest into a temp directory."""
    entries: list[dict[str, Any]] = []
    for instance_id, payload in _DEMO.items():
        raw = _canonical_bytes(payload)
        (tmp_path / f"{instance_id}.json").write_bytes(raw)
        entries.append(
            {
                "R": payload["R"],
                "capacity": payload["capacity"],
                "checksum": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "correlation_type": payload["correlation_type"],
                "instance_id": instance_id,
                "metadata": payload["metadata"],
                "n": payload["n"],
            }
        )
    manifest = {"instances": entries, "schema_version": 1}
    (tmp_path / "manifest.json").write_bytes(_canonical_bytes(manifest))
    return tmp_path


def test_load_manifest_returns_one_entry_per_instance(dataset_dir: Path) -> None:
    entries = load_manifest(dataset_dir)
    assert {e.instance_id for e in entries} == set(_DEMO)
    assert all(isinstance(e, ManifestEntry) for e in entries)


def test_load_manifest_extracts_fields_including_nested_seed(dataset_dir: Path) -> None:
    by_id = {e.instance_id: e for e in load_manifest(dataset_dir)}
    a = by_id["kp_demo_a"]
    assert a.n == 4
    assert a.correlation_type == "uncorrelated"
    assert a.capacity == sum(_DEMO["kp_demo_a"]["weights"]) // 2
    assert a.seed == 111  # lifted out of the nested metadata.seed
    assert a.checksum.startswith("sha256:")


def test_load_instance_returns_populated_instance(dataset_dir: Path) -> None:
    inst = load_instance("kp_demo_b", data_dir=dataset_dir)
    assert isinstance(inst, KnapsackInstance)
    assert inst.instance_id == "kp_demo_b"
    assert inst.n == 3
    assert inst.correlation_type == "strongly"
    assert inst.seed == 222
    assert inst.values == (200, 300, 400)
    assert inst.weights == (100, 200, 300)
    assert inst.capacity == 300
    assert len(inst.values) == inst.n
    assert len(inst.weights) == inst.n


def test_load_instance_verifies_checksum_by_default(dataset_dir: Path) -> None:
    # Happy path: an untouched file must load without raising.
    load_instance("kp_demo_a", data_dir=dataset_dir)


def test_load_instance_raises_on_checksum_mismatch(dataset_dir: Path) -> None:
    target = dataset_dir / "kp_demo_a.json"
    tampered = target.read_bytes().replace(b'"seed": 111', b'"seed": 999')
    assert tampered != target.read_bytes()
    target.write_bytes(tampered)
    with pytest.raises(ChecksumError):
        load_instance("kp_demo_a", data_dir=dataset_dir)


def test_load_instance_verify_false_skips_check(dataset_dir: Path) -> None:
    target = dataset_dir / "kp_demo_a.json"
    target.write_bytes(target.read_bytes().replace(b'"seed": 111', b'"seed": 999'))
    # With verification disabled the loader trusts the bytes and does not raise.
    inst = load_instance("kp_demo_a", data_dir=dataset_dir, verify=False)
    assert inst.instance_id == "kp_demo_a"


def test_load_instance_unknown_id_raises(dataset_dir: Path) -> None:
    with pytest.raises(UnknownInstanceError):
        load_instance("kp_missing", data_dir=dataset_dir)


def test_load_instance_missing_file_raises(dataset_dir: Path) -> None:
    # Entry present in the manifest but the file is gone -> fail fast.
    (dataset_dir / "kp_demo_a.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_instance("kp_demo_a", data_dir=dataset_dir)


def test_load_manifest_malformed_raises_manifest_error(tmp_path: Path) -> None:
    # A manifest missing the "instances" key is structurally invalid.
    (tmp_path / "manifest.json").write_bytes(b'{"schema_version": 1}\n')
    with pytest.raises(ManifestError):
        load_manifest(tmp_path)
