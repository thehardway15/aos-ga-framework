"""Contract tests for the knapsack optima builder and its loader.

These pin the offline build step that fixes the reference optima of the nine
Pisinger knapsack instances. ``build_optima`` reads the checksummed dataset,
solves each instance exactly with ``knapsack_dp``, verifies the result against
:class:`~experiments.problems.knapsack.KnapsackProblem`, and writes a canonical
``optima.json`` artifact; ``load_optima`` reads that artifact back. Unlike the
TSP optima, these live in a *separate* file so the R1-owned, checksummed
``manifest.json`` (schema_version 1) is never touched.

The names these tests import do not all exist yet: this file is the executable
specification. Expected public names: ``build_optima`` (in
``experiments.datasets.knapsack_build``); ``OptimumEntry``, ``load_optima`` and
``OptimaError`` (in ``experiments.datasets.knapsack``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.datasets.knapsack import (
    OptimaError,
    OptimumEntry,
    load_instance,
    load_optima,
)
from experiments.datasets.knapsack_build import build_optima
from experiments.problems.knapsack import KnapsackProblem

# id -> (values, weights, capacity, correlation_type, seed)
_InstanceSpec = tuple[list[int], list[int], int, str, int]

# Two small instances with hand-computed, *unique* optimal subsets:
#   kp_a: items {1,2} weigh 50 and are worth 220 (the classic 0/1 example).
#   kp_b: items {1,3} weigh 7 <= 10 and are worth 90.
_SPECS: dict[str, _InstanceSpec] = {
    "kp_a": ([60, 100, 120], [10, 20, 30], 50, "uncorrelated", 11),
    "kp_b": ([10, 40, 30, 50], [5, 4, 6, 3], 10, "strongly", 22),
}
_KNOWN_OPTIMA = {"kp_a": 220, "kp_b": 90}
_KNOWN_SELECTION = {"kp_a": (0, 1, 1), "kp_b": (0, 1, 0, 1)}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize like the rest of the datasets: sorted keys, 4-space indent, trailing newline."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=4) + "\n"
    return text.encode("utf-8")


def _write_dataset(root: Path, specs: dict[str, _InstanceSpec]) -> None:
    """Write instance files and an R1-style checksummed manifest into ``root``."""
    entries: list[dict[str, Any]] = []
    for instance_id, (values, weights, capacity, correlation_type, seed) in specs.items():
        payload = {
            "R": 1000,
            "capacity": capacity,
            "correlation_type": correlation_type,
            "metadata": {"instance_id": instance_id, "seed": seed},
            "n": len(values),
            "schema_version": 1,
            "values": values,
            "weights": weights,
        }
        raw = _canonical_bytes(payload)
        (root / f"{instance_id}.json").write_bytes(raw)
        entries.append(
            {
                "R": 1000,
                "capacity": capacity,
                "checksum": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "correlation_type": correlation_type,
                "instance_id": instance_id,
                "metadata": {"seed": seed},
                "n": len(values),
            }
        )
    manifest = {"instances": entries, "schema_version": 1}
    (root / "manifest.json").write_bytes(_canonical_bytes(manifest))


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A temp dataset with the two demo instances and their R1-style manifest."""
    _write_dataset(tmp_path, _SPECS)
    return tmp_path


# --- build_optima writes the artifact ------------------------------------------


def test_build_writes_optima_file(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    assert (dataset_dir / "optima.json").exists()


def test_build_leaves_the_manifest_untouched(dataset_dir: Path) -> None:
    # The R1-owned, checksummed manifest must not be rewritten by the R2 build.
    before = (dataset_dir / "manifest.json").read_bytes()
    build_optima(data_dir=dataset_dir)
    assert (dataset_dir / "manifest.json").read_bytes() == before


# --- load_optima round-trip ----------------------------------------------------


def test_load_optima_returns_one_entry_per_instance(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    entries = load_optima(dataset_dir)
    assert {e.instance_id for e in entries} == set(_SPECS)
    assert all(isinstance(e, OptimumEntry) for e in entries)


def test_optima_match_known_values(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    by_id = {e.instance_id: e for e in load_optima(dataset_dir)}
    assert by_id["kp_a"].optimum == _KNOWN_OPTIMA["kp_a"]
    assert by_id["kp_b"].optimum == _KNOWN_OPTIMA["kp_b"]


def test_stored_selection_matches_the_unique_optimum(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    by_id = {e.instance_id: e for e in load_optima(dataset_dir)}
    assert by_id["kp_a"].selection == _KNOWN_SELECTION["kp_a"]
    assert by_id["kp_b"].selection == _KNOWN_SELECTION["kp_b"]


# --- cross-check against the fitness function ----------------------------------


def test_stored_selection_is_feasible_and_evaluates_to_optimum(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    by_id = {e.instance_id: e for e in load_optima(dataset_dir)}
    for instance_id in _SPECS:
        entry = by_id[instance_id]
        instance = load_instance(instance_id, data_dir=dataset_dir)
        # Feasible: total weight within capacity, so no big-M penalty applies.
        total_weight = sum(w * x for w, x in zip(instance.weights, entry.selection, strict=True))
        assert total_weight <= instance.capacity
        # The problem's own fitness of the optimal selection equals the stored optimum.
        problem = KnapsackProblem(instance)
        assert problem.evaluate(list(entry.selection)) == entry.optimum


# --- integrity: optimum bound to the instance file -----------------------------


def test_instance_checksum_binds_optimum_to_the_file(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    by_id = {e.instance_id: e for e in load_optima(dataset_dir)}
    raw = (dataset_dir / "kp_a.json").read_bytes()
    assert by_id["kp_a"].instance_checksum == "sha256:" + hashlib.sha256(raw).hexdigest()


# --- determinism / idempotence -------------------------------------------------


def test_build_is_idempotent(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    first = (dataset_dir / "optima.json").read_bytes()
    build_optima(data_dir=dataset_dir)
    second = (dataset_dir / "optima.json").read_bytes()
    assert first == second


def test_optima_are_sorted_by_instance_id(dataset_dir: Path) -> None:
    build_optima(data_dir=dataset_dir)
    ids = [e.instance_id for e in load_optima(dataset_dir)]
    assert ids == sorted(ids)


# --- guard: undefined gap ------------------------------------------------------


def test_build_rejects_a_zero_optimum_instance(tmp_path: Path) -> None:
    # Every item is heavier than the capacity -> optimum 0 -> gap would divide by zero.
    _write_dataset(tmp_path, {"kp_zero": ([5, 6], [10, 20], 4, "uncorrelated", 33)})
    with pytest.raises(ValueError):
        build_optima(data_dir=tmp_path)


# --- loader validates the schema -----------------------------------------------


def test_load_optima_rejects_unsupported_schema(tmp_path: Path) -> None:
    (tmp_path / "optima.json").write_bytes(_canonical_bytes({"optima": [], "schema_version": 2}))
    with pytest.raises(OptimaError):
        load_optima(tmp_path)
