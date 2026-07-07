"""Contract tests for the study's repetition seeds.

These pin the public API of :mod:`experiments.datasets.seeds`: deterministic
derivation of the 30 paired repetition seeds from the project master seed, a
byte-exact ``seeds.json`` writer, and a fail-fast loader for the frozen artifact.
The seeds must be disjoint from the knapsack instance seeds, which live on a
separate branch of the same master ``SeedSequence``.

The module under test is not implemented yet: this file is the executable
specification of the contract. Expected public names: ``REPETITION_COUNT``,
``SCHEMA_VERSION``, ``SEEDS_PATH``, ``SeedsError``, ``repetition_seeds``,
``write_seeds_json``, ``load_repetition_seeds``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.configs import MASTER_SEED
from experiments.datasets.knapsack import KNAPSACK_DIR, load_manifest
from experiments.datasets.seeds import (
    REPETITION_COUNT,
    SCHEMA_VERSION,
    SEEDS_PATH,
    SeedsError,
    load_repetition_seeds,
    repetition_seeds,
    write_seeds_json,
)


def _valid_document() -> dict[str, Any]:
    """A structurally valid seeds document, for tampering in loader tests."""
    seeds = repetition_seeds()
    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": MASTER_SEED,
        "count": len(seeds),
        "seeds": seeds,
    }


def _write_document(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# --- repetition_seeds ----------------------------------------------------------


def test_repetition_seeds_returns_thirty_distinct_seeds() -> None:
    seeds = repetition_seeds()
    assert len(seeds) == REPETITION_COUNT == 30
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_repetition_seeds_is_deterministic() -> None:
    assert repetition_seeds() == repetition_seeds()


def test_distinct_master_seeds_yield_distinct_streams() -> None:
    assert repetition_seeds(master_seed=1) != repetition_seeds(master_seed=2)


def test_count_controls_the_number_of_seeds() -> None:
    assert len(repetition_seeds(count=5)) == 5


def test_repetition_seeds_are_disjoint_from_the_instance_seeds() -> None:
    # The two seed families are spawned from sibling branches of one master seed
    # precisely so they never overlap; assert that against the frozen dataset.
    if not (KNAPSACK_DIR / "manifest.json").exists():
        pytest.skip("knapsack dataset not generated yet; build it per replication/README.md")
    instance_seeds = {entry.seed for entry in load_manifest()}
    assert set(repetition_seeds()).isdisjoint(instance_seeds)


# --- write_seeds_json ----------------------------------------------------------


def test_write_seeds_json_writes_the_expected_document(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    write_seeds_json(path=path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    document = json.loads(raw)
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["master_seed"] == MASTER_SEED
    assert document["count"] == REPETITION_COUNT
    assert document["seeds"] == repetition_seeds()


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    write_seeds_json(path=path)
    assert load_repetition_seeds(path) == repetition_seeds()


def test_write_seeds_json_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_seeds_json(path=first)
    write_seeds_json(path=second)
    assert first.read_bytes() == second.read_bytes()


# --- load_repetition_seeds -----------------------------------------------------


def test_load_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    document = _valid_document()
    document["schema_version"] = SCHEMA_VERSION + 1
    path = tmp_path / "seeds.json"
    _write_document(path, document)
    with pytest.raises(SeedsError):
        load_repetition_seeds(path)


def test_load_rejects_a_count_that_disagrees_with_the_seeds(tmp_path: Path) -> None:
    document = _valid_document()
    document["count"] = document["count"] + 1
    path = tmp_path / "seeds.json"
    _write_document(path, document)
    with pytest.raises(SeedsError):
        load_repetition_seeds(path)


def test_load_rejects_a_missing_seeds_key(tmp_path: Path) -> None:
    document = _valid_document()
    del document["seeds"]
    path = tmp_path / "seeds.json"
    _write_document(path, document)
    with pytest.raises(SeedsError):
        load_repetition_seeds(path)


# --- frozen artifact -----------------------------------------------------------


def test_committed_seeds_match_the_derivation() -> None:
    # Once generated and committed, the artifact must equal a fresh derivation;
    # skip until it has been written so the suite is green before that step.
    if not SEEDS_PATH.exists():
        pytest.skip("seeds.json not generated yet; create it with write_seeds_json")
    assert load_repetition_seeds() == repetition_seeds()
