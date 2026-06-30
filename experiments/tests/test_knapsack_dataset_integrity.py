"""Integrity tests for the frozen knapsack dataset under ``data/knapsack/``.

Unlike the loader contract tests, these run against the real, versioned
artifact. They are skipped with a clear message until the dataset has been
generated (see ``replication/README.md``); in CI's study job, once the dataset
is in place, they must pass. No R1 import is involved -- integrity is checked
purely against the manifest checksums recorded alongside the instances.
"""

from __future__ import annotations

import sys

import pytest

from experiments.datasets.knapsack import (
    KNAPSACK_DIR,
    load_instance,
    load_manifest,
)

# The nine instances this study owns: instance_id -> (n, correlation_type).
# Ids follow the pattern n{size}_{correlation}, where the correlation segment is
# R1's canonical label verbatim (uncorrelated / weakly / strongly).
EXPECTED_INSTANCES: dict[str, tuple[int, str]] = {
    "n20_uncorrelated": (20, "uncorrelated"),
    "n20_weakly": (20, "weakly"),
    "n20_strongly": (20, "strongly"),
    "n30_uncorrelated": (30, "uncorrelated"),
    "n30_weakly": (30, "weakly"),
    "n30_strongly": (30, "strongly"),
    "n50_uncorrelated": (50, "uncorrelated"),
    "n50_weakly": (50, "weakly"),
    "n50_strongly": (50, "strongly"),
}


@pytest.fixture(autouse=True)
def _require_dataset() -> None:
    """Skip the whole module unless the frozen dataset has been generated."""
    if not (KNAPSACK_DIR / "manifest.json").exists():
        pytest.skip("knapsack dataset not generated yet; build it per replication/README.md")


def test_manifest_lists_expected_nine_instances() -> None:
    by_id = {e.instance_id: e for e in load_manifest()}
    assert set(by_id) == set(EXPECTED_INSTANCES)
    for instance_id, (n, correlation_type) in EXPECTED_INSTANCES.items():
        assert by_id[instance_id].n == n
        assert by_id[instance_id].correlation_type == correlation_type


def test_every_instance_verifies_against_manifest() -> None:
    for instance_id in EXPECTED_INSTANCES:
        load_instance(instance_id)  # raises on checksum mismatch


def test_capacity_is_floor_half_sum_of_weights() -> None:
    for instance_id in EXPECTED_INSTANCES:
        inst = load_instance(instance_id)
        assert inst.capacity == sum(inst.weights) // 2


def test_weights_and_values_are_well_formed() -> None:
    for instance_id, (n, _correlation_type) in EXPECTED_INSTANCES.items():
        inst = load_instance(instance_id)
        assert len(inst.weights) == n
        assert len(inst.values) == n
        assert all(1 <= w <= inst.R for w in inst.weights)
        assert all(v >= 1 for v in inst.values)


def test_instance_seeds_are_distinct() -> None:
    # Each of the nine instances must be an independent draw. A single shared
    # seed makes the weakly/strongly instances of the same n share their entire
    # weight vector and nests smaller n inside larger n -- a hidden artifact we
    # explicitly reject by deriving nine seeds from instances_branch.spawn(9).
    seeds = [e.seed for e in load_manifest()]
    assert len(seeds) == len(set(seeds))


def test_loader_does_not_import_r1() -> None:
    # Regression guard for the "no R1 dependency" rule. R1's package is not a
    # declared dependency of R2, so loading the dataset must not pull it in.
    load_manifest()
    load_instance(next(iter(EXPECTED_INSTANCES)))
    assert "pisinger_knapsack" not in sys.modules
