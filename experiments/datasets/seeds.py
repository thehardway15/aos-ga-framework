"""Repetition seeds for the study.

The 30 paired repetition seeds shared by every configuration, derived
deterministically from the project master seed. This module owns the
study-specific values and the ``data/seeds/seeds.json`` artifact; the generic
seed-derivation plumbing lives in :mod:`aos_ga.rng`. The seeds are spawned from
the repetitions branch of the master sequence, disjoint from the branch that
seeds the knapsack instances, and are shared across configurations so the
Friedman blocks are paired.
"""

import json
from pathlib import Path

import numpy as np

from aos_ga.rng import spawn_seeds
from experiments.configs import DATA_DIR, MASTER_SEED

REPETITION_COUNT = 30
REPETITION_BRANCH_INDEX = 1
SCHEMA_VERSION = 1
SEEDS_PATH = DATA_DIR / "seeds" / "seeds.json"


class SeedsError(Exception):
    """Raised when ``seeds.json`` is missing, malformed, or fails validation."""


def load_repetition_seeds(path: Path = SEEDS_PATH) -> list[int]:
    """Load the repetition seeds from a JSON file.

    Raises:
        SeedsError: If the file is missing, malformed, or has an unexpected schema.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SeedsError(f"Seeds file not found at {path}") from e
    except json.JSONDecodeError as e:
        raise SeedsError(f"Seeds file at {path} is not valid JSON") from e

    if not isinstance(document, dict):
        raise SeedsError(f"Seeds file at {path} does not contain a JSON object")

    schema_version = document.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SeedsError(
            f"Seeds file at {path} has schema version {schema_version}, expected {SCHEMA_VERSION}"
        )

    master_seed = document.get("master_seed")
    if master_seed != MASTER_SEED:
        raise SeedsError(
            f"Seeds file at {path} has master seed {master_seed}, expected {MASTER_SEED}"
        )

    count = document.get("count")
    if count != REPETITION_COUNT:
        raise SeedsError(f"Seeds file at {path} has count {count}, expected {REPETITION_COUNT}")

    seeds = document.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != REPETITION_COUNT
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise SeedsError(f"Seeds file at {path} has invalid seeds list: {seeds}")

    return seeds


def repetition_seeds(master_seed: int = MASTER_SEED, count: int = REPETITION_COUNT) -> list[int]:
    """Derive the ``count`` repetition seeds deterministically from ``master_seed``.

    Splits the master ``SeedSequence`` into two branches and spawns the seeds from
    the repetitions branch (``REPETITION_BRANCH_INDEX``), keeping them disjoint
    from the instance seeds on the other branch.
    """
    root = np.random.SeedSequence(master_seed)
    branches = root.spawn(2)
    repetition_branch = branches[REPETITION_BRANCH_INDEX]
    return spawn_seeds(repetition_branch, count)


def write_seeds_json(path: Path = SEEDS_PATH) -> None:
    """Generate the repetition seeds and write them to ``path`` as JSON.

    Writes a byte-stable document (``schema_version``, ``master_seed``, ``count``,
    ``seeds``) with a trailing newline, creating the parent directory if needed.
    Run once to (re)generate the committed ``seeds.json`` artifact.
    """
    seeds = repetition_seeds()
    document = {
        "schema_version": SCHEMA_VERSION,
        "master_seed": MASTER_SEED,
        "count": len(seeds),
        "seeds": seeds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_seeds_json()
