import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..configs import DATA_DIR

KNAPSACK_DIR = DATA_DIR / "knapsack"


class ChecksumError(Exception):
    pass


class ManifestError(Exception):
    pass


class UnknownInstanceError(KeyError):
    pass


class OptimaError(Exception):
    pass


@dataclass(frozen=True)
class OptimumEntry:
    """One ``optima.json`` record: an instance's exact optimum, selection and file checksum."""

    instance_id: str
    optimum: int
    instance_checksum: str
    selection: tuple[int, ...]


@dataclass(frozen=True)
class KnapsackInstance:
    instance_id: str
    n: int
    R: int
    correlation_type: str
    values: tuple[int, ...]
    weights: tuple[int, ...]
    capacity: int
    seed: int


@dataclass(frozen=True)
class ManifestEntry:
    R: int
    capacity: int
    checksum: str
    correlation_type: str
    instance_id: str
    n: int
    seed: int


def _compute_sha256(data: bytes) -> str:
    """Return the ``"sha256:<hex>"`` digest of ``data``."""
    return "sha256:" + sha256(data).hexdigest()


def _validate_checksum(instance_data: bytes, expected_checksum: str) -> None:

    instance_checksum = _compute_sha256(instance_data)
    if instance_checksum != expected_checksum:
        raise ChecksumError(
            f"Checksum mismatch for instance. "
            f"Expected {expected_checksum}, got {instance_checksum}."
        )


def load_optima(data_dir: Path = KNAPSACK_DIR) -> list[OptimumEntry]:
    """Read ``optima.json`` and return one :class:`OptimumEntry` per instance.

    Validates the schema version and structure, raising :class:`OptimaError` on an
    unsupported version or a missing ``optima`` key.
    """
    with open(data_dir / "optima.json", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != 1:
        raise OptimaError(f"Unsupported optima schema version: {payload.get('schema_version')}")
    if "optima" not in payload:
        raise OptimaError("Optima file is missing 'optima' key.")
    return [
        OptimumEntry(
            instance_id=e["instance_id"],
            optimum=e["optimum"],
            instance_checksum=e["instance_checksum"],
            selection=tuple(e["selection"]),
        )
        for e in payload["optima"]
    ]


def load_instance(
    instance_id: str, *, data_dir: Path = KNAPSACK_DIR, verify: bool = True
) -> KnapsackInstance:
    instance_path = data_dir / f"{instance_id}.json"

    manifest_instances = load_manifest(data_dir)
    manifest_entry = next(
        (entry for entry in manifest_instances if entry.instance_id == instance_id), None
    )
    if manifest_entry is None:
        raise UnknownInstanceError(f"Instance {instance_id} not found in manifest.")

    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file {instance_path} does not exist.")

    instance_raw = instance_path.read_bytes()

    if verify:
        _validate_checksum(instance_raw, manifest_entry.checksum)

    instance_data = json.loads(instance_raw)

    return KnapsackInstance(
        instance_id=instance_id,
        n=instance_data["n"],
        R=instance_data["R"],
        correlation_type=instance_data["correlation_type"],
        values=tuple(instance_data["values"]),
        weights=tuple(instance_data["weights"]),
        capacity=instance_data["capacity"],
        seed=manifest_entry.seed,
    )


def load_manifest(data_dir: Path = KNAPSACK_DIR) -> list[ManifestEntry]:
    instances = []
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("schema_version") != 1:
        raise ManifestError(
            f"Unsupported manifest schema version: {manifest.get('schema_version')}"
        )

    if "instances" not in manifest:
        raise ManifestError("Manifest file is missing 'instances' key.")

    for entry in manifest["instances"]:
        mentry = ManifestEntry(
            R=entry["R"],
            capacity=entry["capacity"],
            checksum=entry["checksum"],
            correlation_type=entry["correlation_type"],
            instance_id=entry["instance_id"],
            n=entry["n"],
            seed=entry["metadata"]["seed"],
        )

        instances.append(mentry)
    return instances
