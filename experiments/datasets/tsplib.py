"""Reader for the study's TSPLIB dataset.

Parses TSPLIB instance (``.tsp``/``.vrp``) and optimal-tour (``.opt.tour``) files --
``EUC_2D`` only -- and loads them against a checksummed manifest, mirroring the
knapsack loader. The ``parse_*`` functions are pure byte-level parsers; the
``load_*`` functions add integrity verification for the frozen artifacts under
``data/tsplib/``.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.configs import DATA_DIR

TSPLIB_DIR = DATA_DIR / "tsplib"


@dataclass(frozen=True)
class TSPInstance:
    """A parsed TSPLIB instance: id, size, edge-weight type and 0-indexed coordinates."""

    instance_id: str
    dimension: int
    edge_weight_type: str
    coordinates: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TSPManifestEntry:
    """One manifest record: instance metadata, optimal length and the file checksums."""

    instance_id: str
    dimension: int
    edge_weight_type: str
    optimal_length: int
    checksum: str
    opt_tour_checksum: str
    source: str


class UnsupportedEdgeWeightError(Exception):
    """Raised for any ``EDGE_WEIGHT_TYPE`` other than ``EUC_2D``."""


class TSPLIBParseError(Exception):
    """Raised on a malformed TSPLIB instance or tour file."""


class ChecksumError(Exception):
    """Raised when a file's SHA-256 digest does not match the manifest."""


class ManifestError(Exception):
    """Raised on a missing or structurally invalid manifest."""


class UnknownInstanceError(KeyError):
    """Raised when an instance id is not present in the manifest."""


def parse_tsplib(raw: bytes) -> TSPInstance:
    """Parse the bytes of a TSPLIB ``.tsp``/``.vrp`` file into a :class:`TSPInstance`.

    Reads the header and ``NODE_COORD_SECTION``, tolerating both ``KEY :`` and
    ``KEY:`` spacing and integer or float coordinates; unknown sections (e.g. VRP
    demands) are ignored. Raises :class:`UnsupportedEdgeWeightError` for a non-EUC_2D
    type and :class:`TSPLIBParseError` on malformed input.
    """
    lines = raw.decode("utf-8").splitlines()

    for i, line in enumerate(lines):
        if line.startswith("NAME"):
            instance_id = line.split(":", 1)[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip()
            if edge_weight_type != "EUC_2D":
                raise UnsupportedEdgeWeightError(
                    f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}"
                )
        elif line.startswith("NODE_COORD_SECTION"):
            coords: list[tuple[float, float]] = [(0.0, 0.0)] * dimension
            for j in range(i + 1, i + 1 + dimension):
                parts = lines[j].split()
                if len(parts) < 3:
                    raise TSPLIBParseError(
                        f"Expected 3 parts for coordinate, got {len(parts)}: {lines[j]}"
                    )
                coords[int(parts[0]) - 1] = (float(parts[1]), float(parts[2]))

            return TSPInstance(
                instance_id=instance_id,
                dimension=dimension,
                edge_weight_type=edge_weight_type,
                coordinates=tuple(coords),
            )
    raise TSPLIBParseError("NODE_COORD_SECTION not found or incomplete in the TSPLIB file.")


def parse_opt_tour(raw: bytes) -> list[int]:
    """Parse a TSPLIB ``.opt.tour`` file into a 0-indexed permutation.

    Reads ``TOUR_SECTION`` up to the ``-1`` terminator, mapping 1-indexed cities to
    0-indexed. Raises :class:`TSPLIBParseError` if the result is not a permutation.
    """
    lines = raw.decode("utf-8").splitlines()
    tour_section_found = False
    tour = []
    for line in lines:
        if line.startswith("TOUR_SECTION"):
            tour_section_found = True
            continue
        if tour_section_found:
            if line.strip() == "-1":
                break
            tour.append(int(line.strip()) - 1)
    if not tour_section_found:
        raise TSPLIBParseError("TOUR_SECTION not found in the .opt.tour file.")

    if sorted(tour) != list(range(len(tour))):
        raise TSPLIBParseError("TOUR_SECTION is not a permutation")

    return tour


def _compute_sha256(data: bytes) -> str:
    """Return the ``"sha256:<hex>"`` digest of ``data``."""
    return "sha256:" + sha256(data).hexdigest()


def _validate_checksum(instance_data: bytes, expected_checksum: str) -> None:
    """Raise :class:`ChecksumError` if ``instance_data``'s digest differs from the expected one."""
    instance_checksum = _compute_sha256(instance_data)
    if instance_checksum != expected_checksum:
        raise ChecksumError(
            f"Checksum mismatch for instance. "
            f"Expected {expected_checksum}, got {instance_checksum}."
        )


def load_manifest(data_dir: Path = TSPLIB_DIR) -> list[TSPManifestEntry]:
    """Load and validate ``manifest.json`` into a list of :class:`TSPManifestEntry`."""
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
        mentry = TSPManifestEntry(
            instance_id=entry["instance_id"],
            dimension=entry["dimension"],
            edge_weight_type=entry["edge_weight_type"],
            checksum=entry["checksum"],
            opt_tour_checksum=entry["opt_tour_checksum"],
            optimal_length=entry["optimal_length"],
            source=entry["source"],
        )

        instances.append(mentry)
    return instances


def _read_verified_file(
    instance_id: str,
    ext: str,
    checksum_property: str,
    data_dir: Path = TSPLIB_DIR,
    verify: bool = True,
) -> bytes:
    """Return the bytes of ``{instance_id}.{ext}``, checksum-verified against the manifest.

    Raises :class:`UnknownInstanceError` if the id is absent from the manifest,
    ``FileNotFoundError`` if the file is missing, and :class:`ChecksumError` on a
    digest mismatch (when ``verify`` is set).
    """
    instance_path = data_dir / f"{instance_id}.{ext}"
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
        _validate_checksum(instance_raw, getattr(manifest_entry, checksum_property))

    return instance_raw


def load_instance(
    instance_id: str, *, data_dir: Path = TSPLIB_DIR, verify: bool = True
) -> TSPInstance:
    """Load a TSPLIB instance by id, checksum-verified against the manifest."""
    instance_raw = _read_verified_file(
        instance_id, "tsp", "checksum", data_dir=data_dir, verify=verify
    )

    return parse_tsplib(instance_raw)


def load_optimal_tour(
    instance_id: str, *, data_dir: Path = TSPLIB_DIR, verify: bool = True
) -> list[int]:
    """Load an instance's optimal tour (0-indexed), checksum-verified against the manifest."""
    instance_raw = _read_verified_file(
        instance_id, "opt.tour", "opt_tour_checksum", data_dir=data_dir, verify=verify
    )

    return parse_opt_tour(instance_raw)
