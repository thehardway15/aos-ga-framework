"""Builder for the knapsack reference optima.

Run once to complete ``data/knapsack/``: solves each of the nine Pisinger instances
exactly with ``knapsack_dp``, verifies the optimal selection against
:class:`~experiments.problems.knapsack.KnapsackProblem`, and writes a canonical
``optima.json``. The optima live in a separate file so the R1-owned, checksummed
``manifest.json`` is never touched. Deterministic and idempotent; run with
``PYTHONPATH=src python -m experiments.datasets.knapsack_build``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..problems.knapsack import KnapsackProblem
from .exact_knapsack import knapsack_dp
from .knapsack import KNAPSACK_DIR, load_instance, load_manifest


def _canonical_json(payload: dict[str, object]) -> bytes:
    """Serialize like the rest of the datasets: sorted keys, 4-space indent, trailing newline."""
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=4) + "\n").encode(
        "utf-8"
    )


def build_optima(*, data_dir: Path = KNAPSACK_DIR) -> None:
    """Solve every instance in the manifest and write ``optima.json``.

    Instances are processed in sorted ``instance_id`` order; each optimum and its 0/1
    selection come from ``knapsack_dp`` and are cross-checked against the fitness
    function: the selection must reproduce the optimum under ``KnapsackProblem.evaluate``
    (hence be feasible and unpenalized) and the optimum must be positive, otherwise a
    ``ValueError`` is raised. Every record keeps the value, the selection and the
    instance-file checksum, binding the optimum to the exact bytes it was solved from.
    """
    records = []
    for entry in sorted(load_manifest(data_dir), key=lambda e: e.instance_id):
        instance = load_instance(entry.instance_id, data_dir=data_dir)
        optimum, selection = knapsack_dp(instance.values, instance.weights, instance.capacity)

        problem = KnapsackProblem(instance)
        if int(problem.evaluate(list(selection))) != optimum:
            raise ValueError(f"{entry.instance_id}: evaluate(selection) != optimum")
        if optimum <= 0:
            raise ValueError(f"{entry.instance_id}: non-positive optimum {optimum}")

        records.append(
            {
                "instance_id": entry.instance_id,
                "optimum": optimum,
                "instance_checksum": entry.checksum,
                "selection": list(selection),
            }
        )

    payload = {"schema_version": 1, "optima": records}
    (data_dir / "optima.json").write_bytes(_canonical_json(payload))


if __name__ == "__main__":
    build_optima()
