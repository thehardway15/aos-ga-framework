# Replication

End-to-end instructions for reproducing the experiments and the analysis.

> This is a skeleton; sections are completed as the corresponding layers land.

## Environment

- Python and pinned library versions: see `pyproject.toml`.
- Install: `pip install -e ".[dev,analysis]"`.

## Inputs

- Repetition seeds: `data/seeds/seeds.json`.
- TSPLIB instances: `data/tsplib/`.
- Knapsack dataset: `data/knapsack/` (+ `manifest.json`).

### Knapsack dataset provenance

The nine knapsack instances are produced with the external Pisinger instance
generator and copied into `data/knapsack/`. Record here the exact generator
version (tag) and the command used, so that the dataset can be regenerated
byte-for-byte and verified against the manifest checksums:

- Generator version: _to be recorded_
- Command: _to be recorded_
- Instance seeds: _to be recorded_

## Running the experiments

_To be completed: configurations, runner invocation, per-phase Git tags._

## Analysis

_To be completed: aggregation, statistics, tables and figures._
