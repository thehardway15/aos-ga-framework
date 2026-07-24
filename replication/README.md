# Replication

End-to-end instructions for reproducing the experiments and the analysis.

> This is a skeleton; sections are completed as the corresponding layers land.

## Environment

Two different things are pinned, for two different purposes:

- **Compatibility** — `pyproject.toml` declares version *ranges*. They are what
  the CI matrix exercises across Python 3.10–3.13, and what a reader who only
  wants to run the code should install: `pip install -e ".[dev,analysis]"`.
- **Reproducibility** — `requirements-lock.txt` records the *exact* versions
  (CPython 3.13.0, NumPy 2.5.0, …) that produced every versioned artifact under
  `results/`. Use it whenever the recorded numbers themselves matter:

  ```bash
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install -r replication/requirements-lock.txt
  pip install -e . --no-deps
  ```

The distinction is not bookkeeping. NumPy guarantees a stable random stream only
for the legacy `RandomState`; a `numpy.random.Generator` may draw differently
after a feature release (NEP 19). Identical code and identical seeds therefore
reproduce the recorded values exactly only against the pinned NumPy — across
versions the *procedure* replicates, the individual numbers need not.

## Inputs

- Repetition seeds: `data/seeds/seeds.json`.
- TSPLIB instances: `data/tsplib/`.
- Knapsack dataset: `data/knapsack/` (+ `manifest.json`).

### Knapsack dataset provenance

The nine knapsack instances under `data/knapsack/` are produced **once**,
externally, with the Pisinger instance generator and committed here as a frozen,
byte-exact artifact. This repository has no code dependency on the generator and
ships no build script; reproducibility rests on the pinned generator version,
the exact commands, the recorded seeds, and the manifest checksums.

**Generator:** `pisinger-knapsack`, tag `v0.2.0`.

**Seed derivation.** Instance seeds are derived deterministically from the
project master seed `20260101`, on a branch that is disjoint from the repetition
seeds:

```python
import numpy as np

root = np.random.SeedSequence(20260101)
instances_branch, repetitions_branch = root.spawn(2)   # [1] feeds data/seeds/seeds.json
children = instances_branch.spawn(9)
seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]
```

Each spawned child seeds exactly one instance, in the following fixed order:

| spawn child | instance_id        | n  | correlation  | seed (uint32) |
|-------------|--------------------|----|--------------|---------------|
| 0           | `n20_uncorrelated` | 20 | uncorrelated | 697299274     |
| 1           | `n30_uncorrelated` | 30 | uncorrelated | 3064857378    |
| 2           | `n50_uncorrelated` | 50 | uncorrelated | 3207017045    |
| 3           | `n50_weakly`       | 50 | weakly       | 502192065     |
| 4           | `n30_weakly`       | 30 | weakly       | 1553210182    |
| 5           | `n20_weakly`       | 20 | weakly       | 1606843269    |
| 6           | `n20_strongly`     | 20 | strongly     | 2577246761    |
| 7           | `n30_strongly`     | 30 | strongly     | 972143592     |
| 8           | `n50_strongly`     | 50 | strongly     | 68504625      |

The seeds are also recorded per instance in `data/knapsack/manifest.json`
(`metadata.seed`); the table above is the authoritative mapping for regeneration.

**Build commands.** With `pisinger-knapsack` v0.2.0 installed, run from the
repository root (`R = 1000` and capacity `W = floor(0.5 * sum(weights))` are
intrinsic to the generator):

```bash
pisinger-knapsack generate --n 20 --correlation uncorrelated --seed 697299274  --R 1000 --out data/knapsack/n20_uncorrelated.json
pisinger-knapsack generate --n 30 --correlation uncorrelated --seed 3064857378 --R 1000 --out data/knapsack/n30_uncorrelated.json
pisinger-knapsack generate --n 50 --correlation uncorrelated --seed 3207017045 --R 1000 --out data/knapsack/n50_uncorrelated.json
pisinger-knapsack generate --n 50 --correlation weakly       --seed 502192065  --R 1000 --out data/knapsack/n50_weakly.json
pisinger-knapsack generate --n 30 --correlation weakly       --seed 1553210182 --R 1000 --out data/knapsack/n30_weakly.json
pisinger-knapsack generate --n 20 --correlation weakly       --seed 1606843269 --R 1000 --out data/knapsack/n20_weakly.json
pisinger-knapsack generate --n 20 --correlation strongly     --seed 2577246761 --R 1000 --out data/knapsack/n20_strongly.json
pisinger-knapsack generate --n 30 --correlation strongly     --seed 972143592  --R 1000 --out data/knapsack/n30_strongly.json
pisinger-knapsack generate --n 50 --correlation strongly     --seed 68504625   --R 1000 --out data/knapsack/n50_strongly.json

pisinger-knapsack manifest build --dir data/knapsack --out data/knapsack/manifest.json
```

**Verification.** The dataset can be re-checked against the recorded checksums
either with the generator (`pisinger-knapsack manifest verify --dir data/knapsack
--manifest data/knapsack/manifest.json`) or, without any dependency on the
generator, with this repository's loader and integrity tests (`pytest
experiments/tests`).

**Idempotency.** The same generator tag (`v0.2.0`), the same seeds and the same
`--R 1000` reproduce the instances byte-for-byte, and therefore the same
checksums. This is verified manually, not automated inside this repository.

## Determinism and randomness

Every run is fully determined by its seed. Framework code draws only from a
single injected `numpy.random.Generator`, created per run from one of the
repetition seeds. There is **no exception**: Python's `random` module is never
seeded or read, and NumPy's legacy global state is never touched. This is why the
GA loop, the selection and the operators are written in-house rather than taken
from an established EC library — those drive their operators from the
process-global `random` module, which would have made the guarantee conditional
and forced per-process isolation on the runner.

Seeds are spawned from the master seed on `numpy.random.SeedSequence` branches:
one branch feeds the knapsack instance seeds, a disjoint branch feeds the 30
repetition seeds in `data/seeds/seeds.json`. The repetition seeds are shared by
every configuration and every level (CGA, FBO, Random, AOS), so the blocks are
paired across configurations.

The limit of the guarantee is the NumPy version, not the code: see
[Environment](#environment).

## Running the experiments

_To be completed: configurations, runner invocation, per-phase Git tags._

## Analysis

_To be completed: aggregation, statistics, tables and figures._
