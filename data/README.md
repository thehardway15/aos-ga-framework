# Data

Versioned, byte-exact input artifacts for the experiments. End-of-line
conversion is disabled for this tree (see `.gitattributes`) so that checksums
are identical across operating systems.

## `seeds/`

`seeds.json` — the 30 paired repetition seeds shared by every configuration,
derived deterministically from the project master seed.

## `tsplib/`

The TSPLIB instances used by the TSP problems: `eil22`, `eil51`, `berlin52`.

## `knapsack/`

The nine 0/1 knapsack instances (`n ∈ {20, 30, 50}` × three correlation classes)
and their `manifest.json`. The instances are produced once with the external
Pisinger instance generator and copied here as a frozen, checksum-verified
artifact; the generator version and the exact command used to produce them are
recorded in [`../replication/README.md`](../replication/README.md). They are
never regenerated from within this repository — integrity is verified against
the manifest checksums.
