# aos-ga

Adaptive operator selection (AOS) for genetic algorithms under tight evaluation
budgets (20–50 generations).

`aos-ga` is a **problem-agnostic** framework for studying how a genetic
algorithm should choose, on the fly, which variation operator to apply. It
provides the engine (operators, GA loop, AOS strategies, credit assignment), the
experiment infrastructure (configuration, runner, recording) and the analysis
toolkit (metrics, statistics, visualization). Everything specific to a
particular study — concrete problems, datasets and baselines — lives separately
in the `experiments` research layer, which depends on the framework but is not
part of the distributable package.

## Repository layout

```
src/aos_ga/     framework (the only part packaged for distribution)
  core/         problem and operator interfaces, g(x), representations, GA engine
  operators/    generic operators (permutation, binary, real)
  variation/    variation steps: canonical pipeline, single, random, adaptive
  aos/          AOS strategy interface and its implementations
  credit/       credit assignment
  rng.py        seeded generators; no global random state anywhere
experiments/    the study: concrete problems, baselines, strategies, configs, datasets
data/           versioned inputs (seeds, TSPLIB, knapsack dataset + manifest)
results/        outputs (aggregated tables, figures; raw runs are git-ignored)
replication/    end-to-end reproduction instructions and the pinned environment
```

Several packages under `src/aos_ga/` are still empty placeholders for layers that
have not landed yet (`config/`, `runner/`, `recording/`, `metrics/`, `analysis/`,
`viz/`); their module docstrings describe the intended contents.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,analysis]"
```

The core install (`pip install -e .`) pulls only the engine dependency (NumPy).
The `analysis` extra adds the data, statistics and plotting libraries used by the
recording, analysis and viz layers and by the study.

Those ranges are the compatibility contract, not the reproducibility one:
reproducing the recorded results bit-for-bit requires the exact versions in
[`replication/requirements-lock.txt`](replication/requirements-lock.txt).

## Development

```bash
ruff check .          # lint
ruff format .         # formatting
mypy                  # type checking (strict)
pytest                # tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the test-tier conventions and
[`replication/README.md`](replication/README.md) for reproducing the
experiments.

## License

[MIT](LICENSE) © Damian Wiśniewski
