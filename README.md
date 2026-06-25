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
  core/         problem and operator interfaces, g(x), representations
  operators/    generic operators (permutation, binary, real) and pools
  ga/           shared-pool GA engine (DEAP)
  aos/          AOS strategies (Random, PM, AP, UCB, DMAB) + round-robin
  credit/       credit assignment (IR, RFI, rank-based)
  config/       run configuration and matrix generation
  runner/       resumable, parallel execution
  recording/    per-step, dynamics and result logs
  metrics/      quality and AOS metrics
  analysis/     statistical tests
  viz/          plots
experiments/    the study: concrete problems, baselines, dataset loaders, configs
data/           versioned inputs (seeds, TSPLIB, knapsack dataset + manifest)
results/        outputs (aggregated tables, figures; raw runs are git-ignored)
replication/    end-to-end reproduction instructions
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,analysis]"
```

The core install (`pip install -e .`) pulls only the engine dependencies (NumPy,
DEAP). The `analysis` extra adds the data, statistics and plotting libraries
used by the recording, analysis and viz layers and by the study.

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
