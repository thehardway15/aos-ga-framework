# Development

## Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,analysis]"
pre-commit install                 # optional
```

## Layout

- `src/aos_ga/` — the problem-agnostic framework (the only part that ships as a
  package).
- `experiments/` — the study: concrete problems, baselines, dataset loaders and
  the per-phase experiment configurations.
- `data/`, `results/` — versioned inputs and outputs.

## Checks

Run the same set of checks locally as CI does:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Tests that need the `analysis` extras (pandas/pyarrow/scipy/matplotlib/seaborn)
must be marked `@pytest.mark.heavy`. The cross-platform framework matrix runs
`-m "not heavy"`, while the single study job installs the extras and runs the
full suite. The experiment campaign itself is never run in CI.

`experiments/tests/test_results_regression.py` re-runs one grid cell per
committed result table and compares it against the stored aggregates. It fails
whenever a change to an operator, the engine, the pools or the seeds would
invalidate numbers already written down. A failure means the code and
`results/aggregated/` have parted ways: either revert, or re-run the affected
sweep and commit the new artifact together with the reason.

## Conventions

- Formatting and linting: **Ruff** (configuration in `pyproject.toml`).
- Typing: **mypy** in `strict` mode; the public API is fully typed.
- Tests: **pytest**; new code should be covered by tests.
- Commits: a short, imperative-mood summary.
