"""Sweeps of the adaptive operator-selection strategies -- the study's treatments.

One module per strategy, each measuring solution quality over the same grid and emitting
the same two artifacts as the reference sweeps in :mod:`experiments.baselines`: the raw
per-run rows and the per-pool-variant aggregates. Keeping the treatments here rather than
alongside the baselines keeps the distinction visible -- the canonical GA, the fixed-best
operator and Random selection are what the strategies are measured *against*, and they
are not themselves under test.

The schemas are deliberately identical across both packages, so a strategy's results, the
ceiling and the floor join on the shared configuration key without reshaping.
"""
