"""Execution of configured runs.

Drives runs from configurations with resumable, parallel execution over
independent seeds and configurations, writing raw results and logs. Concrete
problems and baselines are resolved by name through a registry, which keeps the
runner problem-agnostic. Parallel and sequential execution produce identical
results for a given seed.
"""
