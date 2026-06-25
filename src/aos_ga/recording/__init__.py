"""Logging of runs into separate, analysis-ready streams.

Three independent streams with fixed schemas: the per-step AOS log and the
per-generation dynamics log (columnar / Parquet) and the final result record
(tabular / CSV), with run configuration and metadata serialized alongside.
Outputs are organized per experiment phase.
"""
