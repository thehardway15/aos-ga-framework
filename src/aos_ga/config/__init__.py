"""Experiment configuration schema and matrix generation.

Describes a single run as the tuple <P, B, N, O, S, C_A, s> and builds the full
experimental matrix as a validated Cartesian product, with completeness checks
that guard against missing required cells. The matrix is the source of truth for
the scope of each experiment phase.
"""
