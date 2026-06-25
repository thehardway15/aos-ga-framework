"""Core abstractions shared by the whole framework.

Defines the problem and operator interfaces, the unified quality function
``g(x)`` (the "more is better" convention) and the genome representations the
engine operates on. The engine depends only on these abstractions, never on a
concrete problem.
"""
