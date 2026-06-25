"""Adaptive operator selection strategies.

The strategy interface and its implementations — Random, Probability Matching,
Adaptive Pursuit, UCB and DMAB — together with the round-robin warm-up phase.
Strategies consume non-negative rewards from the credit-assignment module and
expose their internal state for the dynamics logs.
"""
