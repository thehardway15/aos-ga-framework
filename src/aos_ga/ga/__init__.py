"""Genetic algorithm engine built on DEAP, with a shared operator pool.

Implements the generation loop — initialization, evaluation, tournament
selection, single-operator variation and elitist succession — with explicit
hook points for the AOS strategy and the credit-assignment module. Crossover
and mutation frequencies are an outcome of the AOS decisions, not fixed rates.
"""
