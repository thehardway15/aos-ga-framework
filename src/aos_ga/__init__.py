"""aos_ga — adaptive operator selection for genetic algorithms.

A problem-agnostic framework for studying adaptive operator selection (AOS)
strategies in genetic algorithms under tight evaluation budgets. The package
provides the generic engine (operators, GA loop, AOS strategies, credit
assignment), the experiment infrastructure (configuration, runner, recording)
and the analysis toolkit (metrics, statistics, visualization).

Concrete test problems, datasets and baselines specific to a particular study
live outside this package, in the ``experiments`` research layer.
"""

__version__ = "0.1.0"
