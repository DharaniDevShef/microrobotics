"""
Compare - renders the RL-guided vs. random-baseline NSGA-III comparison
figure from two already-completed runs (../output/evolution_run and
../output/evolution_run_norl) into ../output/comparison.
"""

import plotting_api

plotting_api.plot_rl_vs_baseline_comparison(
    {"RL-Guided NSGA-III": "../output/evolution_run", "Standard NSGA-III": "../output/evolution_run_norl"},
    "../output/comparison",
)
