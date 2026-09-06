import plotting_api

plotting_api.plot_rl_vs_baseline_comparison(
    {"RL-Guided NSGA-III": "../output/evolution_run", "Standard NSGA-III": "../output/evolution_run_norl"},
    "../output/comparison",
)
