import plotting_api

plotting_api.plot_rl_vs_baseline_comparison(
    {"RL-assisted": "../output/evolution_run", "Random baseline": "../output/evolution_run_norl"},
    "../output/comparison",
)
