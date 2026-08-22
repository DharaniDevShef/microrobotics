"""
Plotting & Artifacts API - saves per-generation population JSON, a Pareto
front / parallel-coordinates plot, per-fitness trend lines and a GA
convergence plot across generations, and RL (PPO) training diagnostics.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import objectives_api as obj_api


def append_generation_population(gen_idx, records, out_dir):
    """records: list of dict(graph=nx.DiGraph, objectives=dict, ind_id=int).
    Appends the full graph topology + objectives for every surviving
    individual to out_dir/population_history.json - one growing file for
    the whole run instead of a separate generation_{gen}_population.json
    per generation (which cluttered OUTPUT_DIR and made
    _load_all_generations re-open every prior generation's file on every
    call). `ind_id` (optional - None if a caller doesn't have one) is the
    survivor's index into this generation's flat ind0..indN evaluated
    batch - lets helper_scripts/evolution_results_visualizer.py match a
    survivor back to its exact screenshot/XML/stats.json, and tell newly
    bred offspring (ind_id >= that generation's breeding_events.json
    n_parents) apart from carried-over parents.

    Idempotent by generation index (replaces rather than duplicates an
    existing entry for the same `gen_idx`), matching
    append_generation_stats()'s resume-safe behavior."""
    os.makedirs(out_dir, exist_ok=True)
    population = [
        {
            "graph": nx.node_link_data(r["graph"], edges="edges"),
            "objectives": r["objectives"],
            "ind_id": r.get("ind_id"),
        }
        for r in records
    ]
    path = os.path.join(out_dir, "population_history.json")
    history = _load_population_history(out_dir)
    history = [h for h in history if h["generation"] != gen_idx]
    history.append(dict(generation=gen_idx, population=population))
    history.sort(key=lambda h: h["generation"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def _load_population_history(out_dir):
    path = os.path.join(out_dir, "population_history.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_pareto_front(gen_idx, records, out_dir):
    """Parallel-coordinates plot across f1..f5 (natural, maximize-oriented
    units, min-max normalized per objective for a shared y-axis)."""
    os.makedirs(out_dir, exist_ok=True)
    names = obj_api.OBJECTIVE_NAMES
    if not records:
        return None
    values = np.array([[r["objectives"][n] for n in names] for r in records])

    value_range = np.ptp(values, axis=0)
    value_range[value_range == 0] = 1.0
    norm = (values - values.min(axis=0)) / value_range

    fig, ax = plt.subplots(figsize=(8, 4))
    for row in norm:
        ax.plot(range(len(names)), row, alpha=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("normalized objective value")
    ax.set_title(f"Generation {gen_idx} - population (parallel coordinates)")
    fig.tight_layout()

    path = os.path.join(out_dir, f"generation_{gen_idx}_pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _load_all_generations(out_dir):
    """Reads out_dir/population_history.json (written by
    append_generation_population), returning [(gen_idx, population)]
    sorted by generation index - the shared data source for the
    cross-generation trend plots below. Re-reads from disk each call
    (rather than taking an accumulated in-memory history) so these plots
    stay correct even if a run is resumed or these functions are called
    standalone."""
    history = _load_population_history(out_dir)
    return [(h["generation"], h["population"]) for h in sorted(history, key=lambda h: h["generation"])]


def plot_fitness_trends(out_dir):
    """One subplot per objective (f1..f5): each generation's best and mean
    value, in that objective's own "higher/lower is better" direction per
    objectives_api.MAXIMIZE - how each individual fitness improves across
    generations. Placeholder objectives (currently f1, f3, f4, f5 - see
    objectives_api.py) will just plot flat at 0 until they're implemented."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    names = obj_api.OBJECTIVE_NAMES
    gen_indices = [g for g, _ in generations]

    fig, axes = plt.subplots(len(names), 1, figsize=(7, 2.2 * len(names)), sharex=True)
    axes = [axes] if len(names) == 1 else axes
    for ax, name in zip(axes, names):
        best_fn = max if obj_api.MAXIMIZE[name] else min
        best_vals = [best_fn(entry["objectives"][name] for entry in pop) for _, pop in generations]
        mean_vals = [float(np.mean([entry["objectives"][name] for entry in pop])) for _, pop in generations]
        ax.plot(gen_indices, best_vals, marker="o", label="best")
        ax.plot(gen_indices, mean_vals, marker="o", linestyle="--", label="mean")
        ax.set_title(name)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("generation")
    fig.tight_layout()

    path = os.path.join(out_dir, "fitness_trends.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_convergence(out_dir):
    """Best vs. population-mean SCALARIZED fitness (objectives_api.scalarize
    - the equal-weighted, sign-corrected sum across f1..f5) per generation:
    the classic GA "convergence" view - a healthy run's mean climbs toward
    the best line as the population homogenizes around good solutions."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    gen_indices = [g for g, _ in generations]
    best_vals, mean_vals = [], []
    for _, pop in generations:
        scalarized = [obj_api.scalarize(entry["objectives"]) for entry in pop]
        best_vals.append(max(scalarized))
        mean_vals.append(float(np.mean(scalarized)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gen_indices, best_vals, marker="o", label="best")
    ax.plot(gen_indices, mean_vals, marker="o", linestyle="--", label="mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("scalarized fitness (higher is better)")
    ax.set_title("GA convergence - best vs. mean population fitness")
    ax.legend(loc="best")
    fig.tight_layout()

    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def append_generation_stats(gen_idx, log, out_dir):
    """Appends {generation, n_parents, n_offspring, n_collided} to
    out_dir/generation_stats.json - the per-generation breeding/collision
    counts from moo_api.run_generation's `log`, which aren't reconstructable
    from population_history.json alone (that only has survivors, not
    how many offspring were bred or how many collided). Feeds
    plot_rl_vs_baseline_comparison's collision-rate panel.

    Idempotent by generation index (replaces rather than duplicates an
    existing entry for the same `gen_idx`) so re-running a generation
    after a resume doesn't double-count it."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "generation_stats.json")
    stats = _load_generation_stats(out_dir)
    stats = [s for s in stats if s["generation"] != gen_idx]
    stats.append(dict(
        generation=gen_idx,
        n_parents=log["n_parents"],
        n_offspring=log["n_offspring"],
        n_collided=log["n_collided"],
    ))
    stats.sort(key=lambda s: s["generation"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path


def _load_generation_stats(out_dir):
    path = os.path.join(out_dir, "generation_stats.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_rl_vs_baseline_comparison(run_dirs, comparison_out_dir, filename="rl_vs_baseline_comparison.png"):
    """The RL_ASSISTED_GENETIC_OPERATIONS comparison plot: overlays each
    run's f2 (locomotion), f5 (shape-entropy delta), scalarized fitness
    (convergence), and collision rate, one line per run, so the effect of
    the learned policy vs. random_baseline.py's blind variation is visible
    directly, generation by generation.

    `run_dirs`: dict[label -> OUTPUT_DIR] - e.g.
    {"RL-assisted": ".../evolution_run", "Random baseline": ".../evolution_run_norl"}.
    Runs with no data yet are silently skipped (so this is safe to call
    while one arm is still in progress)."""
    os.makedirs(comparison_out_dir, exist_ok=True)

    runs = {}
    for label, out_dir in run_dirs.items():
        generations = _load_all_generations(out_dir)
        if not generations:
            continue
        gen_stats = {s["generation"]: s for s in _load_generation_stats(out_dir)}
        runs[label] = (generations, gen_stats)
    if not runs:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(8, 14), sharex=True)
    ax_f2, ax_f5, ax_conv, ax_collision = axes

    for label, (generations, gen_stats) in runs.items():
        gen_indices = [g for g, _ in generations]

        best_f2 = [max(e["objectives"]["f2_forward_velocity_folded"] for e in pop) for _, pop in generations]
        mean_f2 = [float(np.mean([e["objectives"]["f2_forward_velocity_folded"] for e in pop])) for _, pop in generations]
        ax_f2.plot(gen_indices, best_f2, marker="o", label=f"{label} (best)")
        ax_f2.plot(gen_indices, mean_f2, marker="o", linestyle="--", label=f"{label} (mean)")

        mean_f5 = [float(np.mean([e["objectives"]["f5_entropy"] for e in pop])) for _, pop in generations]
        ax_f5.plot(gen_indices, mean_f5, marker="o", label=f"{label} (mean)")

        scalarized_best = [max(obj_api.scalarize(e["objectives"]) for e in pop) for _, pop in generations]
        ax_conv.plot(gen_indices, scalarized_best, marker="o", label=label)

        collision_rate = [
            gen_stats[g]["n_collided"] / gen_stats[g]["n_offspring"]
            if g in gen_stats and gen_stats[g]["n_offspring"] else None
            for g in gen_indices
        ]
        if any(v is not None for v in collision_rate):
            xs = [g for g, v in zip(gen_indices, collision_rate) if v is not None]
            ys = [v for v in collision_rate if v is not None]
            ax_collision.plot(xs, ys, marker="o", label=label)

    ax_f2.set_title("f2: forward velocity (locomotion)")
    ax_f2.set_ylabel("m/s")
    ax_f2.legend(loc="best", fontsize=8)

    ax_f5.set_title("f5: shape-entropy delta (folding complexity gain)")
    ax_f5.legend(loc="best", fontsize=8)

    ax_conv.set_title("Convergence: best scalarized fitness")
    ax_conv.legend(loc="best", fontsize=8)

    ax_collision.set_title("Collision rate (n_collided / n_offspring)")
    ax_collision.set_xlabel("generation")
    ax_collision.legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = os.path.join(comparison_out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rl_diagnostics(history, out_dir, filename="rl_diagnostics.png"):
    """history: rl_api.PPOTrainer.history (dict of lists: policy_loss,
    value_loss, entropy, reward)."""
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k in ("reward", "policy_loss", "value_loss", "entropy") if history.get(k)]
    if not keys:
        return None

    fig, axes = plt.subplots(len(keys), 1, figsize=(6, 2.5 * len(keys)))
    axes = [axes] if len(keys) == 1 else axes
    for ax, key in zip(axes, keys):
        ax.plot(history[key])
        ax.set_title(key)
        ax.set_xlabel("mutation step" if key == "reward" else "PPO update step")
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

