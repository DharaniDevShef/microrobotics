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


def save_generation_population(gen_idx, records, out_dir):
    """records: list of dict(graph=nx.DiGraph, objectives=dict, ind_id=int).
    Saves the full graph topology + objectives for every surviving
    individual. `ind_id` (optional - None if a caller doesn't have one) is
    the survivor's index into this generation's flat ind0..indN evaluated
    batch - lets helper_scripts/evolution_results_visualizer.py match a
    survivor back to its exact screenshot/XML/stats.json, and tell newly
    bred offspring (ind_id >= that generation's breeding_events.json
    n_parents) apart from carried-over parents."""
    os.makedirs(out_dir, exist_ok=True)
    payload = [
        {
            "graph": nx.node_link_data(r["graph"], edges="edges"),
            "objectives": r["objectives"],
            "ind_id": r.get("ind_id"),
        }
        for r in records
    ]
    path = os.path.join(out_dir, f"generation_{gen_idx}_population.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


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
    """Scans out_dir for every generation_*_population.json (written by
    save_generation_population), returning [(gen_idx, population)] sorted
    by generation index - the shared data source for the cross-generation
    trend plots below. Re-scans from disk each call (rather than taking
    an accumulated in-memory history) so these plots stay correct even if
    a run is resumed or these functions are called standalone."""
    generations = []
    for name in os.listdir(out_dir):
        if not (name.startswith("generation_") and name.endswith("_population.json")):
            continue
        try:
            gen_idx = int(name[len("generation_"):-len("_population.json")])
        except ValueError:
            continue
        with open(os.path.join(out_dir, name), "r", encoding="utf-8") as f:
            population = json.load(f)
        generations.append((gen_idx, population))
    generations.sort(key=lambda item: item[0])
    return generations


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
