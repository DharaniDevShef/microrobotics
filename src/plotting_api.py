"""
Plotting & Artifacts API - saves per-generation population JSON, a Pareto
front / parallel-coordinates plot, and RL (PPO) training diagnostics.
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
    """records: list of dict(graph=nx.DiGraph, objectives=dict). Saves the
    full graph topology + objectives for every surviving individual."""
    os.makedirs(out_dir, exist_ok=True)
    payload = [
        {"graph": nx.node_link_data(r["graph"], edges="edges"), "objectives": r["objectives"]}
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
