"""
Plotting & Artifacts API - saves per-generation population JSON, Pareto
front / fitness-trend / convergence / entropy-vs-velocity plots, and RL
(PPO) training diagnostics across a run. All figures use a shared bright,
high-contrast style (see _apply_bright_style).
"""

import collections
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import moo_api
import objectives_api as obj_api

logger = logging.getLogger(__name__)

DPI = 600

# Shared typography knobs, applied via rcParams instead of per-call fontsize=.
FONT_FAMILY = "Arial"
FONT_SIZE_AXIS_LABEL = 14   # axes.labelsize + tick labels
FONT_SIZE_LEGEND = 11        # legend text + small marker/count annotations


def _save_fig(fig, path, dpi=DPI, **savefig_kwargs):
    """Saves and closes `fig`, logging a warning and returning None instead of
    raising if the write fails (e.g. file locked by a viewer)."""
    try:
        fig.savefig(path, dpi=dpi, **savefig_kwargs)
    except OSError:
        logger.warning(
            "Could not save plot to %s (file may be open in another program) - skipping this plot.",
            path, exc_info=True,
        )
        path = None
    finally:
        plt.close(fig)
    return path


def _integer_x_axis(ax):
    """Forces integer tick labels (avoids fractional generation/step ticks)."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# Human-readable (title, axis-label-with-units) per OBJECTIVE_NAMES entry.
_OBJECTIVE_DISPLAY = {
    "f1_folded_gait_velocity": ("Folded-Gait Velocity", "Velocity - m/s"),
    "f2_entropy": ("Entropy (Folding-Complexity Gain)", "Entropy Δ = H(3D) - H(2D)"),
    "f3_pheromone_yaw_response": ("Pheromone Yaw Response", "Yaw Response - deg"),
    "f4_pheromone_speed_response": ("Pheromone Speed Response", "Speed Response - Δv / v"),
}

# One consistent bright color per objective, reused across plots.
_OBJECTIVE_COLOR = {
    "f1_folded_gait_velocity": "#1f77ff",   # bright blue
    "f2_entropy": "#22b14c",                # bright green
    "f3_pheromone_yaw_response": "#e8382b", # bright red
    "f4_pheromone_speed_response": "#a349e6",  # bright purple
}

_RL_METRIC_COLOR = {
    "reward": "#ff1f8f",
    "collision_rate": "#1f77ff",
    "policy_loss": "#e8382b",
    "value_loss": "#a349e6",
    "entropy": "#22b14c",
    "entropy_coef": "#ff8c00",
}

# Fixed per-arm colors so a single-run convergence.png stays visually
# consistent with the RL-vs-baseline comparison plots.
_RL_ARM_COLOR = "#1f77ff"
_BASELINE_ARM_COLOR = "#e8382b"


def _apply_bright_style():
    """One shared look for every figure: white background, bold titles,
    light grid. Applied once at import time via rcParams."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "font.family": FONT_FAMILY,
        "axes.labelsize": FONT_SIZE_AXIS_LABEL,
        "xtick.labelsize": FONT_SIZE_AXIS_LABEL,
        "ytick.labelsize": FONT_SIZE_AXIS_LABEL,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "font.size": 10,
        "lines.linewidth": 2.0,
    })


_apply_bright_style()


def append_generation_population(gen_idx, records, out_dir):
    """Appends this generation's surviving individuals (graph + objectives) to
    out_dir/population_history.json. Idempotent by generation index."""
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


def save_pareto_archive(archive, out_dir):
    """Writes out_dir/pareto_archive.json with the current Pareto archive
    (list of dict(graph, objectives, F)). Overwrites each call, since the
    archive is already the complete, deduplicated non-dominated set."""
    os.makedirs(out_dir, exist_ok=True)
    payload = [
        {
            "graph": nx.node_link_data(r["graph"], edges="edges"),
            "objectives": r["objectives"],
            "F": list(r["F"]) if not isinstance(r["F"], list) else r["F"],
        }
        for r in archive
    ]
    path = os.path.join(out_dir, "pareto_archive.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_pareto_archive(out_dir):
    """Inverse of save_pareto_archive - returns [] if none exists yet."""
    path = os.path.join(out_dir, "pareto_archive.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        dict(
            graph=nx.node_link_graph(r["graph"], edges="edges"),
            objectives=r["objectives"],
            F=np.array(r["F"]),
        )
        for r in payload
    ]


def _load_population_history(out_dir):
    path = os.path.join(out_dir, "population_history.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_all_generations(out_dir):
    """Reads population_history.json, returning [(gen_idx, population)] sorted
    by generation - the shared data source for every plot below. Objectives
    are migrated to the current schema via migrate_legacy_objectives()."""
    history = _load_population_history(out_dir)
    generations = [(h["generation"], h["population"]) for h in sorted(history, key=lambda h: h["generation"])]
    for _, pop in generations:
        for entry in pop:
            entry["objectives"] = obj_api.migrate_legacy_objectives(entry["objectives"])
    return generations


def _population_costs_and_ranks(pop):
    """(costs, rank): costs is an (n, len(OBJECTIVE_NAMES)) array, each column
    min-max normalized to [0, 1] where LOWER is always better. rank is the
    non-dominated sort front index per individual (0 = Pareto front)."""
    F = np.array([obj_api.to_minimization_vector(e["objectives"]) for e in pop])
    value_range = np.ptp(F, axis=0)
    value_range[value_range == 0] = 1.0
    costs = (F - F.min(axis=0)) / value_range

    fronts = NonDominatedSorting().do(F)
    rank = np.zeros(len(pop), dtype=int)
    for r, front in enumerate(fronts):
        rank[front] = r
    return costs, rank


def plot_pareto_front_last_gen(out_dir, filename="pareto_front.png", n_generations=5):
    """4 panels (f1 vs f2, f2 vs f3, f3 vs f4, f4 vs f1) overlaying up to
    `n_generations` generations' own Pareto fronts (rank-0 members only),
    evenly spaced across the run and colored by generation, so how the
    front moved over the run is visible directly. Normalized against one
    shared basis (min/max across the whole run) so fronts stay comparable
    across generations."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    # Shared normalization basis: every individual, every rank, every
    # generation in the run.
    all_F = np.array([
        obj_api.to_minimization_vector(entry["objectives"])
        for _, pop in generations for entry in pop
    ])
    if len(all_F) == 0:
        return None
    value_range = np.ptp(all_F, axis=0)
    value_range[value_range == 0] = 1.0
    f_min = all_F.min(axis=0)

    # Up to n_generations, evenly spaced, always including first and last.
    n_pick = min(n_generations, len(generations))
    pick_idxs = np.unique(np.round(np.linspace(0, len(generations) - 1, n_pick)).astype(int))
    selected = [generations[i] for i in pick_idxs]

    names = obj_api.OBJECTIVE_NAMES
    titles = [_OBJECTIVE_DISPLAY[n][0] for n in names]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    cmap = plt.cm.viridis
    gen_min, gen_max = selected[0][0], selected[-1][0]

    sc = None
    for gen_idx, pop in selected:
        if len(pop) < 2:
            continue
        F = np.array([obj_api.to_minimization_vector(e["objectives"]) for e in pop])
        front_idx = NonDominatedSorting().do(F)[0]
        front_costs = (F[front_idx] - f_min) / value_range
        gen_colors = np.full(len(front_idx), gen_idx)
        for ax, (i, j) in zip(axes, pairs):
            sc = ax.scatter(
                front_costs[:, i], front_costs[:, j], c=gen_colors,
                cmap=cmap, vmin=gen_min, vmax=max(gen_max, gen_min + 1),
                s=70, alpha=0.9, edgecolors="#333333", linewidths=0.5,
            )

    for ax, (i, j) in zip(axes, pairs):
        ax.set_xlabel(titles[i])
        ax.set_ylabel(titles[j])
        # Short title only - full names are already on the axis labels.
        ax.set_title(f"f{i + 1} vs f{j + 1}", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.05, pad=0.02)
        cbar.set_label("Generation")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    gens_str = ", ".join(str(g) for g, _ in selected)
    fig.suptitle(f"Pareto Front Across Generations", fontsize=15, fontweight="bold") #only few generations

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def plot_pareto_parallel_coordinates(out_dir, filename="pareto_parallel_coordinates.png"):
    """Parallel-coordinates view of the most recent generation's population:
    one axis per objective (f1..f4), one line per individual, showing
    tradeoffs across all objectives at once. "Up" always means "better" on
    every axis; rank 0 (front) is cyan, ranks 1-4 light blue, rank 5+ a
    plasma gradient by rank."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None
    gen_idx, pop = generations[-1]
    if len(pop) < 2:
        return None

    names = obj_api.OBJECTIVE_NAMES
    titles = [_OBJECTIVE_DISPLAY[n][0] for n in names]
    costs, rank = _population_costs_and_ranks(pop)
    goodness = 1.0 - costs
    max_rank = int(rank.max()) if len(rank) else 0

    NEAR_FRONT_MAX_RANK = 4
    NEAR_FRONT_COLOR = "#8ecfff"
    cmap = plt.cm.plasma

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6.5))

    far = rank > NEAR_FRONT_MAX_RANK
    near = (rank > 0) & (rank <= NEAR_FRONT_MAX_RANK)
    front = rank == 0

    if far.any():
        norm = plt.Normalize(vmin=NEAR_FRONT_MAX_RANK + 1, vmax=max(max_rank, NEAR_FRONT_MAX_RANK + 1))
        for idx in np.where(far)[0]:
            ax.plot(x, goodness[idx], color=cmap(norm(rank[idx])), alpha=0.55, linewidth=1.2, zorder=1)
    if near.any():
        for idx in np.where(near)[0]:
            ax.plot(x, goodness[idx], color=NEAR_FRONT_COLOR, alpha=0.8, linewidth=1.6, zorder=2)
    for idx in np.where(front)[0]:
        ax.plot(x, goodness[idx], color="#00e5e5", alpha=0.95, linewidth=2.4, zorder=3)

    for xi in x:
        ax.axvline(xi, color="#999999", linewidth=0.9, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Normalized fitness")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Pareto Front — Parallel Coordinates — Generation {gen_idx}", fontsize=15, fontweight="bold")

    handles = [Line2D([0], [0], color="#00e5e5", linewidth=2.4, label="Pareto front — rank 0")]
    if near.any():
        handles.append(Line2D([0], [0], color=NEAR_FRONT_COLOR, linewidth=1.6, label=f"Rank 1-{NEAR_FRONT_MAX_RANK}"))
    if far.any():
        handles.append(Line2D([0], [0], color=cmap(0.6), linewidth=1.2, label=f"Rank > {NEAR_FRONT_MAX_RANK}"))
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def plot_fitness_trends(out_dir, filename="fitness_trends.png"):
    """One subplot per objective (2x2 grid, f1..f4): each generation's best
    value (solid line) and population mean (dashed line), so a gap that
    isn't closing is visible per-objective rather than one scalarized
    number as in plot_convergence."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    names = obj_api.OBJECTIVE_NAMES
    gen_indices = [g for g, _ in generations]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        title, ylabel = _OBJECTIVE_DISPLAY[name]
        best_fn = max if obj_api.MAXIMIZE[name] else min
        best_vals = [best_fn(entry["objectives"].get(name, 0.0) for entry in pop) for _, pop in generations]
        mean_vals = [
            float(np.mean([entry["objectives"].get(name, 0.0) for entry in pop])) for _, pop in generations
        ]
        ax.plot(gen_indices, best_vals, color=_OBJECTIVE_COLOR[name], label="Best")
        ax.plot(gen_indices, mean_vals, color=_OBJECTIVE_COLOR[name], linestyle="--", alpha=0.6, label="Mean")
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND)
        _integer_x_axis(ax)

    fig.suptitle("Fitness Trends Across Generations - Best vs. Mean per Objective", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path)


def plot_entropy_vs_velocity(out_dir, filename="entropy_vs_velocity.png"):
    """Scatter of every individual across every generation, f2 (entropy)
    against f1 (folded-gait velocity), colored by generation. Pooling every
    individual (not per-generation means) keeps any real correlation
    visible instead of washing it out."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    gens, f1_vals, f2_vals = [], [], []
    for gen_idx, pop in generations:
        for entry in pop:
            gens.append(gen_idx)
            f1_vals.append(entry["objectives"].get("f1_folded_gait_velocity", 0.0))
            f2_vals.append(entry["objectives"].get("f2_entropy", 0.0))
    if not f1_vals:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(f1_vals, f2_vals, c=gens, cmap=plt.cm.viridis, s=45,
                     edgecolors="#333333", linewidths=0.4, alpha=0.9)
    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    ax.set_xlabel(_OBJECTIVE_DISPLAY["f1_folded_gait_velocity"][1])
    ax.set_ylabel(_OBJECTIVE_DISPLAY["f2_entropy"][1])
    ax.set_title("Entropy (Folding-Complexity Gain) vs. Folded-Gait Velocity", fontsize=15, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Generation")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path)


def _scalarize_offline(generations):
    """Same formula as objectives_api.scalarize() (equal-weight mean of each
    objective normalized to [0,1]), but with the running min/max range
    rebuilt locally and chronologically here instead of read from
    objectives_api's live module-global state - so this gives the same
    answer whether called from a live run or standalone (e.g. compare.py).

    `generations`: [(gen_idx, population), ...]. Returns
    dict[gen_idx -> list[float]], one scalarized score per individual."""
    running_min, running_max = {}, {}
    scores_by_gen = {}
    for gen_idx, pop in generations:
        # Pass 1: fold this whole generation into the running range before
        # scoring any of it, so no individual's score depends on processing
        # order within the generation.
        for entry in pop:
            objectives = entry["objectives"]
            for n in obj_api.OBJECTIVE_NAMES:
                v = objectives.get(n, 0.0)
                if n not in running_min or v < running_min[n]:
                    running_min[n] = v
                if n not in running_max or v > running_max[n]:
                    running_max[n] = v

        # Pass 2: score against the now-complete range for this generation.
        scores = []
        for entry in pop:
            objectives = entry["objectives"]
            contributions = []
            for n in obj_api.OBJECTIVE_NAMES:
                lo, hi = running_min.get(n), running_max.get(n)
                if lo is None or (hi - lo) < 1e-9:
                    continue
                norm = (objectives.get(n, 0.0) - lo) / (hi - lo)
                contributions.append(norm if obj_api.MAXIMIZE[n] else (1.0 - norm))

            if not contributions:
                scores.append(float(sum(
                    objectives.get(n, 0.0) if obj_api.MAXIMIZE[n] else -objectives.get(n, 0.0)
                    for n in obj_api.OBJECTIVE_NAMES
                )))
            else:
                scores.append(float(np.mean(contributions)))
        scores_by_gen[gen_idx] = scores
    return scores_by_gen


def plot_convergence(out_dir, is_rl=True, filename="convergence.png"):
    """Best vs. population-mean scalarized fitness per generation - the
    classic GA convergence view. `is_rl` selects which arm's fixed color
    (_RL_ARM_COLOR / _BASELINE_ARM_COLOR) to draw both lines in."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    gen_indices = [g for g, _ in generations]
    scores_by_gen = _scalarize_offline(generations)
    best_vals, mean_vals = [], []
    for gen_idx, _ in generations:
        scalarized = scores_by_gen[gen_idx]
        best_vals.append(max(scalarized))
        mean_vals.append(float(np.mean(scalarized)))

    color = _RL_ARM_COLOR if is_rl else _BASELINE_ARM_COLOR
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(gen_indices, best_vals, color=color, label="Best")
    ax.plot(gen_indices, mean_vals, color=color, linestyle="--", alpha=0.7, label="Mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Scalarized fitness")
    ax.set_title("GA Convergence - Best vs. Mean Population Fitness", fontsize=15, fontweight="bold")
    ax.legend(loc="best")
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path)


# HV's reference point must be worse than every point it's ever handed;
# since fronts are normalized to [0, 1], a fixed point just past 1.0 works.
_HV_REF_POINT_MARGIN = 1.05


def _run_wide_normalized(entries):
    """(n, len(OBJECTIVE_NAMES)) array, min-max normalized against the range
    observed across ALL of `entries` (not one generation's local range), so
    hypervolume is comparable as a trend across the whole run."""
    F = np.array([obj_api.to_minimization_vector(e["objectives"]) for e in entries])
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    return (F - lo) / span


def compute_hypervolume(out_dir):
    """Per-generation Hypervolume (HV) - the standard multi-objective quality
    indicator here since it only needs a reference point, not a known "true"
    Pareto front (unlike GD/IGD/IGD+). Returns dict(generations=[...],
    hv=[...]), or None if no population_history.json exists yet."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    all_entries = [entry for _, pop in generations for entry in pop]
    normalized_all = _run_wide_normalized(all_entries)
    hv_indicator = HV(ref_point=np.full(normalized_all.shape[1], _HV_REF_POINT_MARGIN))

    result = dict(generations=[], hv=[])
    cursor = 0
    for gen_idx, pop in generations:
        n = len(pop)
        gen_normalized = normalized_all[cursor:cursor + n]
        cursor += n
        gen_fronts = NonDominatedSorting().do(gen_normalized)
        gen_front = gen_normalized[gen_fronts[0]]

        result["generations"].append(gen_idx)
        result["hv"].append(float(hv_indicator(gen_front)))
    return result


def plot_hypervolume(out_dir, filename="hypervolume.png"):
    """Hypervolume trend across generations (see compute_hypervolume)."""
    data = compute_hypervolume(out_dir)
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(data["generations"], data["hv"], color="#1f77ff")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume Across Generations", fontsize=15, fontweight="bold")
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path)


def append_generation_stats(gen_idx, log, out_dir):
    """Appends {generation, n_parents, n_offspring, n_collided} to
    out_dir/generation_stats.json. Idempotent by generation index."""
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


# One consistent color per grammar action; mutation actions get a light
# blue/green/gold family, crossover actions bold red/purple so they stand
# out in stacked/grouped views.
_ACTION_COLORS = {
    "ADD_NODE": "#1f77ff",
    "DELETE_NODE": "#5aa0ff",
    "PRUNE_SUBTREE": "#8ec2ff",
    "MUTATE_FOLD_TYPE": "#22b14c",
    "MUTATE_HINGE_ANGLE": "#6fcf87",
    "RECONNECT_PORT": "#c9a227",
    "TOGGLE_LIGHT_SENSOR": "#e0c46c",
    "MUTATE_LIGHT_HINGE_ANGLE": "#f0dfa0",
    "GRAFT_SUBTREE": "#e8382b",
    "SWAP_SUBTREES": "#a349e6",
}


def _load_breeding_events(out_dir):
    """[(gen_idx, events)] for every generation with a breeding_events.json.
    Generations with a missing file are silently skipped."""
    generations = _load_all_generations(out_dir)
    result = []
    for gen_idx, _ in generations:
        path = os.path.join(out_dir, f"generation_{gen_idx}", "breeding_events.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        result.append((gen_idx, payload.get("events", [])))
    return result


def plot_action_distribution(out_dir, filename="action_distribution.png"):
    """Stacked-area chart of which grammar action was picked, as a fraction
    of each generation's breeding decisions, generation by generation."""
    events_by_gen = _load_breeding_events(out_dir)
    if not events_by_gen:
        return None

    gen_indices = [g for g, _ in events_by_gen]
    action_names = list(_ACTION_COLORS.keys())
    fractions = {a: [] for a in action_names}
    for _, events in events_by_gen:
        total = len(events)
        counts = collections.Counter(e["action"] for e in events)
        for a in action_names:
            fractions[a].append((counts.get(a, 0) / total) if total else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.stackplot(
        gen_indices, [fractions[a] for a in action_names],
        labels=[a.replace("_", " ").title() for a in action_names],
        colors=[_ACTION_COLORS[a] for a in action_names],
        alpha=0.9, edgecolor="#ffffff", linewidth=0.3,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fraction of Breeding Decisions")
    ax.set_ylim(0, 1)
    ax.set_title("Action-Type Distribution Across Generations", fontsize=15, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=FONT_SIZE_LEGEND)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def plot_reward_and_loss_by_action(history, out_dir, filename="rl_diagnostics_by_action.png"):
    """Two-panel bar chart of PPO's mean reward and policy loss broken down
    by grammar action type, aggregated over the run so far - diagnoses
    whether one action type's collapsing probability (see
    plot_action_distribution) is explained by systematically worse reward.
    Each bar is annotated with its sample count since rare actions can be
    averaged over far fewer samples."""
    reward_vals = history.get("reward") or []
    reward_actions = history.get("reward_action") or []
    loss_snapshots = history.get("loss_by_action") or []
    if not reward_vals and not loss_snapshots:
        return None

    action_names = list(_ACTION_COLORS.keys())

    reward_by_action = collections.defaultdict(list)
    for r, a in zip(reward_vals, reward_actions):
        reward_by_action[a].append(r)

    policy_loss_sum = collections.defaultdict(float)
    policy_loss_n = collections.defaultdict(int)
    for snapshot in loss_snapshots:
        for a, stats in snapshot.items():
            policy_loss_sum[a] += stats["policy_loss"] * stats["count"]
            policy_loss_n[a] += stats["count"]

    present = [a for a in action_names if reward_by_action.get(a) or policy_loss_n.get(a)]
    if not present:
        return None

    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(10, 9))
    labels = [a.replace("_", " ").title() for a in present]
    colors = [_ACTION_COLORS[a] for a in present]

    reward_means = [float(np.mean(reward_by_action[a])) if reward_by_action.get(a) else 0.0 for a in present]
    reward_counts = [len(reward_by_action.get(a, [])) for a in present]
    bars_r = ax_r.bar(labels, reward_means, color=colors, edgecolor="#333333", linewidth=0.6)
    for bar, n in zip(bars_r, reward_counts):
        ax_r.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=FONT_SIZE_LEGEND)
    ax_r.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_r.set_title("Mean Reward by Action Type", fontsize=13, fontweight="bold")
    ax_r.set_ylabel("Mean Reward")
    ax_r.tick_params(axis="x", rotation=30)

    policy_loss_means = [(policy_loss_sum[a] / policy_loss_n[a]) if policy_loss_n.get(a) else 0.0 for a in present]
    bars_l = ax_l.bar(labels, policy_loss_means, color=colors, edgecolor="#333333", linewidth=0.6)
    for bar, n in zip(bars_l, [policy_loss_n.get(a, 0) for a in present]):
        ax_l.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=FONT_SIZE_LEGEND)
    ax_l.set_title("Mean Policy Loss by Action Type - Final PPO Epoch per Update", fontsize=13, fontweight="bold")
    ax_l.set_ylabel("Mean Policy Loss")
    ax_l.tick_params(axis="x", rotation=30)

    fig.suptitle("PPO Reward / Loss Broken Out by Grammar Action", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def _load_comparison_runs(run_dirs):
    """dict[label -> [(gen_idx, population), ...]] for every run_dirs entry
    that has data yet; runs with no data are silently dropped."""
    runs = {}
    for label, out_dir in run_dirs.items():
        generations = _load_all_generations(out_dir)
        if generations:
            runs[label] = generations
    return runs


def plot_convergence_comparison(run_dirs, comparison_out_dir, filename="convergence_comparison.png"):
    """Multi-run counterpart to plot_convergence(): one best/mean pair of
    scalarized-fitness lines per run, so runs are directly comparable.

    `run_dirs`: dict[label -> OUTPUT_DIR]."""
    os.makedirs(comparison_out_dir, exist_ok=True)
    runs = _load_comparison_runs(run_dirs)
    if not runs:
        return None

    run_colors = ["#1f77ff", "#e8382b", "#22b14c", "#a349e6"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for color, (label, generations) in zip(run_colors, runs.items()):
        gen_indices = [g for g, _ in generations]
        # Each run gets its own independently-rebuilt running range.
        scores_by_gen = _scalarize_offline(generations)
        best_vals, mean_vals = [], []
        for gen_idx, _ in generations:
            scalarized = scores_by_gen[gen_idx]
            best_vals.append(max(scalarized))
            mean_vals.append(float(np.mean(scalarized)))
        ax.plot(gen_indices, best_vals, color=color, label=f"{label} - best")
        ax.plot(gen_indices, mean_vals, color=color, linestyle="--", alpha=0.7, label=f"{label} - mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Scalarized fitness")
    ax.set_title("GA Convergence - Best vs. Mean Population Fitness", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path)


def plot_fitness_trends_comparison(run_dirs, comparison_out_dir, filename="fitness_trends_comparison.png"):
    """Multi-run counterpart to plot_fitness_trends(): one subplot per
    objective, each run's best (solid) and mean (dashed) value per
    generation.

    `run_dirs`: dict[label -> OUTPUT_DIR]."""
    os.makedirs(comparison_out_dir, exist_ok=True)
    runs = _load_comparison_runs(run_dirs)
    if not runs:
        return None

    run_colors = ["#1f77ff", "#e8382b", "#22b14c", "#a349e6"]
    names = obj_api.OBJECTIVE_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        title, ylabel = _OBJECTIVE_DISPLAY[name]
        best_fn = max if obj_api.MAXIMIZE[name] else min
        for color, (label, generations) in zip(run_colors, runs.items()):
            gen_indices = [g for g, _ in generations]
            best_vals = [best_fn(entry["objectives"].get(name, 0.0) for entry in pop) for _, pop in generations]
            mean_vals = [
                float(np.mean([entry["objectives"].get(name, 0.0) for entry in pop])) for _, pop in generations
            ]
            ax.plot(gen_indices, best_vals, color=color, label=f"{label} - best")
            ax.plot(gen_indices, mean_vals, color=color, linestyle="--", alpha=0.6, label=f"{label} - mean")
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        _integer_x_axis(ax)

    # One shared legend below the grid instead of one per identical subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 4), fontsize=FONT_SIZE_LEGEND)

    fig.suptitle("Fitness Trends Across Generations - Best vs. Mean per Objective", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def plot_collision_rate_comparison(run_dirs, comparison_out_dir, filename="collision_rate_comparison.png"):
    """Collision rate (n_collided / n_offspring per generation), one line
    per run."""
    os.makedirs(comparison_out_dir, exist_ok=True)
    runs = _load_comparison_runs(run_dirs)
    if not runs:
        return None

    run_colors = ["#1f77ff", "#e8382b", "#22b14c", "#a349e6"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False

    for color, (label, generations) in zip(run_colors, runs.items()):
        gen_stats = {s["generation"]: s for s in _load_generation_stats(run_dirs[label])}
        gen_indices = [g for g, _ in generations]
        collision_rate = [
            gen_stats[g]["n_collided"] / gen_stats[g]["n_offspring"]
            if g in gen_stats and gen_stats[g]["n_offspring"] else None
            for g in gen_indices
        ]
        xs = [g for g, v in zip(gen_indices, collision_rate) if v is not None]
        ys = [v for v in collision_rate if v is not None]
        if xs:
            any_data = True
            ax.plot(xs, ys, color=color, label=label)

    if not any_data:
        plt.close(fig)
        return None

    ax.set_title("Collision Rate - n_collided / n_offspring", fontsize=15, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Collision rate")
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path)


def plot_hypervolume_comparison(run_dirs, comparison_out_dir, filename="hypervolume_comparison.png"):
    """Multi-run counterpart to plot_hypervolume(): one HV line per run,
    each normalized against that run's own observed range (so it matches
    that run's standalone hypervolume.png). Absolute magnitudes aren't
    strictly apples-to-apples across runs; each run's own shape/trend is
    what's directly comparable.

    `run_dirs`: dict[label -> OUTPUT_DIR]."""
    os.makedirs(comparison_out_dir, exist_ok=True)
    run_colors = ["#1f77ff", "#e8382b", "#22b14c", "#a349e6"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False

    for color, (label, out_dir) in zip(run_colors, run_dirs.items()):
        data = compute_hypervolume(out_dir)
        if not data:
            continue
        any_data = True
        ax.plot(data["generations"], data["hv"], color=color, label=label)

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume (run normalized) ") #(each run normalized against its own range)
    ax.set_title("Hypervolume Across Generations", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path)


def plot_rl_vs_baseline_comparison(run_dirs, comparison_out_dir):
    """Generates the four RL-vs-baseline comparison plots (convergence,
    fitness trends, collision rate, hypervolume) as separate PNGs.

    `run_dirs`: dict[label -> OUTPUT_DIR]. Returns {name: path_or_None}."""
    return dict(
        convergence=plot_convergence_comparison(run_dirs, comparison_out_dir),
        fitness_trends=plot_fitness_trends_comparison(run_dirs, comparison_out_dir),
        collision_rate=plot_collision_rate_comparison(run_dirs, comparison_out_dir),
        hypervolume=plot_hypervolume_comparison(run_dirs, comparison_out_dir),
    )

# A real scalarize()-delta reward is bounded to [-1, 1]; COLLISION_PENALTY
# is a flat penalty far below that, so anything past this threshold is a
# collision event, not real reward signal.
_COLLISION_REWARD_THRESHOLD = (moo_api.COLLISION_PENALTY - 1.0) / 2


_STEP_AXIS_LABEL = "Breeding Decision Step"


def plot_rl_diagnostics(history, out_dir, suffix=""):
    """One separate PNG per PPO training metric (reward with/without
    collision penalty, policy_loss, value_loss, entropy). `history` is
    rl_api.PPOTrainer.history. `suffix` is inserted before each filename's
    .png extension to identify which run it came from. Reward gets two
    plots: full range (collision spikes visible) and with those spikes
    excluded so the real signal isn't dwarfed by them. Returns
    dict[metric_key -> path] for whichever metrics had data."""
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    reward_values = history.get("reward")
    if reward_values:
        values_arr = np.asarray(reward_values, dtype=float)
        collision_mask = values_arr <= _COLLISION_REWARD_THRESHOLD
        n_collision = int(collision_mask.sum())
        steps = np.arange(len(values_arr))

        fig, ax_full = plt.subplots(figsize=(9, 5.5))
        ax_full.plot(steps, values_arr, color=_RL_METRIC_COLOR["reward"], linewidth=1.0)
        ax_full.set_title("Reward With Collision Penalty", fontsize=15, fontweight="bold")
        ax_full.set_xlabel(_STEP_AXIS_LABEL)
        ax_full.set_ylabel("Reward")
        fig.tight_layout()

        path = os.path.join(out_dir, f"rl_diagnostics_reward_with_penalty{suffix}.png")
        saved = _save_fig(fig, path)
        if saved:
            paths["reward_with_penalty"] = saved

        real_steps = steps[~collision_mask]
        real_values = values_arr[~collision_mask]
        fig, ax_zoom = plt.subplots(figsize=(9, 5.5))
        ax_zoom.plot(real_steps, real_values, color=_RL_METRIC_COLOR["reward"], linewidth=1.0)
        if len(real_values):
            span = max(float(real_values.max() - real_values.min()), 0.05)
            pad = 0.1 * span
            ax_zoom.set_ylim(real_values.min() - pad, real_values.max() + pad)
        ax_zoom.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        ax_zoom.set_title(
            f"Reward Without Collision Penalty ({n_collision} Excluded)",
            fontsize=15, fontweight="bold",
        )
        ax_zoom.set_xlabel(_STEP_AXIS_LABEL)
        ax_zoom.set_ylabel("Reward")
        fig.tight_layout()

        path = os.path.join(out_dir, f"rl_diagnostics_reward_without_penalty{suffix}.png")
        saved = _save_fig(fig, path)
        if saved:
            paths["reward_without_penalty"] = saved

        # Per-step outcome drawn faint as context, under a bold rolling-
        # window rejection-rate line that shows the actual trend.
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(steps, collision_mask.astype(float) * 100, color=_RL_METRIC_COLOR["collision_rate"],
                linewidth=0.8, alpha=0.25, zorder=1, label="Per-step outcome")
        if len(values_arr) >= 5:
            window = max(5, min(51, len(values_arr) // 10))
            kernel = np.ones(window) / window
            rolling_rate = np.convolve(collision_mask.astype(float), kernel, mode="same") * 100
            ax.plot(steps, rolling_rate, color=_RL_METRIC_COLOR["collision_rate"], linewidth=2.4, zorder=2,
                    label=f"Rolling rejection rate (window={window})")
            ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
        ax.set_ylim(-5, 105)
        ax.set_xlabel(_STEP_AXIS_LABEL)
        ax.set_ylabel("Collision-Gate Rejection Rate (%)")
        ax.set_title("Collision-Gate Rejection Rate Over Training", fontsize=15, fontweight="bold")
        fig.tight_layout()

        path = os.path.join(out_dir, f"rl_diagnostics_collision_rate{suffix}.png")
        saved = _save_fig(fig, path)
        if saved:
            paths["collision_rate"] = saved

    specs = [
        ("policy_loss", "Actor Policy Loss", "PPO Update Step", "Policy Loss"),
        ("value_loss", "Critic Value Loss", "PPO Update Step", "Value Loss"),
        ("entropy", "Policy Entropy", "PPO Update Step", "Entropy"),
        ("entropy_coef", "Entropy Coefficient Schedule", "PPO Update Step", "Entropy Coefficient"),
    ]
    for key, title, xlabel, ylabel in specs:
        values = history.get(key)
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(values, color=_RL_METRIC_COLOR[key])
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

        path = os.path.join(out_dir, f"rl_diagnostics_{key}{suffix}.png")
        saved = _save_fig(fig, path)
        if saved:
            paths[key] = saved
    return paths
