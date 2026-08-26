"""
Plotting & Artifacts API - saves per-generation population JSON, a Pareto
front plot (last generation, pairwise objective panels), per-fitness trend
lines, a GA convergence plot, an entropy-vs-velocity scatter, and RL (PPO)
training diagnostics (one PNG per metric) across a run.

All figures are saved at DPI (300, "print quality") with a shared bright,
high-contrast style (see _apply_bright_style) - proper titles/units on
every axis, no default matplotlib grey-on-grey look.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import objectives_api as obj_api

DPI = 600


def _integer_x_axis(ax):
    """Generation/step counts are always whole numbers - without this,
    matplotlib's default tick locator can pick fractional ticks (0.00,
    0.25, 0.50, ...) when there are only a few of them."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# Human-readable (title, axis-label-with-units) per current OBJECTIVE_NAMES
# entry - used everywhere a plot needs to show an objective by name instead
# of its raw dict key.
_OBJECTIVE_DISPLAY = {
    "f1_folded_gait_velocity": ("Folded-Gait Velocity", "Velocity - m/s"),
    "f2_entropy": ("Entropy - Folding-Complexity Gain", "Entropy Δ = H₃D - H₂D"),
    "f3_pheromone_yaw_response": ("Pheromone Yaw Response", "Yaw Response - deg"),
    "f4_pheromone_speed_response": ("Pheromone Speed Response", "Speed Response - Δv / v"),
}

# One consistent bright color per current objective, reused across fitness
# trends / entropy plot / anywhere else a specific objective needs a fixed
# identity color instead of an arbitrary cycle color.
_OBJECTIVE_COLOR = {
    "f1_folded_gait_velocity": "#1f77ff",   # bright blue
    "f2_entropy": "#22b14c",                # bright green
    "f3_pheromone_yaw_response": "#e8382b", # bright red
    "f4_pheromone_speed_response": "#a349e6",  # bright purple
}

_RL_METRIC_COLOR = {
    "reward": "#1f77ff",
    "policy_loss": "#e8382b",
    "value_loss": "#a349e6",
    "entropy": "#22b14c",
}


def _apply_bright_style():
    """One shared look for every figure in this module: white background,
    bold titles, a light grid, readable font sizes - applied once at
    import time (matplotlib rcParams are process-global)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "font.size": 10,
        "lines.linewidth": 2.0,
    })


_apply_bright_style()


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


def _load_all_generations(out_dir):
    """Reads out_dir/population_history.json (written by
    append_generation_population), returning [(gen_idx, population)]
    sorted by generation index - the shared data source for every plot
    below. Re-reads from disk each call (rather than taking an accumulated
    in-memory history) so these plots stay correct even if a run is
    resumed or these functions are called standalone.

    Every entry's objectives dict is run through
    objectives_api.migrate_legacy_objectives() here, ONCE, centrally - so
    every plot function below only ever has to know about the CURRENT
    OBJECTIVE_NAMES schema, whether the underlying file was written by
    this code or an older pre-renumbering run (see that function's
    docstring)."""
    history = _load_population_history(out_dir)
    generations = [(h["generation"], h["population"]) for h in sorted(history, key=lambda h: h["generation"])]
    for _, pop in generations:
        for entry in pop:
            entry["objectives"] = obj_api.migrate_legacy_objectives(entry["objectives"])
    return generations


def _population_costs_and_ranks(pop):
    """(costs, rank): costs is an (n, len(OBJECTIVE_NAMES)) array, each
    column min-max normalized to [0, 1] where LOWER is always better
    (objectives_api.to_minimization_vector's convention, so a "maximize"
    objective like f1 is sign-flipped first) - a shared, direction-
    agnostic scale so every pairwise Pareto panel below reads the same
    way ("down-and-left is better") regardless of which axis is which.
    rank is pymoo's non-dominated sort front index per individual (0 =
    the Pareto front itself, larger = more dominated)."""
    F = np.array([obj_api.to_minimization_vector(e["objectives"]) for e in pop])
    value_range = np.ptp(F, axis=0)
    value_range[value_range == 0] = 1.0
    costs = (F - F.min(axis=0)) / value_range

    fronts = NonDominatedSorting().do(F)
    rank = np.zeros(len(pop), dtype=int)
    for r, front in enumerate(fronts):
        rank[front] = r
    return costs, rank


def plot_pareto_front_last_gen(out_dir, filename="pareto_front_last_gen.png"):
    """Single figure, 4 panels (f1 vs f2, f2 vs f3, f3 vs f4, f4 vs f1),
    for the MOST RECENT generation only - replaces the old per-generation
    parallel-coordinates plot_pareto_front(). Each axis is min-max
    normalized within this generation's population; points are colored by
    NSGA-III non-dominated rank (see _population_costs_and_ranks), with
    rank-0 (the actual Pareto front) highlighted in a distinct color.

    No line joins the rank-0 points: each panel is a 2D slice of the full
    4-objective front, and there's no guarantee that slice is monotonic
    the way a true 2-objective front is, so a connecting line would imply
    a curve shape that isn't actually there.

    With only 4 objectives, it's common (not a bug) for MOST or ALL of a
    modest-sized population to land in rank 0 at once ("dominance
    resistance" - each individual wins on at least one objective vs. every
    other) - NSGA-III actively preserves non-dominated individuals, which
    reinforces this. More generations don't reliably reduce that; a larger
    population relative to the objective count is what gives the
    reference-direction niching room to actually differentiate within
    rank 0."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None
    gen_idx, pop = generations[-1]
    if len(pop) < 2:
        return None

    names = obj_api.OBJECTIVE_NAMES
    titles = [_OBJECTIVE_DISPLAY[n][0] for n in names]
    costs, rank = _population_costs_and_ranks(pop)
    max_rank = int(rank.max()) if len(rank) else 0

    # Ranks 1-4 ("near front") get one flat light-blue color rather than
    # their own gradient step - common practice for eyeballing the top few
    # fronts together (see plot_pareto_front_last_gen's docstring for why
    # rank 0 alone isn't always the most useful cut with 4 objectives).
    # Rank 0 keeps its own distinct highlight color on top so the actual
    # front is still identifiable at a glance, not folded into the same
    # band as ranks 1-4.
    NEAR_FRONT_MAX_RANK = 4
    NEAR_FRONT_COLOR = "#8ecfff"

    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    cmap = plt.cm.plasma

    for ax, (i, j) in zip(axes, pairs):
        far = rank > NEAR_FRONT_MAX_RANK
        if far.any():
            ax.scatter(
                costs[far, i], costs[far, j], c=rank[far],
                cmap=cmap, vmin=NEAR_FRONT_MAX_RANK + 1, vmax=max(max_rank, NEAR_FRONT_MAX_RANK + 1),
                s=50, alpha=0.85, edgecolors="#333333", linewidths=0.5,
                label=f"Rank > {NEAR_FRONT_MAX_RANK}",
            )
        near_front = (rank > 0) & (rank <= NEAR_FRONT_MAX_RANK)
        if near_front.any():
            ax.scatter(
                costs[near_front, i], costs[near_front, j], color=NEAR_FRONT_COLOR, s=60,
                edgecolors="#2a6f9e", linewidths=0.6, zorder=2, label=f"Rank 1-{NEAR_FRONT_MAX_RANK}",
            )
        # No connecting line here on purpose: a line only implies a real
        # curve for a genuine 2-objective front. This is a 2D slice of a
        # 4D front (see plot_pareto_front_last_gen's docstring) - nothing
        # says that projection is monotonic, so joining rank-0 points by
        # increasing x previously drew a zigzag that implied a shape that
        # isn't actually there. Scatter only.
        front = rank == 0
        ax.scatter(
            costs[front, i], costs[front, j], color="#00e5e5", s=90,
            edgecolors="#005050", linewidths=0.8, zorder=3, label="Pareto front - rank 0",
        )
        ax.set_xlabel(titles[i])
        ax.set_ylabel(titles[j])
        # Short title only - the full descriptive names are already on the
        # axis labels; the combined "f1 vs f2: <name> vs <name>" version
        # was long enough to visually collide with the adjacent subplot's
        # title in the same row.
        ax.set_title(f"f{i + 1} vs f{j + 1}", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=10)
    fig.suptitle(f"Pareto Front - Final Generation {gen_idx}", fontsize=15, fontweight="bold", y=1.06)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto_parallel_coordinates(out_dir, filename="pareto_parallel_coordinates.png"):
    """Parallel-coordinates view of the same last-generation population as
    plot_pareto_front_last_gen: one axis per objective (f1..f4), one line
    per individual crossing all four - shows tradeoffs across all
    objectives at once instead of one pair at a time.

    Each axis plots costs' complement (1 - min-max-normalized cost), so
    "up" always means "better" on every axis regardless of that
    objective's own maximize/minimize direction - same normalization
    source (_population_costs_and_ranks) and the same 3-tier rank
    coloring as plot_pareto_front_last_gen, so the two plots read
    consistently: rank 0 (the actual front) in distinct cyan, ranks 1-4 in
    flat light blue, rank 5+ in a plasma gradient by rank."""
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
    ax.set_xticklabels(titles, fontsize=10)
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
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fitness_trends(out_dir):
    """One subplot per objective (2x2 grid, f1..f4): each generation's
    BEST value (in that objective's own "higher/lower is better" direction
    per objectives_api.MAXIMIZE) - a single clean line, no markers, no mean
    line (the mean is a population-homogeneity signal, which
    plot_convergence already covers) - just "is the best individual
    actually getting better," with proper units on every y-axis."""
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
        ax.plot(gen_indices, best_vals, color=_OBJECTIVE_COLOR[name])
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        _integer_x_axis(ax)

    fig.suptitle("Fitness Trends Across Generations - Best Individual per Generation", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, "fitness_trends.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_entropy_vs_velocity(out_dir, filename="entropy_vs_velocity.png"):
    """AO-2's "relationship between morphology complexity and locomotion"
    view - every individual across every generation, one point each, f2
    (entropy / folding-complexity gain) against f1 (folded-gait velocity),
    colored by which generation it's from so a temporal drift is still
    visible even though this isn't a per-generation line chart. Pooling
    every individual (not just per-generation means) is what actually lets
    a real correlation between the two objectives show up - averaging per
    generation first would wash that out."""
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
    ax.set_title("Entropy - Folding-Complexity Gain vs. Folded-Gait Velocity", fontsize=15, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Generation")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_convergence(out_dir):
    """Best vs. population-mean SCALARIZED fitness (objectives_api.scalarize
    - the equal-weight, normalized combination across f1..f4) per
    generation: the classic GA "convergence" view - a healthy run's mean
    climbs toward the best line as the population homogenizes around good
    solutions."""
    generations = _load_all_generations(out_dir)
    if not generations:
        return None

    gen_indices = [g for g, _ in generations]
    best_vals, mean_vals = [], []
    for _, pop in generations:
        scalarized = [obj_api.scalarize(entry["objectives"]) for entry in pop]
        best_vals.append(max(scalarized))
        mean_vals.append(float(np.mean(scalarized)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(gen_indices, best_vals, color="#1f77ff", label="Best")
    ax.plot(gen_indices, mean_vals, color="#e8382b", linestyle="--", label="Mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Scalarized fitness")
    ax.set_title("GA Convergence - Best vs. Mean Population Fitness", fontsize=15, fontweight="bold")
    ax.legend(loc="best")
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=DPI)
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
    run's f1 (locomotion), f2 (shape-entropy delta), scalarized fitness
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

    run_colors = ["#1f77ff", "#e8382b", "#22b14c", "#a349e6"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 16))
    ax_f1, ax_f2, ax_conv, ax_collision = axes

    for color, (label, (generations, gen_stats)) in zip(run_colors, runs.items()):
        gen_indices = [g for g, _ in generations]

        best_f1 = [max(e["objectives"].get("f1_folded_gait_velocity", 0.0) for e in pop) for _, pop in generations]
        mean_f1 = [float(np.mean([e["objectives"].get("f1_folded_gait_velocity", 0.0) for e in pop])) for _, pop in generations]
        ax_f1.plot(gen_indices, best_f1, color=color, label=f"{label} - best")
        ax_f1.plot(gen_indices, mean_f1, color=color, linestyle="--", alpha=0.7, label=f"{label} - mean")

        mean_f2 = [float(np.mean([e["objectives"].get("f2_entropy", 0.0) for e in pop])) for _, pop in generations]
        ax_f2.plot(gen_indices, mean_f2, color=color, label=f"{label} - mean")

        scalarized_best = [max(obj_api.scalarize(e["objectives"]) for e in pop) for _, pop in generations]
        ax_conv.plot(gen_indices, scalarized_best, color=color, label=label)

        collision_rate = [
            gen_stats[g]["n_collided"] / gen_stats[g]["n_offspring"]
            if g in gen_stats and gen_stats[g]["n_offspring"] else None
            for g in gen_indices
        ]
        if any(v is not None for v in collision_rate):
            xs = [g for g, v in zip(gen_indices, collision_rate) if v is not None]
            ys = [v for v in collision_rate if v is not None]
            ax_collision.plot(xs, ys, color=color, label=label)

    ax_f1.set_title("f1: Folded-Gait Velocity - Locomotion", fontsize=13, fontweight="bold")
    ax_f1.set_ylabel("Velocity - m/s")
    ax_f1.legend(loc="best", fontsize=8)
    _integer_x_axis(ax_f1)

    ax_f2.set_title("f2: Entropy - Folding-Complexity Gain", fontsize=13, fontweight="bold")
    ax_f2.set_ylabel("Entropy Δ")
    ax_f2.legend(loc="best", fontsize=8)
    _integer_x_axis(ax_f2)

    ax_conv.set_title("Convergence: Best Scalarized Fitness", fontsize=13, fontweight="bold")
    ax_conv.set_ylabel("Scalarized fitness")
    ax_conv.legend(loc="best", fontsize=8)
    _integer_x_axis(ax_conv)

    ax_collision.set_title("Collision Rate - n_collided / n_offspring", fontsize=13, fontweight="bold")
    ax_collision.set_xlabel("Generation")
    ax_collision.set_ylabel("Collision rate")
    ax_collision.legend(loc="best", fontsize=8)
    _integer_x_axis(ax_collision)

    fig.suptitle("RL-Assisted vs. Random-Baseline Genetic Operations", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(comparison_out_dir, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_rl_diagnostics(history, out_dir):
    """One SEPARATE PNG per PPO training metric (reward, policy_loss,
    value_loss, entropy) - previously a single 4x1 combined figure.
    history: rl_api.PPOTrainer.history (dict of lists). Returns
    dict[metric_key -> path] for whichever metrics had data."""
    os.makedirs(out_dir, exist_ok=True)
    specs = [
        ("reward", "Reward per Mutation Step", "Mutation Step", "Reward"),
        ("policy_loss", "Actor Policy Loss", "PPO Update Step", "Policy Loss"),
        ("value_loss", "Critic Value Loss", "PPO Update Step", "Value Loss"),
        ("entropy", "Policy Entropy", "PPO Update Step", "Entropy"),
    ]
    paths = {}
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

        path = os.path.join(out_dir, f"rl_diagnostics_{key}.png")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        paths[key] = path
    return paths
