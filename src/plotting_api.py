"""
Plotting & Artifacts API - saves per-generation population JSON, a Pareto
front plot (last generation, pairwise objective panels), per-fitness trend
lines, a GA convergence plot, an entropy-vs-velocity scatter, and RL (PPO)
training diagnostics (one PNG per metric) across a run.

All figures are saved at DPI (300, "print quality") with a shared bright,
high-contrast style (see _apply_bright_style) - proper titles/units on
every axis, no default matplotlib grey-on-grey look.
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


def _save_fig(fig, path, dpi=DPI, **savefig_kwargs):
    """Saves and closes `fig`, but never lets a failed write (the PNG open
    in an image viewer/IDE preview, a virus scanner or sync client holding
    a transient lock - the OSError this raises on Windows varies:
    PermissionError, or "[Errno 22] Invalid argument") propagate out of a
    plotting call. main.py calls these once per generation, AFTER the
    (expensive) simulation/breeding work is done and BEFORE
    checkpoint.save() - an uncaught exception here would abort the whole
    generation loop and force a resume to re-run that entire generation's
    simulation just to regenerate a picture. Logs a warning and returns
    None instead.

    dpi defaults to this module's usual DPI=600 "print quality" - pass an
    explicit override (e.g. the RL-vs-baseline comparison plots' 300) for a
    figure that's meant to be a quick screen/slide look rather than
    archival print output."""
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
    "entropy_coef": "#ff8c00",
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


def save_pareto_archive(archive, out_dir):
    """Writes out_dir/pareto_archive.json - moo_api.update_pareto_archive's
    current archive (list of dict(graph, objectives, F)), serialized the
    same way append_generation_population serializes a graph
    (nx.node_link_data). Unlike population_history.json this OVERWRITES
    rather than appends each call: the archive is already the complete,
    deduplicated, up-to-date non-dominated set as of this generation, not
    a per-generation increment - there's nothing to accumulate on top of
    it. Called every generation from main.py right after run_generation,
    same as population_history.json, so a resumed run can reload it via
    load_pareto_archive() instead of restarting the archive empty."""
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
    """Inverse of save_pareto_archive - returns [] if the run hasn't
    written one yet (fresh run, or a run from before this archive
    existed), otherwise the archive in the same dict(graph, objectives, F)
    shape moo_api.update_pareto_archive expects, with `graph` restored to
    an nx.DiGraph (nx.node_link_graph) and `F` restored to an np.ndarray."""
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


def plot_pareto_front_last_gen(out_dir, filename="pareto_front_last_gen.png", n_generations=5):
    """Single figure, 4 panels (f1 vs f2, f2 vs f3, f3 vs f4, f4 vs f1),
    overlaying up to `n_generations` generations' own Pareto fronts (rank-0,
    non-dominated members only - NonDominatedSorting run separately per
    generation, never pooled across generations), evenly spaced across the
    run (always including the first and last recorded generation) and
    colored by generation (like plot_entropy_vs_velocity) - so how the
    front actually MOVED over the run is visible directly, not just its
    final-generation position.

    Deliberately NOT "every individual from a few generations, colored by
    generation" (that's already plot_entropy_vs_velocity, for f1 vs f2) -
    keeping only each generation's rank-0 members is what keeps this a
    Pareto-front plot rather than a general population-drift scatter: only
    non-dominated points are ever shown, generation by generation.

    Every panel is normalized against ONE shared basis - each objective's
    min/max across every individual, every rank, every generation in the
    WHOLE run (not just the plotted generations) - not the last-gen-only
    per-population normalization _population_costs_and_ranks uses
    elsewhere. A per-generation basis would make each selected
    generation's front independently stretch to fill the full 0-1 axis
    regardless of whether it actually improved, hiding the very movement
    this plot exists to show.

    No line joins a front's points: each panel is a 2D slice of the full
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

    # Shared normalization basis: every individual, every rank, every
    # generation in the run - see docstring for why this can't be
    # per-generation once multiple generations share the same axes.
    all_F = np.array([
        obj_api.to_minimization_vector(entry["objectives"])
        for _, pop in generations for entry in pop
    ])
    if len(all_F) == 0:
        return None
    value_range = np.ptp(all_F, axis=0)
    value_range[value_range == 0] = 1.0
    f_min = all_F.min(axis=0)

    # Up to n_generations, evenly spaced by index across every recorded
    # generation, always including the first and last.
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
        # Short title only - the full descriptive names are already on the
        # axis labels; the combined "f1 vs f2: <name> vs <name>" version
        # was long enough to visually collide with the adjacent subplot's
        # title in the same row.
        ax.set_title(f"f{i + 1} vs f{j + 1}", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.05, pad=0.02)
        cbar.set_label("Generation")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    gens_str = ", ".join(str(g) for g, _ in selected)
    fig.suptitle(f"Pareto Front Across Generations ({gens_str})", fontsize=15, fontweight="bold")

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def plot_pareto_parallel_coordinates(out_dir, filename="pareto_parallel_coordinates.png"):
    """Parallel-coordinates view of the MOST RECENT generation's population
    only (plot_pareto_front_last_gen moved to overlaying several
    generations' own fronts instead - see its docstring - so this is no
    longer the "same population" as that plot): one axis per objective
    (f1..f4), one line per individual crossing all four - shows tradeoffs
    across all objectives at once instead of one pair at a time.

    Each axis plots costs' complement (1 - min-max-normalized cost), so
    "up" always means "better" on every axis regardless of that
    objective's own maximize/minimize direction - normalized within this
    one generation's population (_population_costs_and_ranks), with the
    same 3-tier rank coloring this file used to also use for
    plot_pareto_front_last_gen: rank 0 (the actual front) in distinct
    cyan, ranks 1-4 in flat light blue, rank 5+ in a plasma gradient by
    rank."""
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
    return _save_fig(fig, path, bbox_inches="tight")


def plot_fitness_trends(out_dir):
    """One subplot per objective (2x2 grid, f1..f4): each generation's BEST
    value (in that objective's own "higher/lower is better" direction per
    objectives_api.MAXIMIZE) as a solid line, PLUS the population MEAN
    (raw arithmetic mean of that objective, not sign-corrected - unlike
    "best," a mean has no direction to chase, it's a homogeneity readout)
    as a dashed line - so a gap that isn't closing (mean far below best)
    is visible per-OBJECTIVE here, not just once as one scalarized number
    in plot_convergence."""
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
        ax.legend(loc="best", fontsize=8)
        _integer_x_axis(ax)

    fig.suptitle("Fitness Trends Across Generations - Best vs. Mean per Objective", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, "fitness_trends.png")
    return _save_fig(fig, path)


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
    return _save_fig(fig, path)


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
    return _save_fig(fig, path)


# HV's reference point must weakly dominate (be worse than) every point
# it's ever handed - since every front below is normalized into [0, 1]
# (lower-is-better) against the whole run's own observed range, a fixed
# point just past that range's worst possible corner works for every
# generation without needing per-call retuning.
_HV_REF_POINT_MARGIN = 1.05


def _run_wide_normalized(entries):
    """(n, len(OBJECTIVE_NAMES)) array: objectives_api.to_minimization_vector
    per entry, then min-max normalized against the range observed ACROSS
    ALL of `entries` (not one generation's own local range, unlike
    _population_costs_and_ranks) - so hypervolume means the same thing
    regardless of which generation computed it, comparable as a trend
    across the whole run. Recomputed fresh from population_history.json
    every call - no dependency on objectives_api's in-memory running-
    min/max state, so this gives the same answer whether called from
    inside the live training process or standalone later (e.g. from
    evolution_results_visualizer.py)."""
    F = np.array([obj_api.to_minimization_vector(e["objectives"]) for e in entries])
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    return (F - lo) / span


def compute_hypervolume(out_dir):
    """Per-generation Hypervolume (HV) - computed fresh from
    population_history.json every call (like every other function in
    this module - see _load_all_generations' docstring).

    HV is the standard multi-objective quality indicator for a real
    engineering problem like this one: unlike GD/IGD/IGD+, it only needs
    a reference POINT, not a reference FRONT, so it doesn't need a known
    "true" Pareto front (which doesn't exist here - this isn't a
    benchmark test function) or an empirically-best-known substitute
    (which would partly measure the run against its own output - a
    circularity GD/IGD/IGD+ can't avoid without real ground truth, which
    is why this module deliberately doesn't compute them).

    Returns dict(generations=[...], hv=[...]), or None if there's no
    population_history.json yet."""
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
    """Hypervolume trend across generations - see compute_hypervolume for
    why this module tracks HV alone and not GD/IGD/IGD+ (no known true
    Pareto front to compare against for a real engineering problem)."""
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


# One consistent color per grammar action - reused by every plot below
# that breaks something out by action type, split visually into a light
# blue/green/gold family for the 8 mutation actions vs. bold red/purple
# for the 2 crossover actions, so a crossover action is immediately
# distinguishable in a stacked/grouped view instead of blending into the
# mutation majority (see plot_action_distribution).
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
    """[(gen_idx, events)] for every generation with a breeding_events.json
    (moo_api._write_breeding_events - written once per generation,
    unconditionally, in both the RL-assisted and random-baseline arms).
    Reuses population_history.json's own generation indices (via
    _load_all_generations) rather than re-scanning generation_* folders
    independently, so the two data sources never disagree about which
    generations exist. A generation whose file is missing (e.g. an
    interrupted/partial folder) is silently skipped."""
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
    """Stacked-area chart of which grammar action moo_api.make_children()
    actually picked, as a FRACTION of that generation's breeding
    decisions, generation by generation - the automatic version of what
    previously required hand-counting breeding_events.json across every
    generation folder to notice (e.g. a crossover action's band
    collapsing toward zero within the first ~10 generations)."""
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=8)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def plot_reward_and_loss_by_action(history, out_dir, filename="rl_diagnostics_by_action.png"):
    """Two-panel bar chart breaking PPO's reward and policy loss down BY
    grammar action type, aggregated over the whole run so far - the
    direct diagnostic for WHY one action type's sampling probability
    might be collapsing (see plot_action_distribution): if its reward is
    systematically worse than other actions', that explains it rather
    than just describing it.

    `history`: rl_api.PPOTrainer.history. reward_action (set in
    PPOTrainer.record) pairs 1:1 with reward. loss_by_action (set once
    per PPOTrainer.update() call, from that update's LAST PPO epoch) is a
    list of {action_name: {policy_loss, value_loss, count}} snapshots -
    count-weighted-averaged across all of them here. Each bar is
    annotated with its sample count, since rare actions (crossover,
    especially once/if its probability collapses) can end up averaged
    over far fewer samples than common ones - a mean alone would hide
    that it's a much noisier estimate."""
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
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax_r.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_r.set_title("Mean Reward by Action Type", fontsize=13, fontweight="bold")
    ax_r.set_ylabel("Mean Reward")
    ax_r.tick_params(axis="x", rotation=30)

    policy_loss_means = [(policy_loss_sum[a] / policy_loss_n[a]) if policy_loss_n.get(a) else 0.0 for a in present]
    bars_l = ax_l.bar(labels, policy_loss_means, color=colors, edgecolor="#333333", linewidth=0.6)
    for bar, n in zip(bars_l, [policy_loss_n.get(a, 0) for a in present]):
        ax_l.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax_l.set_title("Mean Policy Loss by Action Type - Final PPO Epoch per Update", fontsize=13, fontweight="bold")
    ax_l.set_ylabel("Mean Policy Loss")
    ax_l.tick_params(axis="x", rotation=30)

    fig.suptitle("PPO Reward / Loss Broken Out by Grammar Action", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, filename)
    return _save_fig(fig, path, bbox_inches="tight")


def _load_comparison_runs(run_dirs):
    """dict[label -> [(gen_idx, population), ...]] for every run_dirs entry
    that has any data yet - shared by every plot_*_comparison function
    below. Runs with no data yet are silently dropped (so each comparison
    plot stays safe to call while one arm is still in progress)."""
    runs = {}
    for label, out_dir in run_dirs.items():
        generations = _load_all_generations(out_dir)
        if generations:
            runs[label] = generations
    return runs


def plot_convergence_comparison(run_dirs, comparison_out_dir, filename="convergence_comparison.png"):
    """Multi-run counterpart to plot_convergence(): best vs. population-mean
    SCALARIZED fitness (objectives_api.scalarize), one best/mean pair of
    lines per run, so the RL-assisted policy's convergence behavior is
    directly comparable against random_baseline.py's. Saved at this
    module's usual DPI=600 "print quality" - unlike the per-generation
    plots elsewhere in this file, this one isn't regenerated every
    generation of a run, so there's no storage-cost reason to shrink it.

    `run_dirs`: dict[label -> OUTPUT_DIR] - e.g.
    {"RL-assisted": ".../evolution_run", "Random baseline": ".../evolution_run_norl"}."""
    os.makedirs(comparison_out_dir, exist_ok=True)
    runs = _load_comparison_runs(run_dirs)
    if not runs:
        return None

    run_colors = ["#1f77ff", "#e8382b", "#22b14c", "#a349e6"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for color, (label, generations) in zip(run_colors, runs.items()):
        gen_indices = [g for g, _ in generations]
        best_vals, mean_vals = [], []
        for _, pop in generations:
            scalarized = [obj_api.scalarize(entry["objectives"]) for entry in pop]
            best_vals.append(max(scalarized))
            mean_vals.append(float(np.mean(scalarized)))
        ax.plot(gen_indices, best_vals, color=color, label=f"{label} - best")
        ax.plot(gen_indices, mean_vals, color=color, linestyle="--", alpha=0.7, label=f"{label} - mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Scalarized fitness")
    ax.set_title("GA Convergence - Best vs. Mean Population Fitness", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path)


def plot_fitness_trends_comparison(run_dirs, comparison_out_dir, filename="fitness_trends_comparison.png"):
    """Multi-run counterpart to plot_fitness_trends(): one subplot per
    objective (2x2 grid, f1..f4), each generation's BEST value (solid) AND
    population MEAN (dashed, same color) per run - so a run's best-so-far
    progress AND how far the rest of its population lags behind it are
    both visible per-objective, not just once as one scalarized number in
    plot_convergence_comparison. Saved at this module's usual DPI=600 - see
    plot_convergence_comparison's docstring for why.

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
        ax.legend(loc="best", fontsize=7)
        _integer_x_axis(ax)

    fig.suptitle("Fitness Trends Across Generations - Best vs. Mean per Objective", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path)


def plot_collision_rate_comparison(run_dirs, comparison_out_dir, filename="collision_rate_comparison.png"):
    """Collision rate (n_collided / n_offspring per generation, from
    append_generation_stats' generation_stats.json), one line per run - the
    piece of plot_rl_vs_baseline_comparison's old combined figure that's
    neither a fitness trend nor convergence, so it stays its own plot."""
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
    ax.legend(loc="best", fontsize=8)
    _integer_x_axis(ax)
    fig.tight_layout()

    path = os.path.join(comparison_out_dir, filename)
    return _save_fig(fig, path)


def plot_rl_vs_baseline_comparison(run_dirs, comparison_out_dir):
    """The RL_ASSISTED_GENETIC_OPERATIONS comparison plots: GA convergence
    (with mean), fitness trends (best only, no mean), and collision rate -
    each its own separate PNG (see plot_convergence_comparison /
    plot_fitness_trends_comparison / plot_collision_rate_comparison) - so
    the effect of the learned policy vs. random_baseline.py's blind
    variation is visible directly, generation by generation.

    `run_dirs`: dict[label -> OUTPUT_DIR] - e.g.
    {"RL-assisted": ".../evolution_run", "Random baseline": ".../evolution_run_norl"}.
    Runs with no data yet are silently skipped (so this is safe to call
    while one arm is still in progress). Returns {name: path_or_None} for
    the three files."""
    return dict(
        convergence=plot_convergence_comparison(run_dirs, comparison_out_dir),
        fitness_trends=plot_fitness_trends_comparison(run_dirs, comparison_out_dir),
        collision_rate=plot_collision_rate_comparison(run_dirs, comparison_out_dir),
    )

# A real scalarize()-delta reward (scalarize(child) - scalarize(parent))
# is mathematically bounded to [-1, 1] - see objectives_api.scalarize.
# moo_api.COLLISION_PENALTY is a worse-than-any-legitimate-outcome flat
# penalty added on top of (or instead of) that delta, so anything below
# halfway between the two clearly-separated ranges is a collision/failure
# event, not real signal - separates the two for plot_rl_diagnostics'
# reward panel. Derived from the actual constant (not a hardcoded copy of
# its value) so the two can never silently drift out of sync again if
# COLLISION_PENALTY is retuned.
_COLLISION_REWARD_THRESHOLD = (moo_api.COLLISION_PENALTY - 1.0) / 2


def plot_rl_diagnostics(history, out_dir):
    """One SEPARATE PNG per PPO training metric (reward, policy_loss,
    value_loss, entropy) - previously a single 4x1 combined figure.
    history: rl_api.PPOTrainer.history (dict of lists). Returns
    dict[metric_key -> path] for whichever metrics had data.

    The reward metric gets a 2-panel figure instead of the generic
    single-panel treatment: full range (so collision-gate failures -
    moo_api.COLLISION_PENALTY - are still visible as spikes) on top, and
    the same series with those spikes excluded and the y-axis rescaled to
    fit what remains on the bottom - see _COLLISION_REWARD_THRESHOLD.
    Without this, the real reward signal is invisible against the -10
    spikes on a single linear axis."""
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    reward_values = history.get("reward")
    if reward_values:
        values_arr = np.asarray(reward_values, dtype=float)
        collision_mask = values_arr <= _COLLISION_REWARD_THRESHOLD
        n_collision = int(collision_mask.sum())
        steps = np.arange(len(values_arr))

        fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(8, 9))
        ax_full.plot(steps, values_arr, color=_RL_METRIC_COLOR["reward"], linewidth=1.0)
        ax_full.set_title("Reward per Mutation Step - Full Range", fontsize=13, fontweight="bold")
        ax_full.set_xlabel("Mutation Step")
        ax_full.set_ylabel("Reward")

        real_steps = steps[~collision_mask]
        real_values = values_arr[~collision_mask]
        ax_zoom.plot(real_steps, real_values, color=_RL_METRIC_COLOR["reward"], linewidth=1.0)
        if len(real_values):
            span = max(float(real_values.max() - real_values.min()), 0.05)
            pad = 0.1 * span
            ax_zoom.set_ylim(real_values.min() - pad, real_values.max() + pad)
        ax_zoom.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        ax_zoom.set_title(
            f"Reward per Mutation Step - {n_collision} Collision-Gate Failure(s) Excluded",
            fontsize=13, fontweight="bold",
        )
        ax_zoom.set_xlabel("Mutation Step")
        ax_zoom.set_ylabel("Reward")
        fig.tight_layout()

        path = os.path.join(out_dir, "rl_diagnostics_reward.png")
        saved = _save_fig(fig, path)
        if saved:
            paths["reward"] = saved

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

        path = os.path.join(out_dir, f"rl_diagnostics_{key}.png")
        saved = _save_fig(fig, path)
        if saved:
            paths[key] = saved
    return paths
