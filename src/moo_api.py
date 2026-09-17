"""
MOO API - Multi-Objective Optimization Engine (RL-guided NSGA-III). Owns the
population loop: Sobol-seeded initial genotypes, breeding (mutation and
crossover picked/parameterized by rl_api.py's actor/critic), batched
parallel MuJoCo evaluation (sim_executor.py), and NSGA-III environmental
selection via pymoo's ReferenceDirectionSurvival (driven "by hand" since
genotypes are variable-size graphs, not pymoo's usual fixed-length vector).

Every genotype here is only HALF the final shape - symmetry.py mirrors it
into the full bilaterally-symmetric morphology right before an assembly is
built (see _prepare_assembly).
"""

import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile

import networkx as nx
import numpy as np
from scipy.stats import qmc
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

import objectives_api as obj_api
import random_baseline
import rl_api
import roblet_grammar as rg
import sim_executor
import symmetry

logger = logging.getLogger(__name__)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER_SCRIPTS_DIR = os.path.join(_SRC_DIR, "..", "helper_scripts")
if _HELPER_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _HELPER_SCRIPTS_DIR)

from mjcf_generator import build_assembly, ModuleCollisionError  # noqa: E402

_MESHDIR = os.path.abspath(os.path.join(_SRC_DIR, "..", "meshes")).replace("\\", "/")

# Historical RL-reward penalty for a colliding child (no longer applied - see
# run_generation's docstring on why collision outcomes are now excluded
# from the reward stream entirely - kept as a documented reference value).
COLLISION_PENALTY = -2.0

# A stand-in stats dict for a graph that couldn't even be built into a valid
# MJCF (ModuleCollisionError) - scores as failed/infeasible like any other collision.
_FAILED_STATS = {"success": 0, "physics_ok": 0, "is_stable": 0, "average_velocity_mmps": 0.0, "total_distance_mm": 0.0}

# graph content-hash -> {"stats": ..., "screenshot_path": ...}. Physics is
# deterministic, so an unchanged genotype (e.g. a carried-over survivor)
# skips re-simulation. Keyed by content hash, not id() (which can be reused).
_EVAL_CACHE = {}


def _graph_hash(G):
    """Deterministic content hash of a genotype - _EVAL_CACHE's key."""
    payload = json.dumps(nx.node_link_data(G, edges="edges"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_collision_free(G, scratch_dir, rng):
    """True if the FULL mirrored graph (symmetry.build_symmetric_graph(G))
    builds cleanly through mjcf_generator's 3D collision check - checking the
    mirrored graph matters since the two halves can collide with each other
    even when the half alone is fine. Also repairs `G` in place
    (rg.ensure_min_light_sensitive) before mirroring."""
    os.makedirs(scratch_dir, exist_ok=True)
    rg.ensure_min_light_sensitive(G, rng)
    try:
        full_G = symmetry.build_symmetric_graph(G)
    except symmetry.MirrorAnchorViolation:
        return False
    graph_json_path = os.path.join(scratch_dir, "_collision_check_graph.json")
    xml_path = os.path.join(scratch_dir, "_collision_check_assembly.xml")
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(full_G, edges="edges"), f)
    try:
        build_assembly(graph_json_path, xml_path, meshdir=_MESHDIR)
        return True
    except ModuleCollisionError:
        return False


def _build_collision_free_seed(rng, n_modules, type_weights, hinge_angle_fn, scratch_dir, max_attempts):
    """Resamples a fresh random_seed_graph at `n_modules` up to `max_attempts`
    times looking for one that passes _is_collision_free(). If every attempt
    collides, halves the module count and tries again (down to MIN_MODULES)."""
    candidate_n = max(rg.MIN_MODULES, n_modules)
    while True:
        for _ in range(max_attempts):
            type_choices = [rng.choices(rg.MODULE_TYPES, weights=type_weights, k=1)[0] for _ in range(candidate_n)]
            G = rg.random_seed_graph(rng, candidate_n, module_type_choices=type_choices,
                                      hinge_angle_fn=hinge_angle_fn)
            if _is_collision_free(G, scratch_dir, rng):
                return G
        if candidate_n <= rg.MIN_MODULES:
            logger.warning(
                "No collision-free seed graph found at MIN_MODULES=%d after %d attempts; "
                "using the last candidate anyway (evaluate_population()'s ModuleCollisionError "
                "handling remains a safety net).", rg.MIN_MODULES, max_attempts,
            )
            return G
        logger.info("No collision-free seed graph at %d modules after %d attempts; shrinking.",
                    candidate_n, max_attempts)
        candidate_n = max(rg.MIN_MODULES, candidate_n // 2)


def sobol_seed_population(pop_size, seed=0, scratch_dir=None, max_attempts=15):
    """Sobol-sampled initial genotypes: a 3D Sobol sequence over (module_count,
    hinge_angle_bias, fold_type_bias) gives a uniform, low-discrepancy spread
    across the design-variable space. Every returned graph is pre-validated
    collision-free (see _build_collision_free_seed)."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    n = 1 << max(1, (pop_size - 1).bit_length())  # Sobol is balanced at powers of two
    draws = sampler.random(n)[:pop_size]
    rng = random.Random(seed)
    scratch_dir = scratch_dir or tempfile.mkdtemp(prefix="roblet_seed_check_")

    population = []
    for module_count_u, hinge_u, fold_u in draws:
        n_modules = int(round(rg.MIN_MODULES + module_count_u * (rg.MAX_MODULES - rg.MIN_MODULES)))
        base_angle = rg.MIN_HINGE_ANGLE + hinge_u * (rg.MAX_HINGE_ANGLE - rg.MIN_HINGE_ANGLE)

        def hinge_angle_fn(base_angle=base_angle):
            return float(np.clip(rng.gauss(base_angle, 15.0), rg.MIN_HINGE_ANGLE, rg.MAX_HINGE_ANGLE))

        weights = [1.0, 1.0 + 2 * fold_u, 1.0 + 2 * (1 - fold_u)]  # biases Mountain vs. valley fold
        G = _build_collision_free_seed(rng, n_modules, weights, hinge_angle_fn, scratch_dir, max_attempts)
        population.append(G)
    return population


def _prepare_assembly(G, work_dir, tag):
    """Mirrors the half-genotype `G` into the full symmetric shape, writes its
    graph JSON, and calls build_assembly. Returns an xml_path, or None if
    the full (mirrored) graph is invalid (ModuleCollisionError or
    symmetry.MirrorAnchorViolation) - callers treat that as a failed
    simulation. Defensive backstop; graphs reaching here already passed
    _is_collision_free() at creation time."""
    try:
        full_G = symmetry.build_symmetric_graph(G)
    except symmetry.MirrorAnchorViolation:
        return None
    graph_json_path = os.path.join(work_dir, f"{tag}_graph.json")
    xml_path = os.path.join(work_dir, f"{tag}_assembly.xml")
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(full_G, edges="edges"), f)
    try:
        build_assembly(graph_json_path, xml_path, meshdir=_MESHDIR)
    except ModuleCollisionError:
        return None
    return xml_path


# Each evaluate_population() call re-simulates up to this many cache hits
# anyway, as a "trust but verify" check that _EVAL_CACHE hasn't gone stale.
_CACHE_SPOT_CHECK_MAX_PER_CALL = 1
# A cache entry is flagged stale if a fresh re-simulation's avg_velocity_mmps
# disagrees with the cached value by more than this fraction (physics is
# meant to be deterministic, so any consistent drift is worth flagging).
_CACHE_SPOT_CHECK_TOLERANCE = 0.10


def evaluate_population(graphs, work_dir, sim_seconds=7.0, max_workers=None):
    """Writes an MJCF assembly for every graph, runs every valid one through
    roblet_simulator.py --headless in parallel OS processes (sim_executor.py),
    then scores each from its stats.json. Returns a list of
    (objectives, f_vec, constraint) aligned to `graphs`' order.
    Individuals already simulated in an earlier generation (_EVAL_CACHE,
    e.g. a carried-over survivor) skip the MuJoCo run, except for a few
    spot-checked anyway (see _CACHE_SPOT_CHECK_MAX_PER_CALL)."""
    os.makedirs(work_dir, exist_ok=True)

    stats_paths = [None] * len(graphs)
    hashes = [None] * len(graphs)
    jobs = []
    cache_hit_indices = []
    for i, G in enumerate(graphs):
        xml_path = _prepare_assembly(G, work_dir, tag=f"ind{i}")
        if xml_path is None:
            continue
        stats_path = os.path.join(work_dir, f"ind{i}_stats.json")
        stats_paths[i] = stats_path
        h = _graph_hash(G)
        hashes[i] = h

        cached = _EVAL_CACHE.get(h)
        if cached is None:
            jobs.append((xml_path, stats_path))
            continue

        cache_hit_indices.append(i)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(cached["stats"], f, indent=4)
        src_screenshot = cached.get("screenshot_path")
        if src_screenshot and os.path.exists(src_screenshot):
            try:
                shutil.copyfile(src_screenshot, os.path.join(work_dir, f"screenshot_ind{i}_assembly.png"))
            except OSError:
                logger.warning("Could not copy forward cached screenshot for ind%d", i)

    # Redirect a few cache hits back into `jobs` for a fresh re-simulation -
    # their stats_path already has the cached copy written above, which the
    # fresh run below simply overwrites once it finishes.
    spot_checked = random.sample(cache_hit_indices, min(_CACHE_SPOT_CHECK_MAX_PER_CALL, len(cache_hit_indices)))
    spot_check_old_stats = {}
    for i in spot_checked:
        spot_check_old_stats[i] = _EVAL_CACHE[hashes[i]]["stats"]
        xml_path = os.path.join(work_dir, f"ind{i}_assembly.xml")
        jobs.append((xml_path, stats_paths[i]))

    logger.info(
        "Evaluating population: %d graphs, work_dir=%s (%d cache hits, %d simulated, %d spot-checked)",
        len(graphs), work_dir, len(cache_hit_indices) - len(spot_checked), len(jobs), len(spot_checked),
    )
    sim_executor.run_batch(jobs, max_workers=max_workers, max_sim_time=sim_seconds)
    logger.info("Finished simulation batch: %d jobs", len(jobs))

    results = []
    for i, stats_path in enumerate(stats_paths):
        if stats_path is None:
            stats = dict(_FAILED_STATS)
        else:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            h = hashes[i]
            if i in spot_check_old_stats:
                old_v = spot_check_old_stats[i].get("avg_velocity_mmps", 0.0)
                new_v = stats.get("avg_velocity_mmps", 0.0)
                if abs(new_v - old_v) > _CACHE_SPOT_CHECK_TOLERANCE * max(abs(old_v), abs(new_v), 1e-9):
                    logger.warning(
                        "Stale/wrong _EVAL_CACHE entry caught by spot check on ind%d (hash %s): "
                        "cached avg_velocity_mmps=%.4f, fresh re-simulation=%.4f - overwriting the "
                        "cache entry with the fresh result.",
                        i, h[:12], old_v, new_v,
                    )
                    screenshot_path = os.path.join(work_dir, f"screenshot_ind{i}_assembly.png")
                    _EVAL_CACHE[h] = dict(
                        stats=stats,
                        screenshot_path=screenshot_path if os.path.exists(screenshot_path) else None,
                    )
            elif h is not None and h not in _EVAL_CACHE:
                screenshot_path = os.path.join(work_dir, f"screenshot_ind{i}_assembly.png")
                _EVAL_CACHE[h] = dict(
                    stats=stats,
                    screenshot_path=screenshot_path if os.path.exists(screenshot_path) else None,
                )
        objectives = obj_api.compute_objectives(stats)
        f_vec = obj_api.to_minimization_vector(objectives)
        constraint = obj_api.collision_constraint(stats)
        results.append((objectives, f_vec, constraint))
    return results


def evaluate_individual(G, work_dir=None, sim_seconds=7.0):
    """Convenience single-graph wrapper around evaluate_population (no
    parallelism benefit for just one graph - useful for quick checks)."""
    work_dir = work_dir or tempfile.mkdtemp(prefix="roblet_eval_")
    (result,) = evaluate_population([G], work_dir, sim_seconds=sim_seconds, max_workers=1)
    return result


def make_children(parent_a, parent_b, ppo_trainer, rng, rl_assisted=True):
    """One breeding step. `rl_assisted` picks mutation vs. crossover (and its
    params) via the trained RL policy; otherwise random_baseline.py draws
    uniformly over the same action space. Returns (children, decision_or_None):
    1 child for a mutation/GRAFT_SUBTREE, 2 for a SWAP_SUBTREES; `decision`
    is None only if nothing was legal (falls back to a same-graph copy)."""
    if not rl_api.has_any_legal_action(parent_a, parent_b):
        return [parent_a.copy()], None

    if rl_assisted:
        decision = ppo_trainer.select_action(parent_a, parent_b)
    else:
        decision = random_baseline.act(parent_a, parent_b, rng)
    children = rl_api.apply_decision(parent_a, parent_b, decision)
    return children, decision


def make_children_collision_free(parent_a, parent_b, ppo_trainer, rng, scratch_dir,
                                  max_attempts=8, rl_assisted=True):
    """Wraps make_children() with a 3D collision gate: a colliding
    decision is rejected and re-sampled, up to `max_attempts` (no RL reward
    recorded for rejected attempts - see run_generation's docstring). Falls
    back to a no-op copy of parent_a if every attempt still collides. Also
    absorbs roblet_grammar.GraftPortConflict, treating it like a rejection."""
    for _ in range(max_attempts):
        try:
            children, decision = make_children(parent_a, parent_b, ppo_trainer, rng, rl_assisted=rl_assisted)
        except rg.GraftPortConflict:
            continue
        if all(_is_collision_free(child, scratch_dir, rng) for child in children):
            return children, decision
    return [parent_a.copy()], None


def _write_breeding_events(work_dir, n_parents, n_offspring, events):
    """Writes `work_dir/breeding_events.json`, the per-generation lineage log
    evolution_results_visualizer.py reads. `parent_ids`/`child_ids` are
    indices into this generation's flat evaluated batch (matching the
    `ind{i}_...` filenames evaluate_population() writes)."""
    payload = dict(n_parents=n_parents, n_offspring=n_offspring, events=events)
    with open(os.path.join(work_dir, "breeding_events.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _reference_directions(n_obj, n_partitions=2):
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def _auto_ref_partitions(n_obj, pop_size):
    """Largest das-dennis n_partitions whose reference-direction count doesn't
    exceed `pop_size`, so NSGA-III's niching resolution scales with
    population size instead of staying fixed at whatever fit an earlier,
    smaller pop_size."""
    partitions = 1
    while math.comb(partitions + 1 + n_obj - 1, n_obj - 1) <= pop_size:
        partitions += 1
    return partitions


def update_pareto_archive(archive, candidates):
    """Maintains a run-wide, read-only record of every non-dominated
    (Pareto-optimal) individual ever evaluated, independent of NSGA-III's own
    environmental selection - so a good trade-off dropped from the breeding
    population (legitimate under reference-direction niching) is still kept
    for reporting/analysis. Never fed back into breeding.

    `archive`: list of dict(graph, objectives, F) so far (pass [] for a fresh
    run). `candidates`: this generation's feasible evaluated individuals.
    Returns the updated archive, deduplicated by graph content hash."""
    combined = list(archive) + [dict(graph=c["graph"], objectives=c["objectives"], F=c["F"]) for c in candidates]
    if not combined:
        return []

    seen_hashes = set()
    deduped = []
    for r in combined:
        h = _graph_hash(r["graph"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append(r)

    F = np.array([r["F"] for r in deduped])
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return [deduped[i] for i in front]


def run_generation(population_graphs, ppo_trainer, rng, work_dir=None, sim_seconds=7.0,
                    n_offspring=None, ref_partitions=None, max_workers=None, rl_assisted=True,
                    archive=None):
    """One NSGA-III generation. Returns (next_generation_graphs, log) where
    `log` carries per-survivor objectives/rank for plotting_api.py, plus
    `log["archive"]` - the updated Pareto archive (see
    update_pareto_archive) computed from `archive` (the archive so far;
    pass [] or omit on generation 0) and this generation's evaluated pool.
    Thread `log["archive"]` back in as next call's `archive` to persist it.

    `rl_assisted` switches breeding between the trained RL policy and
    random_baseline.py's uniform-random choice over the same action space -
    see make_children()'s docstring; everything else is unchanged between
    the two arms. Collision/instability outcomes are excluded from the RL
    reward stream entirely (not just penalized) - see make_children_
    collision_free's docstring; a flat penalty was tried and collapsed the
    policy onto whichever action type can never trigger the collision gate.

    `n_offspring` is the number of breeding steps (parent-pair draws), not
    the final offspring count - a SWAP_SUBTREES step produces 2 children, so
    `log["n_offspring"]` can be slightly larger.

    Structured in 3 phases so every individual's MuJoCo evaluation happens in
    one parallel batch: (1) breeding decisions, (2) evaluate_population() on
    parents + all children together, (3) reward assignment + PPO update +
    NSGA-III survival.
    """
    pop_size = len(population_graphs)
    n_offspring = n_offspring or pop_size
    ref_partitions = ref_partitions or _auto_ref_partitions(len(obj_api.OBJECTIVE_NAMES), pop_size)
    work_dir = work_dir or tempfile.mkdtemp(prefix="roblet_gen_")

    # Phase 1: breeding decisions.
    breeding = []  # list of (decision_or_None, children, baseline_parent_indices)
    for _ in range(n_offspring):
        if pop_size > 1:
            pa, pb = rng.sample(population_graphs, 2)
        else:
            pa = pb = population_graphs[0]
        pa_idx = population_graphs.index(pa)
        pb_idx = population_graphs.index(pb)

        children, decision = make_children_collision_free(pa, pb, ppo_trainer, rng, work_dir, rl_assisted=rl_assisted)
        # Which parent each child's improvement is measured against (SWAP_SUBTREES
        # produces one child per parent; everything else produces one child from parent_a).
        baseline_idxs = [pa_idx] if len(children) == 1 else [pa_idx, pb_idx]
        breeding.append((decision, children, baseline_idxs))

    # Phase 2: evaluate parents + every child in one parallel batch.
    all_graphs = list(population_graphs) + [c for _, children, _ in breeding for c in children]
    all_results = evaluate_population(all_graphs, work_dir, sim_seconds=sim_seconds, max_workers=max_workers)

    parent_results = all_results[:pop_size]
    parent_records = [
        dict(graph=g, objectives=o, F=f, constraint=c)
        for g, (o, f, c) in zip(population_graphs, parent_results)
    ]

    # Phase 3: reward assignment + PPO update + NSGA-III survival.
    offspring_records = []
    breeding_events = []  # lineage log for the visualizer - see _write_breeding_events()
    cursor = pop_size
    for decision, children, baseline_idxs in breeding:
        improvements = []
        child_ids = []
        for child, baseline_idx in zip(children, baseline_idxs):
            objectives, f_vec, constraint = all_results[cursor]
            child_ids.append(cursor)
            cursor += 1
            # `constraint` (post-sim physics_ok/is_stable failure) is tracked on
            # offspring_records/log["n_collided"] but never folded into the reward - see docstring.
            improvement = obj_api.scalarize(objectives) - obj_api.scalarize(parent_records[baseline_idx]["objectives"])
            improvements.append(improvement)
            offspring_records.append(dict(graph=child, objectives=objectives, F=f_vec, constraint=constraint))

        if decision is not None:
            if rl_assisted:
                ppo_trainer.record(decision, float(np.mean(improvements)))
            # Logged for BOTH arms regardless (evolution_results_visualizer.py's
            # Mutations/Crossover tabs work identically either way) - only the
            # PPO training call above is RL-specific.
            breeding_events.append(dict(
                type=("mutation" if decision.action in rg.MUTATION_ACTIONS else "crossover"),
                action=decision.action.name,
                parent_ids=sorted(set(baseline_idxs)),
                child_ids=child_ids,
            ))

    if rl_assisted:
        ppo_trainer.update()
    _write_breeding_events(work_dir, pop_size, len(offspring_records), breeding_events)

    all_records = parent_records + offspring_records

    # Feasibility-first selection: ReferenceDirectionSurvival does pure Pareto/niche
    # sorting on F alone, so a failed simulation (all-zero objectives) could otherwise
    # "survive" by not being strictly worse. Feasible individuals (constraint <= 0) are
    # selected first; infeasible ones only pad out the population if there aren't enough.
    # objectives/ind_id are attached directly onto each Individual (not looked up by
    # id(graph) afterward, since a discarded object's id() can be reused).
    def _survive(indices, n_survive):
        if not indices or n_survive <= 0:
            return []
        sub_records = [all_records[i] for i in indices]
        sub_F = np.array([r["F"] for r in sub_records])
        sub_individuals = []
        for j, (i, r) in enumerate(zip(indices, sub_records)):
            ind = Individual(X=r["graph"], F=sub_F[j])
            ind.set("objectives", r["objectives"])
            ind.set("ind_id", i)
            sub_individuals.append(ind)
        sub_pop = Population.create(*sub_individuals)
        ref_dirs = _reference_directions(sub_F.shape[1], n_partitions=ref_partitions)
        survival = ReferenceDirectionSurvival(ref_dirs)
        n_take = min(n_survive, len(sub_pop))
        return list(survival._do(None, sub_pop, n_take))

    feasible_idx = [i for i, r in enumerate(all_records) if r["constraint"] <= 0]
    infeasible_idx = [i for i, r in enumerate(all_records) if r["constraint"] > 0]

    survivors = _survive(feasible_idx, pop_size)
    if len(survivors) < pop_size:
        logger.warning(
            "Only %d/%d feasible individuals available this generation; padding survivors with %d infeasible one(s).",
            len(survivors), pop_size, pop_size - len(survivors),
        )
        survivors += _survive(infeasible_idx, pop_size - len(survivors))

    next_generation = [ind.X for ind in survivors]
    updated_archive = update_pareto_archive(archive or [], [all_records[i] for i in feasible_idx])
    log = dict(
        survivor_objectives=[ind.get("objectives") for ind in survivors],
        survivor_ind_ids=[ind.get("ind_id") for ind in survivors],
        survivor_rank=[ind.get("rank") for ind in survivors],
        n_parents=len(parent_records),
        n_offspring=len(offspring_records),
        n_collided=sum(1 for r in all_records if r["constraint"] > 0),
        archive=updated_archive,
    )
    return next_generation, log
