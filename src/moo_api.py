"""
MOO API - Multi-Objective Optimization Engine (RL-guided NSGA-III).

Owns the population loop: Sobol-seeded initial genotypes, breeding (BOTH
mutation and crossover are picked and parameterized by rl_api.py's
actor/critic - this module no longer chooses the operator itself, just
supplies two parents and lets the policy decide), MuJoCo evaluation, and
NSGA-III environmental selection (via pymoo's reference-direction
survival).

Evaluation is batched and parallel: every generation writes an MJCF
assembly for every individual (helper_scripts/mjcf_generator.py) up
front, then runs them all through roblet_simulator.py --headless as
separate OS processes (sim_executor.py, generalizing
parallel_executor.py's subprocess.Popen pattern), and only then reads
back each individual's stats.json - see evaluate_population().

Every genotype here is only HALF the final shape - symmetry.py mirrors
it into the full bilaterally-symmetric morphology right before an
assembly is built (see _prepare_assembly), so evolution itself (mutation,
crossover, the RL policy, NSGA-III) only ever sees/touches the half.

pymoo's `Problem`/`Algorithm` classes assume a fixed-length real/int
decision vector, which doesn't fit a variable-size graph genotype - so
this module drives NSGA-III "by hand": genotypes are plain nx.DiGraph
objects carried as `Individual.X`, and only the resulting 5-objective
matrix F is handed to pymoo's `ReferenceDirectionSurvival`, which only
needs F (and optional constraints) to do non-dominated sorting + niching.
That's the one piece of NSGA-III that's genuinely genotype-agnostic, and
it's exactly the piece the design doc wants (avoids hand-engineering a
scalar fitness formula).
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

# Added to the RL reward when a child collides/fails - see run_generation's
# reward loop and make_children_collision_free's exhausted-attempts path.
# Proportioned to objectives_api.scalarize()'s own [0, 1] range: a real
# reward delta (scalarize(child) - scalarize(parent)) is mathematically
# bounded to [-1, 1], so this used to be -10.0 - 10x any possible
# legitimate outcome. At tiny per-generation sample counts (pop_size
# transitions/update), a handful of early collisions at that magnitude
# could dominate the running advantage estimate and bias PPO away from
# whichever action type triggers them most (crossover, empirically - see
# plot_rl_diagnostics' reward panel). -2.0 is still a clearly worse
# outcome than any legitimate one (max legitimate delta is -1.0), without
# being an order of magnitude larger than the signal it's mixed with.
COLLISION_PENALTY = -2.0

# What roblet_simulator.py's save_simulation_stats(physics_ok=False) writes -
# used verbatim for a graph that couldn't even be built into a valid MJCF
# (ModuleCollisionError), so it never wastes a simulation slot but still
# scores as a failed/infeasible individual like any other collision.
_FAILED_STATS = {"success": 0, "physics_ok": 0, "is_stable": 0, "average_velocity_mmps": 0.0, "total_distance_mm": 0.0}

# graph content-hash -> {"stats": <stats.json dict>, "screenshot_path": <path or None>}.
# run_generation() re-evaluates every carried-over survivor alongside new
# offspring each generation (see evaluate_population()'s docstring), but
# roblet_simulator.py's physics is fully deterministic (fixed B-sweep
# values, no RNG) - an unchanged genotype produces byte-identical results
# every time, so re-running the whole MuJoCo B-sweep for it is pure waste.
# Keyed by content hash (see _graph_hash), not Python id() - a discarded
# graph's id() can be reused by an unrelated later object once garbage
# collected, which would silently return the WRONG graph's cached result.
_EVAL_CACHE = {}


def _graph_hash(G):
    """Deterministic content hash of a genotype (full structure + node/edge
    attributes) - _EVAL_CACHE's key. Two graphs that serialize identically
    WILL simulate identically, so a hit here is always a correct reuse,
    never an approximation."""
    payload = json.dumps(nx.node_link_data(G, edges="edges"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_collision_free(G, scratch_dir, rng):
    """True if the FULL mirrored graph (symmetry.build_symmetric_graph(G))
    builds cleanly through mjcf_generator.build_assembly's own 3D
    collision check (check_collisions=True - the same check
    evaluate_population() relies on): no un-mated module overlaps in
    either the flat or fully-folded pose. Checking the mirrored graph, not
    just the half `G`, matters - the two mirrored halves can collide with
    EACH OTHER even when the half alone is fine on its own. Also rejects
    (False) a half-graph build_symmetric_graph can't even mirror -
    symmetry.MirrorAnchorViolation, which GRAFT_SUBTREE/SWAP_SUBTREES can
    produce (see its docstring) - same "invalid genotype" bucket as a
    geometric collision, not a crash. Used to GATE a graph before it's
    accepted into the population at all - see sobol_seed_population() and
    make_children_collision_free() - rather than just detecting and
    penalizing the collision after the fact during evaluation.

    Repairs `G` in place (rg.ensure_min_light_sensitive) right before
    mirroring, so every graph that passes this gate - and therefore every
    graph that ever reaches _prepare_assembly() later - is guaranteed to
    have at least one light-sensitive joint whenever it has a foldable
    module at all, regardless of whether it arrived here freshly seeded or
    post-mutation."""
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
    """Resamples a fresh random_seed_graph at `n_modules` up to
    `max_attempts` times looking for one that passes _is_collision_free().
    If every attempt at that size collides, halves the module count and
    tries again (down to MIN_MODULES) - sparser graphs are far less
    likely to self-overlap, so this reliably converges to SOME valid
    graph rather than exhausting attempts forever at a size that's simply
    too dense for random_seed_graph's geometry-blind construction."""
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
    """Sobol-sampled initial genotypes (design doc's sampling plan): a
    3D Sobol sequence over (module_count, hinge_angle_bias, fold_type_bias)
    gives a uniform, low-discrepancy spread across the design-variable
    space before RL-guided evolution starts refining it.

    Every returned graph is validated collision-free up front (see
    _build_collision_free_seed) - no individual enters generation 0
    without already having passed mjcf_generator.py's 3D collision test."""
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
    """Mirrors the half-genotype `G` into the full symmetric shape
    (symmetry.py), writes its graph JSON, and calls build_assembly.
    Returns an xml_path, or None if the FULL (mirrored) graph is invalid
    - geometrically (ModuleCollisionError) or structurally
    (symmetry.MirrorAnchorViolation, see its docstring) - callers treat
    that the same as a failed simulation, without wasting a subprocess on
    a model that can't even compile. Checking the mirrored graph (not
    just the half) matters: the two mirrored halves can collide with EACH
    OTHER even when the half alone is perfectly valid on its own.
    Shouldn't actually trigger in practice - every graph reaching here
    already passed _is_collision_free() at creation time - but is kept as
    a defensive backstop, same as the ModuleCollisionError catch below."""
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


def evaluate_population(graphs, work_dir, sim_seconds=7.0, max_workers=None):
    """Writes an MJCF assembly for every graph, runs every valid one
    through roblet_simulator.py --headless in parallel OS processes
    (sim_executor.py), then scores each from its stats.json via
    objectives_api. Returns a list of (objectives, f_vec, constraint)
    aligned to `graphs`' order.

    Individuals whose exact genotype (_graph_hash) was already simulated
    in an earlier generation (typically a carried-over NSGA-III survivor,
    re-appearing in `graphs` unchanged) skip the MuJoCo run entirely -
    see _EVAL_CACHE - since the physics is deterministic and would just
    reproduce the same stats.json. Its stats.json is still (re)written and
    its previous screenshot copied forward into THIS generation's
    work_dir, so every downstream reader (objectives_api,
    evolution_results_visualizer.py) sees the same per-generation file
    layout as before, just without paying for a redundant simulation."""
    os.makedirs(work_dir, exist_ok=True)

    stats_paths = [None] * len(graphs)
    hashes = [None] * len(graphs)
    jobs = []
    cache_hits = 0
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

        cache_hits += 1
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(cached["stats"], f, indent=4)
        src_screenshot = cached.get("screenshot_path")
        if src_screenshot and os.path.exists(src_screenshot):
            try:
                shutil.copyfile(src_screenshot, os.path.join(work_dir, f"screenshot_ind{i}_assembly.png"))
            except OSError:
                logger.warning("Could not copy forward cached screenshot for ind%d", i)

    logger.info(
        "Evaluating population: %d graphs, work_dir=%s (%d cache hits, %d simulated)",
        len(graphs), work_dir, cache_hits, len(jobs),
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
            if h is not None and h not in _EVAL_CACHE:
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
    """One breeding step. When `rl_assisted` (main.py's
    RL_ASSISTED_GENETIC_OPERATIONS), the RL policy itself picks mutation
    vs. crossover (and every parameter of whichever it picks) from
    (parent_a, parent_b) - see rl_api.py's ActorNet. When not, the exact
    same grammar-legal action space is used but every choice is drawn
    uniformly at random instead (random_baseline.py) - the classic-GA
    "blind variation + NSGA-III selection" comparison arm. Returns
    (children, decision_or_None): `children` is a list of 1 graph for a
    mutation/GRAFT_SUBTREE, or 2 graphs for a SWAP_SUBTREES (one
    recombined offspring per parent); `decision` is None only in the rare
    case nothing at all was legal (falls back to a same-graph copy)."""
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
    """Wraps make_children() with a 3D collision gate: if the proposed
    child/children fail mjcf_generator.py's collision check
    (_is_collision_free), the decision is rejected and a fresh one is
    re-sampled. When `rl_assisted`, a rejected attempt also records a flat
    COLLISION_PENALTY reward so the policy gets a learning signal against
    it (the random baseline has no policy to train, so it just re-draws).
    Falls back to a guaranteed-valid no-op copy of parent_a (parent_a is
    already known collision-free, by induction from this same gate) if
    every attempt still collides.

    This is what keeps every individual entering a generation already
    validated - true for BOTH arms of the RL-vs-baseline comparison,
    since collision-gating is a controlled variable, not part of what's
    being compared: parents start collision-free (sobol_seed_population),
    and this function is the only source of new offspring, so the
    invariant holds by construction for every later generation too.

    Also absorbs roblet_grammar.GraftPortConflict, which apply_decision's
    GRAFT_SUBTREE/SWAP_SUBTREES can raise (see its docstring) - there's no
    way to mask that one in advance (it depends on the donor's internal
    structure, only known once both the host and donor node are already
    sampled), so it's treated the same as a rejected/colliding attempt
    here rather than escaping as a crash."""
    for _ in range(max_attempts):
        try:
            children, decision = make_children(parent_a, parent_b, ppo_trainer, rng, rl_assisted=rl_assisted)
        except rg.GraftPortConflict:
            continue
        if all(_is_collision_free(child, scratch_dir, rng) for child in children):
            return children, decision
        if rl_assisted and decision is not None:
            ppo_trainer.record(decision, COLLISION_PENALTY)
    return [parent_a.copy()], None


def _write_breeding_events(work_dir, n_parents, n_offspring, events):
    """Writes `work_dir/breeding_events.json` - the per-generation lineage
    log helper_scripts/evolution_results_visualizer.py reads for its
    Mutations/Crossover sub-tabs. `parent_ids`/`child_ids` in each event
    are indices into this generation's flat evaluated batch (0..n_parents-1
    = carried-over survivors, re-simulated fresh; n_parents..n_parents+
    n_offspring-1 = newly bred offspring this generation), matching the
    `ind{i}_...` filenames evaluate_population() writes - so
    `screenshot_ind{i}_assembly.png` is each index's screenshot."""
    payload = dict(n_parents=n_parents, n_offspring=n_offspring, events=events)
    with open(os.path.join(work_dir, "breeding_events.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _reference_directions(n_obj, n_partitions=2):
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def _auto_ref_partitions(n_obj, pop_size):
    """Largest das-dennis n_partitions whose reference-direction count
    (comb(n_partitions + n_obj - 1, n_obj - 1)) doesn't exceed `pop_size`
    - so NSGA-III's niching resolution scales with population size
    instead of silently staying fixed at whatever partition count
    happened to fit an earlier, smaller POP_SIZE (a caller that never
    overrides run_generation's ref_partitions - true of main.py - would
    otherwise keep e.g. 10 reference directions for 4 objectives even
    after raising pop_size to 30, where each direction ends up niching
    ~3 individuals on average - coarser diversity-preservation than the
    population size could support). Falls back to 1 (comb(n_obj, n_obj-1)
    = n_obj directions - the minimum meaningful partition count) if even
    that already exceeds pop_size."""
    partitions = 1
    while math.comb(partitions + 1 + n_obj - 1, n_obj - 1) <= pop_size:
        partitions += 1
    return partitions


def run_generation(population_graphs, ppo_trainer, rng, work_dir=None, sim_seconds=7.0,
                    n_offspring=None, ref_partitions=None, max_workers=None, rl_assisted=True):
    """One NSGA-III generation. Returns (next_generation_graphs, log) where
    `log` carries per-survivor objectives/rank for plotting_api.py.

    `rl_assisted` (main.py's RL_ASSISTED_GENETIC_OPERATIONS) switches the
    breeding operator between the trained RL policy and random_baseline.py's
    uniform-random choice over the identical grammar-legal action space -
    see make_children()'s docstring. Everything else (collision-gating,
    evaluation, NSGA-III survival) is unchanged between the two, so this
    is the one knob a RL-vs-classic-GA comparison run should toggle.

    `n_offspring` is the number of breeding STEPS (parent-pair draws), not
    the final offspring count: most decisions (mutation, GRAFT_SUBTREE)
    produce 1 child, but a SWAP_SUBTREES decision produces 2 - so
    `log["n_offspring"]` (the actual pool size fed to NSGA-III survival)
    can be slightly larger than the `n_offspring` requested here.

    Structured in 3 phases so every individual's MuJoCo evaluation - both
    parents and every child - happens in ONE parallel batch:
      1. breeding decisions (sequential, cheap - only needs graph
         structure, not this generation's objective values)
      2. evaluate_population() on parents + all children together
      3. reward assignment (needs both parent + child objectives),
         PPO update, and NSGA-III survival
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
        # Which parent each child's improvement is measured against: a
        # mutation/GRAFT_SUBTREE child is a single offspring bred from
        # parent_a, but SWAP_SUBTREES returns one recombined offspring
        # per parent, so its second child is scored against parent_b.
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
            improvement = obj_api.scalarize(objectives) - obj_api.scalarize(parent_records[baseline_idx]["objectives"])
            if constraint > 0:
                improvement += COLLISION_PENALTY
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
    obj_lookup = {id(r["graph"]): r["objectives"] for r in all_records}
    # Position in all_records == position in the ind0..indN batch
    # evaluate_population() just wrote/simulated (parents first, then
    # offspring in breeding order) - kept per-survivor so the visualizer
    # can tell exactly which screenshot/XML/stats.json belongs to each
    # surviving individual, and which survivors are newly-bred offspring
    # vs. carried-over parents (ind_id >= n_parents).
    ind_id_lookup = {id(r["graph"]): i for i, r in enumerate(all_records)}

    # Feasibility-first selection: ReferenceDirectionSurvival._do() does
    # pure Pareto/niche sorting on F alone - it has no idea `constraint`
    # even exists (this module only ever used it for RL reward penalties
    # and the n_collided log, never attached it to the pymoo Individuals).
    # Left unchecked, a failed simulation (all-zero objectives, since
    # f1/f3/f4/f5 are still placeholders) isn't necessarily Pareto-
    # dominated by anything, so it can survive into the next generation
    # purely by not being strictly worse - which is exactly how a failed,
    # screenshot-less individual ends up looking like a "survivor". Feasible
    # individuals (constraint <= 0) are selected first; only if there
    # aren't enough of them to fill the population are infeasible ones
    # used to pad it out, so the population size never shrinks.
    def _survive(indices, n_survive):
        if not indices or n_survive <= 0:
            return []
        sub_records = [all_records[i] for i in indices]
        sub_F = np.array([r["F"] for r in sub_records])
        sub_individuals = [Individual(X=r["graph"], F=sub_F[j]) for j, r in enumerate(sub_records)]
        sub_pop = Population.create(*sub_individuals)
        ref_dirs = _reference_directions(sub_F.shape[1], n_partitions=ref_partitions)
        survival = ReferenceDirectionSurvival(ref_dirs)
        n_take = min(n_survive, len(sub_pop))
        return list(survival._do(None, sub_pop, n_take))

    feasible_idx = [i for i, r in enumerate(all_records) if r["constraint"] <= 0]
    infeasible_idx = [i for i, r in enumerate(all_records) if r["constraint"] > 0]

    # Elitism: force the single best-scalarized FEASIBLE individual through
    # regardless of niching. ReferenceDirectionSurvival._do() only does
    # crowding/niche selection against reference directions - it has no
    # notion of "best" - so once the population homogenizes (e.g. the RL
    # policy collapsing onto one low-risk action type - see
    # plot_action_distribution), niching can legitimately drop the best
    # individual found so far in favor of reference-direction coverage,
    # producing exactly the "best fitness decreases" artifact convergence.png
    # shows. Restricted to feasible_idx, not all_records, so a failed
    # simulation's placeholder objectives (see the feasibility-first
    # comment above) can never be elitism-locked in as "best".
    elite_idx = max(feasible_idx, key=lambda i: obj_api.scalarize(all_records[i]["objectives"])) if feasible_idx else None
    elite_survivor = None
    if elite_idx is not None:
        elite_record = all_records[elite_idx]
        elite_survivor = Individual(X=elite_record["graph"], F=np.array(elite_record["F"]))
        feasible_idx = [i for i in feasible_idx if i != elite_idx]

    survivors = _survive(feasible_idx, pop_size - (1 if elite_survivor is not None else 0))
    if elite_survivor is not None:
        survivors = [elite_survivor] + survivors
    if len(survivors) < pop_size:
        logger.warning(
            "Only %d/%d feasible individuals available this generation; padding survivors with %d infeasible one(s).",
            len(survivors), pop_size, pop_size - len(survivors),
        )
        survivors += _survive(infeasible_idx, pop_size - len(survivors))

    next_generation = [ind.X for ind in survivors]
    log = dict(
        survivor_objectives=[obj_lookup[id(ind.X)] for ind in survivors],
        survivor_ind_ids=[ind_id_lookup[id(ind.X)] for ind in survivors],
        survivor_rank=[ind.get("rank") for ind in survivors],
        n_parents=len(parent_records),
        n_offspring=len(offspring_records),
        n_collided=sum(1 for r in all_records if r["constraint"] > 0),
    )
    return next_generation, log
