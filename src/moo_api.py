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

import json
import logging
import os
import random
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
import rl_api
import roblet_grammar as rg
import sim_executor

logger = logging.getLogger(__name__)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER_SCRIPTS_DIR = os.path.join(_SRC_DIR, "..", "helper_scripts")
if _HELPER_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _HELPER_SCRIPTS_DIR)

from mjcf_generator import build_assembly, ModuleCollisionError  # noqa: E402

_MESHDIR = os.path.abspath(os.path.join(_SRC_DIR, "..", "meshes")).replace("\\", "/")

COLLISION_PENALTY = -10.0  # subtracted from the RL reward when a child collides

# What roblet_simulator.py's save_simulation_stats(physics_ok=False) writes -
# used verbatim for a graph that couldn't even be built into a valid MJCF
# (ModuleCollisionError), so it never wastes a simulation slot but still
# scores as a failed/infeasible individual like any other collision.
_FAILED_STATS = {"success": 0, "physics_ok": 0, "is_stable": 0, "average_velocity_mmps": 0.0, "total_distance_mm": 0.0}


def sobol_seed_population(pop_size, seed=0):
    """Sobol-sampled initial genotypes (design doc's sampling plan): a
    3D Sobol sequence over (module_count, hinge_angle_bias, fold_type_bias)
    gives a uniform, low-discrepancy spread across the design-variable
    space before RL-guided evolution starts refining it."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    n = 1 << max(1, (pop_size - 1).bit_length())  # Sobol is balanced at powers of two
    draws = sampler.random(n)[:pop_size]
    rng = random.Random(seed)

    population = []
    for module_count_u, hinge_u, fold_u in draws:
        n_modules = int(round(rg.MIN_MODULES + module_count_u * (rg.MAX_MODULES - rg.MIN_MODULES)))
        base_angle = rg.MIN_HINGE_ANGLE + hinge_u * (rg.MAX_HINGE_ANGLE - rg.MIN_HINGE_ANGLE)

        def hinge_angle_fn(base_angle=base_angle):
            return float(np.clip(rng.gauss(base_angle, 15.0), rg.MIN_HINGE_ANGLE, rg.MAX_HINGE_ANGLE))

        weights = [1.0, 1.0 + 2 * fold_u, 1.0 + 2 * (1 - fold_u)]  # biases Mountain vs. valley fold
        type_choices = [rng.choices(rg.MODULE_TYPES, weights=weights, k=1)[0] for _ in range(n_modules)]

        G = rg.random_seed_graph(rng, n_modules, module_type_choices=type_choices,
                                  hinge_angle_fn=hinge_angle_fn)
        population.append(G)
    return population


def _prepare_assembly(G, work_dir, tag):
    """Writes graph JSON + calls build_assembly. Returns an xml_path, or
    None if the graph is geometrically invalid (ModuleCollisionError) -
    callers treat that the same as a failed simulation, without wasting a
    subprocess on a model that can't even compile."""
    graph_json_path = os.path.join(work_dir, f"{tag}_graph.json")
    xml_path = os.path.join(work_dir, f"{tag}_assembly.xml")
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f)
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
    aligned to `graphs`' order."""
    os.makedirs(work_dir, exist_ok=True)
    logger.info("Evaluating population: %d graphs, work_dir=%s", len(graphs), work_dir)

    stats_paths = [None] * len(graphs)
    jobs = []
    for i, G in enumerate(graphs):
        xml_path = _prepare_assembly(G, work_dir, tag=f"ind{i}")
        if xml_path is None:
            continue
        stats_paths[i] = os.path.join(work_dir, f"ind{i}_stats.json")
        jobs.append((xml_path, stats_paths[i]))

    sim_executor.run_batch(jobs, max_workers=max_workers, max_sim_time=sim_seconds)
    logger.info("Finished simulation batch: %d jobs", len(jobs))

    results = []
    for stats_path in stats_paths:
        if stats_path is None:
            stats = dict(_FAILED_STATS)
        else:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
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


def make_children(parent_a, parent_b, ppo_trainer, rng):
    """One breeding step: the RL policy itself picks mutation vs.
    crossover (and every parameter of whichever it picks) from
    (parent_a, parent_b) - see rl_api.py's ActorNet. Returns
    (children, decision_or_None): `children` is a list of 1 graph for a
    mutation/GRAFT_SUBTREE, or 2 graphs for a SWAP_SUBTREES (one
    recombined offspring per parent); `decision` is None only in the rare
    case nothing at all was legal (falls back to a same-graph copy)."""
    if not rl_api.has_any_legal_action(parent_a, parent_b):
        return [parent_a.copy()], None

    decision = ppo_trainer.select_action(parent_a, parent_b)
    children = rl_api.apply_decision(parent_a, parent_b, decision)
    return children, decision


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


def run_generation(population_graphs, ppo_trainer, rng, work_dir=None, sim_seconds=7.0,
                    n_offspring=None, ref_partitions=2, max_workers=None):
    """One NSGA-III generation. Returns (next_generation_graphs, log) where
    `log` carries per-survivor objectives/rank for plotting_api.py.

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

        children, decision = make_children(pa, pb, ppo_trainer, rng)
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
            ppo_trainer.record(decision, float(np.mean(improvements)))
            breeding_events.append(dict(
                type=("mutation" if decision.action in rg.MUTATION_ACTIONS else "crossover"),
                action=decision.action.name,
                parent_ids=sorted(set(baseline_idxs)),
                child_ids=child_ids,
            ))

    ppo_trainer.update()
    _write_breeding_events(work_dir, pop_size, len(offspring_records), breeding_events)

    all_records = parent_records + offspring_records
    F = np.array([r["F"] for r in all_records])
    obj_lookup = {id(r["graph"]): r["objectives"] for r in all_records}

    individuals = [Individual(X=r["graph"], F=F[i]) for i, r in enumerate(all_records)]
    pop = Population.create(*individuals)

    ref_dirs = _reference_directions(F.shape[1], n_partitions=ref_partitions)
    survival = ReferenceDirectionSurvival(ref_dirs)
    n_survive = min(pop_size, len(pop))
    survivors = survival._do(None, pop, n_survive)

    next_generation = [ind.X for ind in survivors]
    log = dict(
        survivor_objectives=[obj_lookup[id(ind.X)] for ind in survivors],
        survivor_rank=survivors.get("rank"),
        n_parents=len(parent_records),
        n_offspring=len(offspring_records),
        n_collided=sum(1 for r in all_records if r["constraint"] > 0),
    )
    return next_generation, log
