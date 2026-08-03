"""
MOO API - Multi-Objective Optimization Engine (RL-guided NSGA-III).

Owns the population loop: Sobol-seeded initial genotypes, breeding
(RL-guided mutation via rl_api.py, grammar-legal random crossover via
roblet_grammar.py), MuJoCo/objectives evaluation, and NSGA-III
environmental selection (via pymoo's reference-direction survival).

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

import random

import numpy as np
from scipy.stats import qmc
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions

import mujoco_api
import objectives_api as obj_api
import rl_api
import roblet_grammar as rg

COLLISION_PENALTY = -10.0  # subtracted from the RL reward when a child collides


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


def evaluate_individual(G, work_dir=None, sim_seconds=3.0):
    """Runs the MuJoCo rollout + objectives for one genotype.
    Returns (objectives_dict, minimization_vector, constraint_value)."""
    rollout = mujoco_api.evaluate_graph(G, work_dir=work_dir, sim_seconds=sim_seconds)
    objectives = obj_api.compute_objectives(rollout)
    f_vec = obj_api.to_minimization_vector(objectives)
    constraint = obj_api.collision_constraint(rollout)
    return objectives, f_vec, constraint


def make_child(parent_a, parent_b, ppo_trainer, rng, mutation_prob=0.7):
    """One offspring via RL-guided mutation (preferred) or grammar-legal
    random crossover. Returns (child_graph, decision_or_None); decision is
    None when a crossover was performed (crossover isn't RL-driven in
    this pass - see rl_api.py's module docstring)."""
    legal_mutation = any(rg.any_node_allows(parent_a, a) for a in rg.MUTATION_ACTIONS)

    if legal_mutation and (rng.random() < mutation_prob or parent_a is parent_b):
        decision = ppo_trainer.select_mutation(parent_a)
        child = rl_api.apply_decision(parent_a, decision)
        return child, decision

    non_root_a = [n for n in parent_a.nodes if not rg.is_root(parent_a, n)]
    non_root_b = [n for n in parent_b.nodes if not rg.is_root(parent_b, n)]
    if non_root_a and non_root_b:
        node_a, node_b = rng.choice(non_root_a), rng.choice(non_root_b)
        try:
            child, _ = rg.swap_subtrees(parent_a, node_a, parent_b, node_b)
            return child, None
        except ValueError:
            pass

    if legal_mutation:
        decision = ppo_trainer.select_mutation(parent_a)
        return rl_api.apply_decision(parent_a, decision), decision
    return parent_a.copy(), None


def _reference_directions(n_obj, n_partitions=2):
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def run_generation(population_graphs, ppo_trainer, rng, work_dir=None, sim_seconds=3.0,
                    n_offspring=None, ref_partitions=2):
    """One NSGA-III generation. Returns (next_generation_graphs, log) where
    `log` carries per-survivor objectives/rank for plotting_api.py."""
    pop_size = len(population_graphs)
    n_offspring = n_offspring or pop_size

    parent_records = []
    for G in population_graphs:
        objectives, f_vec, constraint = evaluate_individual(G, work_dir=work_dir, sim_seconds=sim_seconds)
        parent_records.append(dict(graph=G, objectives=objectives, F=f_vec, constraint=constraint))

    offspring_records = []
    for _ in range(n_offspring):
        if pop_size > 1:
            pa, pb = rng.sample(population_graphs, 2)
        else:
            pa = pb = population_graphs[0]
        pa_idx = population_graphs.index(pa)

        child, decision = make_child(pa, pb, ppo_trainer, rng)
        objectives, f_vec, constraint = evaluate_individual(child, work_dir=work_dir, sim_seconds=sim_seconds)

        if decision is not None:
            reward = obj_api.scalarize(objectives) - obj_api.scalarize(parent_records[pa_idx]["objectives"])
            if constraint > 0:
                reward += COLLISION_PENALTY
            ppo_trainer.record(decision, reward)

        offspring_records.append(dict(graph=child, objectives=objectives, F=f_vec, constraint=constraint))

    ppo_trainer.update()

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
