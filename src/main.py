"""
Main Orchestrator - drives the Sobol-seeded, RL-guided NSGA-III
evolutionary loop over roblet morphology graphs.

    python main.py

Loop, per generation:
    moo_api.evaluate_individual  -> mujoco_api.evaluate_graph -> objectives_api.compute_objectives
    moo_api.make_child           -> rl_api (mutation) / roblet_grammar (crossover)
    moo_api NSGA-III survival    -> pymoo ReferenceDirectionSurvival
    plotting_api                 -> generation JSON + Pareto plot + RL diagnostics
"""

import os
import random

import moo_api
import plotting_api
import rl_api

POP_SIZE = 3
N_GENERATIONS = 5
SIM_SECONDS = 30
SEED = 0

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SRC_DIR, "..", "output", "evolution_run")
SCRATCH_DIR = os.path.join(OUTPUT_DIR, "_scratch")


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    rng = random.Random(SEED)
    ppo_trainer = rl_api.PPOTrainer(seed=SEED)

    print(f"Sobol-seeding initial population (pop_size={POP_SIZE})...")
    population = moo_api.sobol_seed_population(POP_SIZE, seed=SEED)

    for gen in range(N_GENERATIONS):
        print(f"=== Generation {gen} ===")
        population, log = moo_api.run_generation(
            population, ppo_trainer, rng, work_dir=SCRATCH_DIR, sim_seconds=SIM_SECONDS,
        )

        records = [
            dict(graph=g, objectives=obj)
            for g, obj in zip(population, log["survivor_objectives"])
        ]
        plotting_api.save_generation_population(gen, records, OUTPUT_DIR)
        plotting_api.plot_pareto_front(gen, records, OUTPUT_DIR)
        plotting_api.plot_rl_diagnostics(ppo_trainer.history, OUTPUT_DIR)

        best = max(records, key=lambda r: r["objectives"]["f1_forward_velocity_flat"])
        print(
            f"  parents={log['n_parents']} offspring={log['n_offspring']} "
            f"collided={log['n_collided']} | survivors={len(population)} | "
            f"best f1 (flat vel.)={best['objectives']['f1_forward_velocity_flat']:.4f} m/s"
        )

    print(f"Done. Artifacts written to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
