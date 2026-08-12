"""
Main Orchestrator - drives the Sobol-seeded, RL-guided NSGA-III
evolutionary loop over roblet morphology graphs.

    python main.py

Loop, per generation:
    moo_api.evaluate_population  -> mjcf_generator.build_assembly (XML per individual)
                                  -> sim_executor.run_batch (roblet_simulator.py --headless, N parallel OS processes)
                                  -> objectives_api.compute_objectives (reads each stats.json)
    moo_api.make_children        -> rl_api.PPOTrainer.select_action (mutation AND crossover) -> roblet_grammar
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


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = random.Random(SEED)
    ppo_trainer = rl_api.PPOTrainer(seed=SEED)

    print(f"Sobol-seeding initial population (pop_size={POP_SIZE})...")
    population = moo_api.sobol_seed_population(POP_SIZE, seed=SEED)

    for gen in range(N_GENERATIONS):
        print(f"=== Generation {gen} ===")
        # Each generation gets its own folder (XMLs, stats.json, screenshots,
        # breeding_events.json) instead of a shared _scratch dir that the
        # next generation's same-named files would just overwrite - see
        # helper_scripts/evolution_results_visualizer.py, which reads these
        # per-generation folders for its Screenshots/Mutations/Crossover tabs.
        gen_dir = os.path.join(OUTPUT_DIR, f"generation_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        population, log = moo_api.run_generation(
            population, ppo_trainer, rng, work_dir=gen_dir, sim_seconds=SIM_SECONDS,
        )

        records = [
            dict(graph=g, objectives=obj)
            for g, obj in zip(population, log["survivor_objectives"])
        ]
        plotting_api.save_generation_population(gen, records, OUTPUT_DIR)
        plotting_api.plot_pareto_front(gen, records, OUTPUT_DIR)
        plotting_api.plot_rl_diagnostics(ppo_trainer.history, OUTPUT_DIR)

        # f1 (flat-state velocity) is ignored for now - see objectives_api.py -
        # so f2 (the single evolved-gait velocity) is the real signal to watch.
        best = max(records, key=lambda r: r["objectives"]["f2_forward_velocity_folded"])
        print(
            f"  parents={log['n_parents']} offspring={log['n_offspring']} "
            f"collided={log['n_collided']} | survivors={len(population)} | "
            f"best f2 (velocity)={best['objectives']['f2_forward_velocity_folded']:.4f} m/s"
        )

    print(f"Done. Artifacts written to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
