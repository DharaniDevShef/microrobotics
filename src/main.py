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

import logging
import os
import random

import moo_api
import plotting_api
import rl_api

logger = logging.getLogger(__name__)

POP_SIZE = 3
N_GENERATIONS = 5
SIM_SECONDS = 30
SEED = 0


def configure_logging(log_file, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SRC_DIR, "..", "output", "evolution_run")


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_logging(os.path.join(OUTPUT_DIR, "main.log"))

    rng = random.Random(SEED)
    ppo_trainer = rl_api.PPOTrainer(seed=SEED)

    logger.info("Sobol-seeding initial population (pop_size=%d)...", POP_SIZE)
    population = moo_api.sobol_seed_population(POP_SIZE, seed=SEED)

    for gen in range(N_GENERATIONS):
        logger.info("\n------------------------Generation %d------------------------", gen)
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
        plotting_api.plot_fitness_trends(OUTPUT_DIR)
        plotting_api.plot_convergence(OUTPUT_DIR)
        plotting_api.plot_rl_diagnostics(ppo_trainer.history, OUTPUT_DIR)

        # f1 (flat-state velocity) is ignored for now - see objectives_api.py -
        # so f2 (the single evolved-gait velocity) is the real signal to watch.
        best = max(records, key=lambda r: r["objectives"]["f2_forward_velocity_folded"])
        logger.info(
            "parents=%d offspring=%d collided=%d | survivors=%d | "
            "best f2 (velocity)=%.4f m/s",
            log['n_parents'], log['n_offspring'], log['n_collided'],
            len(population), best['objectives']['f2_forward_velocity_folded'],
        )

    logger.info("Done. Artifacts written to %s", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
