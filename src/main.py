"""
Main Orchestrator - drives the Sobol-seeded, RL-guided NSGA-III
evolutionary loop over roblet morphology graphs.

    python main.py

Loop, per generation:
    moo_api.evaluate_population  -> mjcf_generator.build_assembly (XML per individual)
                                  -> sim_executor.run_batch (roblet_simulator.py --headless, N parallel OS processes)
                                  -> objectives_api.compute_objectives (reads each stats.json)
    moo_api.make_children_collision_free -> rl_api.PPOTrainer.select_action (mutation AND crossover)
                                  -> roblet_grammar, gated on mjcf_generator's 3D collision check
    moo_api NSGA-III survival    -> pymoo ReferenceDirectionSurvival
    plotting_api                 -> generation JSON + Pareto plot + RL diagnostics
"""

import logging
import os
import random
import time

import moo_api
import objectives_api
import plotting_api
import rl_api

logger = logging.getLogger(__name__)

POP_SIZE = 4
N_GENERATIONS = 5
SIM_SECONDS = 7
SEED = 42  # reproducible Sobol-seeding of initial population


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
    # Every seed graph is validated collision-free (mjcf_generator.py's 3D
    # check) before it's accepted - see moo_api._build_collision_free_seed.
    # This scratch dir is just where those validation attempts get built
    # and checked, not a real generation's artifacts.
    seed_scratch_dir = os.path.join(OUTPUT_DIR, "_seed_check")
    population = moo_api.sobol_seed_population(POP_SIZE, seed=SEED, scratch_dir=seed_scratch_dir)

    for gen in range(N_GENERATIONS):
        gen_start_time = time.time()
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
            dict(graph=g, objectives=obj, ind_id=ind_id)
            for g, obj, ind_id in zip(population, log["survivor_objectives"], log["survivor_ind_ids"])
        ]
        plotting_api.save_generation_population(gen, records, OUTPUT_DIR)
        plotting_api.plot_pareto_front(gen, records, OUTPUT_DIR)
        plotting_api.plot_fitness_trends(OUTPUT_DIR)
        plotting_api.plot_convergence(OUTPUT_DIR)
        plotting_api.plot_rl_diagnostics(ppo_trainer.history, OUTPUT_DIR)

        # objectives_api.scalarize() - the SAME function moo_api.py uses
        # for the RL reward, and evolution_results_visualizer.py's
        # _aggregate_fitness() now calls too - so "Individual index" below
        # always names the exact ind_id whose XML/screenshot is the UI's
        # #1 Population card, not just whichever happens to have the best
        # f2 (velocity) alone. Sign-corrects "minimize" objectives (per
        # MAXIMIZE) before summing, unlike a raw sum, so this stays correct
        # once f3/f4/f5 stop being placeholder zeros.
        best = max(records, key=lambda r: objectives_api.scalarize(r["objectives"]))
        logger.info(
            "parents=%d offspring=%d collided=%d | survivors=%d | "
            "best f2 (velocity)=%.4f m/s | Individual index: %d",
            log['n_parents'], log['n_offspring'], log['n_collided'],
            len(population), best['objectives']['f2_forward_velocity_folded'], best['ind_id'],
        )
        logger.info("Generation %d finished in %.2f s", gen, time.time() - gen_start_time)

    logger.info("Done. Artifacts written to %s", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
