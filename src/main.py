"""
Main Orchestrator - drives the Sobol-seeded, RL-guided NSGA-III evolutionary
loop over roblet morphology graphs. Run with `python main.py`; it
auto-resumes from OUTPUT_DIR/checkpoint.pt if one exists.
"""

import logging
import os
import random
import time

import checkpoint as ckpt
import moo_api
import objectives_api
import plotting_api
import rl_api

logger = logging.getLogger(__name__)

POP_SIZE = 30
N_GENERATIONS = 40
SIM_SECONDS = 7
SEED = 42  # reproducible Sobol-seeding of initial population (fresh runs only)

# Plots are expensive to fully re-render every generation, so they're only
# regenerated every N generations (always still on the last one).
PLOT_EVERY_N_GENERATIONS = 5

# True: breeding uses rl_api's trained policy. False: breeding uses
# random_baseline.py's uniform-random choice instead (classic-GA comparison
# arm). Each writes to its own OUTPUT_DIR so toggling never disturbs the other.
RL_ASSISTED_GENETIC_OPERATIONS = True

# Which pheromone-response evolution run this is: "attractive" optimizes
# turning toward/speeding up under a light stimulus; "repulsive" optimizes
# turning away/slowing down. Two separate runs, each with its own OUTPUT_DIR.
PHEROMONE_RESPONSE_TYPE = "attractive"  # "attractive" | "repulsive"


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
# Default settings (True/"attractive") use the unsuffixed "evolution_run" dir
# so an existing checkpoint is never orphaned; other settings get their own dir.
_pheromone_suffix = "" if PHEROMONE_RESPONSE_TYPE == "attractive" else f"_{PHEROMONE_RESPONSE_TYPE}"
OUTPUT_DIR = os.path.join(
    _SRC_DIR, "..", "output",
    ("evolution_run" if RL_ASSISTED_GENETIC_OPERATIONS else "evolution_run_norl") + _pheromone_suffix,
)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_logging(os.path.join(OUTPUT_DIR, "main.log"))
    objectives_api.configure_pheromone_response(PHEROMONE_RESPONSE_TYPE)
    logger.info("Pheromone response mode: %s", PHEROMONE_RESPONSE_TYPE)

    ppo_trainer = rl_api.PPOTrainer(seed=SEED)
    archive = plotting_api.load_pareto_archive(OUTPUT_DIR)  # Pareto archive, kept as its own JSON side-file

    resumed = ckpt.load(CHECKPOINT_PATH, ppo_trainer)
    if resumed is not None:
        start_gen, resumed_seed, rng, population = resumed
        if resumed_seed != SEED:
            logger.warning(
                "Checkpoint was saved with SEED=%d, but this run is configured with SEED=%d - "
                "resuming with the checkpoint's RNG state anyway (SEED here only matters for fresh runs).",
                resumed_seed, SEED,
            )
        logger.info("Resuming from checkpoint at generation %d (pop_size=%d)", start_gen, len(population))
    else:
        random.seed(SEED)
        rng = random.Random(SEED)
        logger.info("Sobol-seeding initial population (pop_size=%d)...", POP_SIZE)
        seed_scratch_dir = os.path.join(OUTPUT_DIR, "_seed_check")
        population = moo_api.sobol_seed_population(POP_SIZE, seed=SEED, scratch_dir=seed_scratch_dir)
        start_gen = 0

        # Warm-start objectives_api's reward-normalization range from the seed
        # population before any breeding, so PPO's first rewards aren't
        # normalized against an empty range (cheap - moo_api caches these evals).
        logger.info("Warm-starting objective normalization range from the Sobol-seeded population...")
        moo_api.evaluate_population(population, os.path.join(OUTPUT_DIR, "_warm_start"), sim_seconds=SIM_SECONDS)

    if start_gen >= N_GENERATIONS:
        logger.info("Checkpoint already covers all %d requested generations - nothing to do "
                    "(raise N_GENERATIONS to continue training this run further).", N_GENERATIONS)
        return

    try:
        for gen in range(start_gen, N_GENERATIONS):
            gen_start_time = time.time()
            logger.info("\n------------------------Generation %d------------------------", gen)
            gen_dir = os.path.join(OUTPUT_DIR, f"generation_{gen}")  # own folder per generation
            os.makedirs(gen_dir, exist_ok=True)
            population, log = moo_api.run_generation(
                population, ppo_trainer, rng, work_dir=gen_dir, sim_seconds=SIM_SECONDS,
                rl_assisted=RL_ASSISTED_GENETIC_OPERATIONS, archive=archive,
            )
            archive = log["archive"]
            plotting_api.save_pareto_archive(archive, OUTPUT_DIR)

            records = [
                dict(graph=g, objectives=obj, ind_id=ind_id)
                for g, obj, ind_id in zip(population, log["survivor_objectives"], log["survivor_ind_ids"])
            ]
            plotting_api.append_generation_population(gen, records, OUTPUT_DIR)

            if gen % PLOT_EVERY_N_GENERATIONS == 0 or gen == N_GENERATIONS - 1:
                plotting_api.plot_pareto_front_last_gen(OUTPUT_DIR, filename=f"pareto_front{_pheromone_suffix}.png")
                plotting_api.plot_pareto_parallel_coordinates(
                    OUTPUT_DIR, filename=f"pareto_parallel_coordinates{_pheromone_suffix}.png")
                plotting_api.plot_fitness_trends(OUTPUT_DIR, filename=f"fitness_trends{_pheromone_suffix}.png")
                plotting_api.plot_convergence(
                    OUTPUT_DIR, is_rl=RL_ASSISTED_GENETIC_OPERATIONS, filename=f"convergence{_pheromone_suffix}.png")
                plotting_api.plot_hypervolume(OUTPUT_DIR, filename=f"hypervolume{_pheromone_suffix}.png")
                plotting_api.plot_entropy_vs_velocity(
                    OUTPUT_DIR, filename=f"entropy_vs_velocity{_pheromone_suffix}.png")
                plotting_api.plot_rl_diagnostics(ppo_trainer.history, OUTPUT_DIR, suffix=_pheromone_suffix)
                plotting_api.plot_action_distribution(OUTPUT_DIR, filename=f"action_distribution{_pheromone_suffix}.png")
                plotting_api.plot_reward_and_loss_by_action(
                    ppo_trainer.history, OUTPUT_DIR, filename=f"rl_diagnostics_by_action{_pheromone_suffix}.png")

            best = max(records, key=lambda r: objectives_api.scalarize(r["objectives"]))
            logger.info(
                "parents=%d offspring=%d collided=%d | survivors=%d | pareto_archive=%d | "
                "best f1 (velocity)=%.4f m/s | Individual index: %d",
                log['n_parents'], log['n_offspring'], log['n_collided'], len(population), len(archive),
                best['objectives']['f1_folded_gait_velocity'], best['ind_id'],
            )

            plotting_api.append_generation_stats(gen, log, OUTPUT_DIR)

            # Only reached once generation `gen` is fully finished, so a saved
            # checkpoint always represents a complete generation.
            ckpt.save(CHECKPOINT_PATH, gen + 1, SEED, rng, population, ppo_trainer)
            logger.info("Generation %d finished in %.2f s", gen, time.time() - gen_start_time)
    except Exception:
        logger.exception(
            "Generation loop stopped due to an unhandled error. Progress through the last "
            "completed generation is safely checkpointed at %s - just re-run main.py to resume.",
            CHECKPOINT_PATH,
        )
        raise

    logger.info("Done. Artifacts written to %s", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
