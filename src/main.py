"""
Main Orchestrator - drives the Sobol-seeded, RL-guided NSGA-III
evolutionary loop over roblet morphology graphs.

    python main.py

Loop, per generation:
    moo_api.evaluate_population  -> mjcf_generator.build_assembly (XML per individual)
                                  -> sim_executor.run_batch (roblet_simulator.py --headless, N parallel OS processes)
                                  -> objectives_api.compute_objectives (reads each stats.json)
    moo_api.make_children_collision_free -> RL_ASSISTED_GENETIC_OPERATIONS switches this between
                                  rl_api.PPOTrainer.select_action (learned policy, mutation AND
                                  crossover) and random_baseline.act (uniform-random choice over the
                                  identical grammar-legal action space - the classic-GA comparison
                                  arm) - gated either way on mjcf_generator's 3D collision check
    moo_api NSGA-III survival    -> pymoo ReferenceDirectionSurvival
    plotting_api                 -> generation JSON + Pareto plot + RL diagnostics
    checkpoint.save               -> RL weights/optimizers/history + population + RNG state,
                                      atomically written to OUTPUT_DIR/checkpoint.pt

Resuming: this script always checks OUTPUT_DIR/checkpoint.pt at startup
and picks up right after the last generation that finished (no flag
needed) - see checkpoint.py. So a crash, a manual Ctrl-C, or any
unhandled exception loses at most the generation that was in progress;
just run `python main.py` again to continue.

RL_ASSISTED_GENETIC_OPERATIONS = False runs the identical pipeline with
random_baseline.py driving breeding instead of the trained policy,
writing to a SEPARATE output dir (evolution_run_norl/) so it never
collides with (or overwrites the checkpoint of) a True run - see
plotting_api.plot_rl_vs_baseline_comparison for comparing the two.
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

POP_SIZE = 4
N_GENERATIONS = 5
SIM_SECONDS = 7
SEED = 42  # reproducible Sobol-seeding of initial population (fresh runs only - a resumed run's RNG/seed come from the checkpoint)

# True (default): breeding uses rl_api's trained policy, as it always has.
# False: breeding uses random_baseline.py's uniform-random choice over the
# SAME grammar-legal action space instead - a classic-GA "blind variation
# + NSGA-III selection" comparison arm, for measuring what the learned
# policy actually contributes. Writes to a different OUTPUT_DIR (below) so
# toggling this never disturbs an in-progress True run's checkpoint/data.
RL_ASSISTED_GENETIC_OPERATIONS = True


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
# RL_ASSISTED_GENETIC_OPERATIONS=True keeps the EXACT same path this has
# always been ("evolution_run", no suffix) - so flipping the switch back
# to True (the default) never orphans or resets an existing run's
# checkpoint/progress. Only the False/baseline arm gets a new directory,
# since it has no prior data to preserve.
OUTPUT_DIR = os.path.join(
    _SRC_DIR, "..", "output",
    "evolution_run" if RL_ASSISTED_GENETIC_OPERATIONS else "evolution_run_norl",
)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_logging(os.path.join(OUTPUT_DIR, "main.log"))

    ppo_trainer = rl_api.PPOTrainer(seed=SEED)

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
        # Every seed graph is validated collision-free (mjcf_generator.py's
        # 3D check) before it's accepted - see moo_api._build_collision_free_seed.
        # This scratch dir is just where those validation attempts get built
        # and checked, not a real generation's artifacts.
        seed_scratch_dir = os.path.join(OUTPUT_DIR, "_seed_check")
        population = moo_api.sobol_seed_population(POP_SIZE, seed=SEED, scratch_dir=seed_scratch_dir)
        start_gen = 0

    if start_gen >= N_GENERATIONS:
        logger.info("Checkpoint already covers all %d requested generations - nothing to do "
                    "(raise N_GENERATIONS to continue training this run further).", N_GENERATIONS)
        return

    try:
        for gen in range(start_gen, N_GENERATIONS):
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
                rl_assisted=RL_ASSISTED_GENETIC_OPERATIONS,
            )

            records = [
                dict(graph=g, objectives=obj, ind_id=ind_id)
                for g, obj, ind_id in zip(population, log["survivor_objectives"], log["survivor_ind_ids"])
            ]
            plotting_api.append_generation_population(gen, records, OUTPUT_DIR)
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

            # Per-generation n_parents/n_offspring/n_collided, appended
            # generation by generation - not reconstructable from
            # population_history.json alone (that only has survivors),
            # and it's what plot_rl_vs_baseline_comparison's collision-rate
            # panel reads.
            plotting_api.append_generation_stats(gen, log, OUTPUT_DIR)

            # Only reached once generation `gen` has fully finished (evaluation,
            # NSGA-III survival, plotting all succeeded) - so a checkpoint on
            # disk always represents a complete generation, never a half-written
            # one, and resuming just re-runs whatever generation was in flight
            # when something went wrong, rather than needing finer-grained
            # per-individual recovery.
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
