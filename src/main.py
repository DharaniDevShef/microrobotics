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

POP_SIZE = 30
N_GENERATIONS = 10
SIM_SECONDS = 7
SEED = 42  # reproducible Sobol-seeding of initial population (fresh runs only - a resumed run's RNG/seed come from the checkpoint)

# plotting_api's ~13 PNGs are all DPI=600 (print quality) and get fully
# re-rendered from a growing population_history.json every time they're
# called - measured at ~25-40s+ per generation combined, and growing as
# the run gets longer, even though nobody actually needs all of them
# rewritten every single generation (they're progress-monitoring, not
# live telemetry). Regenerated every PLOT_EVERY_N_GENERATIONS generations
# instead - always still on the very last one (see the main loop), so the
# final saved plots are never stale. Data persistence (population_history.
# json, generation_stats.json, checkpoint.pt) is NOT throttled by this -
# only the matplotlib rendering is.
PLOT_EVERY_N_GENERATIONS = 10

# True (default): breeding uses rl_api's trained policy, as it always has.
# False: breeding uses random_baseline.py's uniform-random choice over the
# SAME grammar-legal action space instead - a classic-GA "blind variation
# + NSGA-III selection" comparison arm, for measuring what the learned
# policy actually contributes. Writes to a different OUTPUT_DIR (below) so
# toggling this never disturbs an in-progress True run's checkpoint/data.
RL_ASSISTED_GENETIC_OPERATIONS = True

# Which of the two pheromone-response evolution runs this is (Week 2's
# Reaction Primitives 2.1-3.2 - see objectives_api.configure_pheromone_
# response for the full reasoning): "attractive" optimizes turning TOWARD
# a one-sided light stimulus and speeding up under a full-width one (RPs
# 2.1/3.2); "repulsive" optimizes turning away and slowing down (RPs
# 2.2/3.1). These are two SEPARATE evolution runs, not two objectives
# added to one run - sharing hinge_angle_on_light_detection as one scalar
# design-variable lever in opposite directions within a single run would
# be a self-contradictory objective pair. Each writes to its own
# OUTPUT_DIR (below) so switching this never disturbs the other's
# checkpoint/data, same principle as RL_ASSISTED_GENETIC_OPERATIONS above.
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
# RL_ASSISTED_GENETIC_OPERATIONS=True keeps the EXACT same path this has
# always been ("evolution_run", no suffix) - so flipping the switch back
# to True (the default) never orphans or resets an existing run's
# checkpoint/progress. Only the False/baseline arm gets a new directory,
# since it has no prior data to preserve. Same principle for
# PHEROMONE_RESPONSE_TYPE: "attractive" (the default) adds no suffix, so
# an existing pre-pheromone checkpoint's directory name is unchanged;
# "repulsive" gets its own "_repulsive" suffixed directory instead of
# reusing "attractive"'s.
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

        # Warm-start objectives_api.scalarize()'s running min/max (the RL
        # reward's normalization reference - see its docstring) from the
        # Sobol-seeded population's own spread, BEFORE any breeding or PPO
        # update happens. Without this, the running range starts completely
        # empty and only widens as generation 0's individuals happen to get
        # evaluated one at a time - so the very first rewards PPO ever sees
        # (which is also when a failure mode like a low-probability action
        # type getting an early bad outcome and never recovering is most
        # likely to take hold - see rl_api.py's module docstring) are
        # normalized against the least reliable range the run will ever
        # have. Evaluates the SAME graphs generation 0 is about to breed
        # from anyway, so moo_api._EVAL_CACHE makes this free: generation
        # 0's own evaluate_population() call re-hashes these exact
        # (unchanged) genotypes and skips re-simulating them.
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

            # See PLOT_EVERY_N_GENERATIONS' docstring - rendering, not data
            # persistence, so safe to skip most generations. Always runs on
            # the last generation so the final saved plots are current.
            if gen % PLOT_EVERY_N_GENERATIONS == 0 or gen == N_GENERATIONS - 1:
                plotting_api.plot_pareto_front_last_gen(OUTPUT_DIR)
                plotting_api.plot_pareto_parallel_coordinates(OUTPUT_DIR)
                plotting_api.plot_fitness_trends(OUTPUT_DIR)
                plotting_api.plot_convergence(OUTPUT_DIR)
                plotting_api.plot_hypervolume(OUTPUT_DIR)
                plotting_api.plot_entropy_vs_velocity(OUTPUT_DIR)
                plotting_api.plot_rl_diagnostics(ppo_trainer.history, OUTPUT_DIR)
                plotting_api.plot_action_distribution(OUTPUT_DIR)
                plotting_api.plot_reward_and_loss_by_action(ppo_trainer.history, OUTPUT_DIR)

            # objectives_api.scalarize() - the SAME function moo_api.py uses
            # for the RL reward, and evolution_results_visualizer.py's
            # _aggregate_fitness() now calls too - so "Individual index" below
            # always names the exact ind_id whose XML/screenshot is the UI's
            # #1 Population card, not just whichever happens to have the best
            # f1 (velocity) alone. Sign-corrects "minimize" objectives (per
            # MAXIMIZE) before summing, unlike a raw sum.
            best = max(records, key=lambda r: objectives_api.scalarize(r["objectives"]))
            logger.info(
                "parents=%d offspring=%d collided=%d | survivors=%d | "
                "best f1 (velocity)=%.4f m/s | Individual index: %d",
                log['n_parents'], log['n_offspring'], log['n_collided'],
                len(population), best['objectives']['f1_folded_gait_velocity'], best['ind_id'],
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
