"""
Checkpoint - save/resume the whole evolution run: RL weights + optimizer
state + training history, the current population, the breeding RNG state,
and which generation to resume at. main.py checks for a checkpoint at
startup and auto-resumes from it if present - no flag needed - so a crash,
manual stop, or unhandled exception loses at most the generation that was
in progress, never everything before it.

Written via temp-file-then-atomic-rename (os.replace): a checkpoint file
on disk is always either the previous complete one or the new complete
one, never a partial write from a crash mid-save.
"""

import logging
import os
import random
import tempfile

import networkx as nx
import torch

import objectives_api as obj_api

logger = logging.getLogger(__name__)


def save(path, generation, seed, rng, population, ppo_trainer):
    """Saves state to resume AT `generation` (i.e. call this with
    `gen + 1` right after generation `gen` finishes)."""
    payload = dict(
        generation=generation,
        seed=seed,
        rng_state=rng.getstate(),
        population=[nx.node_link_data(g, edges="edges") for g in population],
        ppo=ppo_trainer.state_dict(),
        # scalarize()'s per-objective running min/max - without this, a
        # resumed run would start normalizing from an empty range again
        # (briefly treating every objective as "no variation yet").
        obj_norm=obj_api.get_normalization_state(),
    )

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    # Temp file in the SAME directory so os.replace is an atomic rename
    # (not a cross-filesystem copy+delete, which isn't atomic and could
    # leave a half-written file on a crash between the two steps).
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".checkpoint_", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Saved checkpoint at %s (resume generation=%d, pop_size=%d)",
                path, generation, len(population))


def load(path, ppo_trainer):
    """Restores `ppo_trainer`'s weights/optimizers/history/rng in place.
    Returns (next_generation, seed, rng, population), or None if no
    checkpoint exists at `path`."""
    if not os.path.exists(path):
        return None

    # weights_only=False: this payload carries plain Python objects
    # (graph JSON dicts, RNG state tuples) alongside tensors, not just
    # tensors - safe here since it's a checkpoint this same codebase
    # wrote, not an untrusted third-party file.
    payload = torch.load(path, weights_only=False)
    try:
        ppo_trainer.load_state_dict(payload["ppo"])
    except RuntimeError:
        # nn.Module.load_state_dict is shape-strict: a checkpoint saved
        # before the RL action space or node feature vector grew (e.g.
        # adding the light-sensitive-joint design variables' new
        # TOGGLE_LIGHT_SENSOR/MUTATE_LIGHT_HINGE_ANGLE actions and their
        # two new per-node features - see rl_api.py) has ActorNet/CriticNet
        # tensors of the WRONG size for the current architecture. There's
        # no safe partial-load here (the mismatched layers are exactly the
        # ones every other layer's weights were jointly trained against),
        # so treat this the same as "no checkpoint" - a full fresh
        # Sobol-reseeded start - rather than crashing main.py outright.
        logger.warning(
            "Checkpoint at %s has RL weights that don't match the current network "
            "architecture (likely from before a design-variable/action-space change) - "
            "starting a fresh run instead of resuming.", path,
        )
        return None
    obj_api.set_normalization_state(payload.get("obj_norm"))  # None for pre-existing checkpoints

    rng = random.Random()
    rng.setstate(payload["rng_state"])
    population = [nx.node_link_graph(g, edges="edges") for g in payload["population"]]

    logger.info("Loaded checkpoint from %s (resume generation=%d, pop_size=%d)",
                path, payload["generation"], len(population))
    return payload["generation"], payload["seed"], rng, population
