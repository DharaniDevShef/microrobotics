"""
Checkpoint - save/load evolution run state (RL weights, population, RNG,
generation) to/from a .pt file. main.py auto-resumes from this at startup.
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
        obj_norm=obj_api.get_normalization_state(),  # objectives_api's running min/max
    )

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    # Temp file + os.replace for an atomic write (no partial file on a crash mid-save).
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

    # weights_only=False: payload has plain Python objects (graphs, RNG state) alongside tensors.
    payload = torch.load(path, weights_only=False)
    try:
        ppo_trainer.load_state_dict(payload["ppo"])
    except (RuntimeError, KeyError):
        # Checkpoint's RL architecture doesn't match the current one (e.g. after an
        # action-space/network change) - no safe partial-load, so start fresh instead.
        logger.warning(
            "Checkpoint at %s has RL weights/optimizer state that don't match the current "
            "network/trainer architecture (likely from before a design-variable/action-space "
            "or RL-architecture change) - starting a fresh run instead of resuming.", path,
        )
        return None
    obj_api.set_normalization_state(payload.get("obj_norm"))  # None for pre-existing checkpoints

    rng = random.Random()
    rng.setstate(payload["rng_state"])
    population = [nx.node_link_graph(g, edges="edges") for g in payload["population"]]

    logger.info("Loaded checkpoint from %s (resume generation=%d, pop_size=%d)",
                path, payload["generation"], len(population))
    return payload["generation"], payload["seed"], rng, population
