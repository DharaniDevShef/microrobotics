"""
Entropy API - multi-scale Shannon "shape entropy" over module-center
positions (voxelize, build local-occupancy patterns per window size,
average normalized entropy across window sizes). Used by
roblet_simulator.py to compute objectives_api's f5 (flat vs. folded delta).
"""

import collections
import itertools
import math

import numpy as np


def voxelize(positions, cell_size):
    """Rounds (x, y[, z]) positions onto an integer grid; returns the set of occupied cells."""
    cells = set()
    for pos in positions:
        cells.add(tuple(int(round(c / cell_size)) for c in pos))
    return cells


def _window_offsets(dims, half_w):
    """All integer offsets within a (2*half_w+1)^dims hypercube, in a fixed canonical order."""
    return sorted(itertools.product(range(-half_w, half_w + 1), repeat=dims))


def pattern_entropy(occupied_cells, dims, window_size):
    """Shannon entropy (bits) of local-occupancy patterns at this window size (must be odd),
    plus the number of distinct patterns observed."""
    half_w = window_size // 2
    offsets = _window_offsets(dims, half_w)
    counts = collections.Counter()
    for cell in occupied_cells:
        pattern = tuple(
            tuple(c + o for c, o in zip(cell, offset)) in occupied_cells
            for offset in offsets
        )
        counts[pattern] += 1

    total = sum(counts.values())
    if total == 0:
        return 0.0, 0
    probs = np.array(list(counts.values()), dtype=float) / total
    entropy = float(-np.sum(probs * np.log2(probs)))
    return entropy, len(counts)


def multiscale_shape_entropy(positions, dims, window_sizes, cell_size):
    """Voxelizes `positions` and averages the normalized entropy (in [0, 1]) across
    `window_sizes` into one scalar. Returns 0.0 for a degenerate (<=1 cell) shape."""
    occupied = voxelize(positions, cell_size)
    if len(occupied) <= 1:
        return 0.0
    normalized = []
    for w in window_sizes:
        h, n_patterns = pattern_entropy(occupied, dims, w)
        normalized.append(h / math.log2(n_patterns) if n_patterns > 1 else 0.0)
    return float(np.mean(normalized)) if normalized else 0.0


def adaptive_cell_size(positions, default=0.0085):
    """Median nearest-neighbor distance among `positions` (dict or iterable of (x,y[,z])),
    used as the voxel cell size. Falls back to `default` (meters) with fewer than 2 positions."""
    pts = np.array([np.asarray(p, dtype=float)
                     for p in (positions.values() if isinstance(positions, dict) else positions)])
    if len(pts) < 2:
        return default
    nn_dists = []
    for i in range(len(pts)):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        dists[i] = np.inf
        nn_dists.append(dists.min())
    return float(np.median(nn_dists))
