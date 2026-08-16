"""
Entropy API - multi-scale Shannon "shape entropy", per the reference
thesis's 2D/3D shape entropy method (eq. 3.15/3.16): voxelize a set of
module-center positions onto an axis-aligned grid, then for each odd
window size build a canonical local-occupancy pattern around every
occupied cell, count pattern frequencies, and compute Shannon entropy -
normalized by log2(number of distinct patterns) so different window
sizes (and 2D vs 3D) are directly comparable - then averaged across
window sizes into one scalar per shape state.

Pure math, no MuJoCo dependency, so it's testable in isolation.
roblet_simulator.py is what actually calls this: 2D entropy from a
graph's flat/unfolded module positions (qpos0, no simulation needed),
3D entropy from the already-settled folded pose (reusing the same
settle pass done for the screenshot). objectives_api.py's f5 is the
normalized delta between the two - see roblet_simulator.py's run_headless
for where each side gets computed and written into stats.json.
"""

import collections
import itertools
import math

import numpy as np


def voxelize(positions, cell_size):
    """positions: iterable of (x, y[, z]) coordinates (2 or 3 dims - the
    caller decides by what it passes in). Returns the set of unique
    integer grid-cell coordinates - simple axis-aligned rounding, robust
    regardless of the assembly's absolute world orientation, since only
    RELATIVE occupancy structure matters here, not exact lattice
    alignment with any particular axis."""
    cells = set()
    for pos in positions:
        cells.add(tuple(int(round(c / cell_size)) for c in pos))
    return cells


def _window_offsets(dims, half_w):
    """All integer offsets within a (2*half_w+1)^dims hypercube, in a
    fixed canonical order - the neighborhood template every occupied
    cell's local pattern is read against, so two cells with the same
    surroundings always produce the same pattern tuple."""
    return sorted(itertools.product(range(-half_w, half_w + 1), repeat=dims))


def pattern_entropy(occupied_cells, dims, window_size):
    """Shannon entropy (bits) of the local-occupancy-pattern frequency
    distribution at this window size, plus the number of distinct
    patterns observed (used to normalize the entropy elsewhere).
    `window_size` must be odd (an odd cubic/square window, per the
    thesis's formulation, so every occupied cell sits at its center)."""
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
    """The scalar H-bar this module exists for: voxelize `positions`
    (each an (x, y[, z]) coordinate matching `dims`), then average the
    per-window-size NORMALIZED entropy (H / log2(n_distinct_patterns), in
    [0, 1], so different window sizes - and 2D vs 3D - are comparable)
    across `window_sizes`. Returns 0.0 for a degenerate shape (<=1
    occupied cell, or every window seeing only a single pattern - no
    information content either way)."""
    occupied = voxelize(positions, cell_size)
    if len(occupied) <= 1:
        return 0.0
    normalized = []
    for w in window_sizes:
        h, n_patterns = pattern_entropy(occupied, dims, w)
        normalized.append(h / math.log2(n_patterns) if n_patterns > 1 else 0.0)
    return float(np.mean(normalized)) if normalized else 0.0


def adaptive_cell_size(positions, default=0.0085):
    """Median nearest-neighbor distance among `positions` (a dict or any
    iterable of (x, y[, z]) coordinates) - used as the voxel cell size so
    it always matches THIS assembly's actual module spacing (from
    mjcf_generator.py's real build geometry) rather than a hardcoded
    constant that could silently drift out of sync with it if that
    geometry ever changes. Position-only (no graph adjacency needed) so
    roblet_simulator.py can call this directly from MJCF geometry alone -
    it never loads the graph JSON, only the built XML. Falls back to
    `default` (module spacing in meters) with fewer than 2 positions."""
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
