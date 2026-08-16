"""
Objectives API - Fitness Evaluator.

Turns a roblet_simulator.py `stats.json` dict (see its save_simulation_stats)
into the design doc's f1..f5 objective vector. This reads ONLY the JSON
fields that process writes to disk - roblet_simulator.py runs as its own
OS process per individual (see sim_executor.py), so no raw trajectory
data ever comes back to this process.

Scope for this pass: f2 (evolved-gait forward velocity) and f5 (shape
entropy - see below) are real. f1 (a separate flat-state velocity) is
ignored for now - roblet_simulator.run_headless runs one rollout per
model, driven by each module's own evolved `hinge_angle` (via the XML's
<custom> ctrl_joint fields), not a separate all-flat baseline pass. f3, f4
are dummy placeholders (always 0.0) - see their docstrings for what a
real implementation needs.

f5 ("entropy") is a single NSGA-III objective built from entropy_api.py's
2D/3D shape entropy (computed inside roblet_simulator.run_headless and
written into stats.json as shape_entropy_2d/shape_entropy_3d): f5 is
their normalized delta (H3D - H2D, "folding complexity gain"), which is
what NSGA-III/the RL reward actually optimizes. Both raw components are
ALSO exposed in compute_objectives()'s return dict, purely for AO-2's
"relationship with locomotion behaviour and evolutionary outcomes"
analysis (logged in every generation's population.json) - not fed back
into selection separately, so 2D/3D don't silently double the delta's
influence on scalarize()/NSGA-III (see scalarize()'s docstring).
"""

import numpy as np

OBJECTIVE_NAMES = [
    "f1_forward_velocity_flat",
    "f2_forward_velocity_folded",
    "f3_relative_yaw",
    "f4_gait_stability",
    "f5_entropy",
]

# True = higher is better (maximize), False = lower is better (minimize).
# pymoo/NSGA-III minimizes everything, so moo_api flips sign on the
# "maximize" objectives via to_minimization_vector() below.
MAXIMIZE = {
    "f1_forward_velocity_flat": True,
    "f2_forward_velocity_folded": True,
    "f3_relative_yaw": True,
    "f4_gait_stability": False,
    "f5_entropy": True,
}


def _f1_forward_velocity_flat(stats):
    """IGNORED for now - always 0.0. roblet_simulator.py runs a single
    rollout per model (the graph's own evolved hinge angles), not a
    separate all-flat baseline pass, so there is nothing to score here."""
    return 0.0


def _f2_forward_velocity_folded(stats):
    """Forward displacement speed (m/s) of the single evolved-gait
    rollout, read directly from stats.json's average_velocity_mmps.
    0.0 if the run failed (stats["success"] == 0)."""
    if not stats.get("success", 0):
        return 0.0
    return float(stats.get("average_velocity_mmps", 0.0)) / 1000.0


def _f3_relative_yaw(stats):
    """PLACEHOLDER (dummy) - always 0.0.

    TODO: needs a whole-body heading tracked and written into stats.json
    by roblet_simulator.py (e.g. from module_1's own orientation), since
    this process only ever sees the JSON, never raw trajectories.
    """
    return 0.0


def _f4_gait_stability(stats):
    """PLACEHOLDER (dummy) - always 0.0.

    """
    return 0.0


def _f5_entropy(stats):
    """"Folding complexity gain": normalized 3D shape entropy minus
    normalized 2D shape entropy (entropy_api.py, computed and written
    into stats.json by roblet_simulator.run_headless). Maximized -
    rewards morphologies whose folding process meaningfully transforms
    structural complexity, rather than folding being a shape no-op.
    0.0 for a failed run (both components default to 0.0 in that case)."""
    return float(stats.get("shape_entropy_3d", 0.0)) - float(stats.get("shape_entropy_2d", 0.0))


_OBJECTIVE_FUNCS = {
    "f1_forward_velocity_flat": _f1_forward_velocity_flat,
    "f2_forward_velocity_folded": _f2_forward_velocity_folded,
    "f3_relative_yaw": _f3_relative_yaw,
    "f4_gait_stability": _f4_gait_stability,
    "f5_entropy": _f5_entropy,
}


def compute_objectives(stats):
    """stats: the dict loaded from a roblet_simulator.py stats.json (or an
    equivalent all-failed dict for a graph that couldn't even be built -
    see moo_api.py's ModuleCollisionError handling). Returns
    dict[name -> float] in natural ("MAXIMIZE says which way is good")
    units - the 5 core objectives (f1..f5) PLUS shape_entropy_2d/3d
    (f5's raw components, for AO-2 analysis only - see this module's
    docstring for why they're not separately optimized)."""
    objectives = {name: fn(stats) for name, fn in _OBJECTIVE_FUNCS.items()}
    objectives["shape_entropy_2d"] = float(stats.get("shape_entropy_2d", 0.0))
    objectives["shape_entropy_3d"] = float(stats.get("shape_entropy_3d", 0.0))
    return objectives


def to_minimization_vector(objectives_dict):
    """np.ndarray[5] in pymoo's minimize-everything convention."""
    return np.array(
        [(-objectives_dict[n] if MAXIMIZE[n] else objectives_dict[n]) for n in OBJECTIVE_NAMES],
        dtype=float,
    )


def scalarize(objectives_dict):
    """Single 'higher is better' scalar (equal-weighted sum, each term
    sign-flipped to a common maximize direction). Used by moo_api.py as
    the RL reward signal (child improvement vs. parent), not for NSGA-III
    selection itself (which uses the full 5-vector) - also what main.py's
    "Individual index" log line and the visualizer's Population-tab
    ranking use, so they all agree (see evolution_results_visualizer.py's
    _aggregate_fitness).

    Iterates OBJECTIVE_NAMES explicitly, NOT objectives_dict.items() -
    compute_objectives() returns extra logging-only fields
    (shape_entropy_2d/3d, f5's raw components) alongside the 5 core
    objectives; summing "whatever's in the dict" would silently double
    that entropy delta's weight in every score this function drives."""
    return float(sum(
        objectives_dict[n] if MAXIMIZE[n] else -objectives_dict[n] for n in OBJECTIVE_NAMES
    ))


def collision_constraint(stats):
    """pymoo constraint convention: <= 0 is feasible. stats["success"] is
    0 whenever a MuJoCo warning (bad qpos/qvel/qacc) fired during the
    rollout, or when the graph couldn't even be built into a valid model
    (see moo_api.py's ModuleCollisionError handling)."""
    return 1.0 if not stats.get("success", 0) else -1.0
