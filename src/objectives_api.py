"""
Objectives API - Fitness Evaluator.

Turns mujoco_api.evaluate_graph()'s raw trajectories into the design
doc's f1..f5 objective vector.

Scope for this pass: f1, f2, f5 are real; f3 and f4 are dummy
placeholders (always 0.0) - see their docstrings below for what a real
implementation needs.
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


def _forward_velocity(rollout):
    """Mean horizontal (x-y) COM speed (m/s)."""
    if rollout.com_vel.shape[0] == 0:
        return 0.0
    horizontal_speed = np.linalg.norm(rollout.com_vel[:, :2], axis=1)
    return float(np.mean(horizontal_speed))


def _f1_forward_velocity_flat(state1, state2):
    """Baseline forward displacement speed in the flat (State 1) rollout."""
    return _forward_velocity(state1)


def _f2_forward_velocity_folded(state1, state2):
    """Forward displacement speed in the folded (State 2) rollout."""
    return _forward_velocity(state2)


def _f3_relative_yaw(state1, state2):
    """PLACEHOLDER (dummy) - always 0.0.

    TODO: real implementation needs a well-defined whole-body heading
    (e.g. tracked from module_1's own body orientation via data.xmat,
    not the subtree COM which has no orientation) sampled at the start
    and end of each rollout, then the flat-vs-folded heading delta.
    """
    return 0.0


def _f4_gait_stability(state1, state2):
    """PLACEHOLDER (dummy) - always 0.0.

    TODO: real implementation should penalize vertical (z) COM
    oscillation during locomotion, e.g. -std(state2.com_pos[:, 2]) (or
    some other z-jitter measure), evaluated after the torque ramp settles.
    """
    return 0.0


def _f5_entropy(state1, state2):
    """Shannon entropy (bits) of the folded-state COM heading-direction
    histogram - a real, if simple, proxy for "diversity/complexity" of the
    locomotion pattern: a robot walking a straight line has low entropy; one
    that wanders/turns unpredictably has high entropy."""
    vel = state2.com_vel
    if vel.shape[0] < 2:
        return 0.0
    speed = np.linalg.norm(vel[:, :2], axis=1)
    moving = speed > 1e-6
    if not np.any(moving):
        return 0.0
    headings = np.arctan2(vel[moving, 1], vel[moving, 0])
    n_bins = 16
    hist, _ = np.histogram(headings, bins=n_bins, range=(-np.pi, np.pi))
    probs = hist[hist > 0] / hist.sum()
    return float(-np.sum(probs * np.log2(probs)))


_OBJECTIVE_FUNCS = {
    "f1_forward_velocity_flat": _f1_forward_velocity_flat,
    "f2_forward_velocity_folded": _f2_forward_velocity_folded,
    "f3_relative_yaw": _f3_relative_yaw,
    "f4_gait_stability": _f4_gait_stability,
    "f5_entropy": _f5_entropy,
}


def compute_objectives(rollout_result):
    """rollout_result: the dict returned by mujoco_api.evaluate_graph().
    Returns dict[name -> float] in natural ("MAXIMIZE says which way is
    good") units."""
    state1, state2 = rollout_result["state1"], rollout_result["state2"]
    return {name: fn(state1, state2) for name, fn in _OBJECTIVE_FUNCS.items()}


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
    selection itself (which uses the full 5-vector)."""
    return float(sum(v if MAXIMIZE[n] else -v for n, v in objectives_dict.items()))


def collision_constraint(rollout_result):
    """pymoo constraint convention: <= 0 is feasible."""
    return 1.0 if rollout_result["collided"] else -1.0
