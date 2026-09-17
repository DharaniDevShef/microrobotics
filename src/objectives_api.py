"""
Objectives API - Fitness Evaluator. Turns a roblet_simulator.py stats.json
dict into the f1..f4 objective vector:

    f1 (folded-gait velocity)    - evolved-gait forward speed (m/s)
    f2 (entropy)                 - normalized 3D shape entropy minus 2D
                                    ("folding-complexity gain", entropy_api.py)
    f3 (pheromone yaw response)  - signed yaw turn under a one-sided light
                                    stimulus vs. no-light baseline
    f4 (pheromone speed response) - signed speed change under a full-width
                                    light stimulus vs. no-light baseline
"""

import numpy as np

OBJECTIVE_NAMES = [
    "f1_folded_gait_velocity",
    "f2_entropy",
    "f3_pheromone_yaw_response",
    "f4_pheromone_speed_response",
]

# True = higher is better (maximize), False = lower is better (minimize).
# pymoo/NSGA-III minimizes everything, so moo_api flips sign via to_minimization_vector().
# f3/f4 here are just the "attractive" default - configure_pheromone_response() flips
# them for a "repulsive" run.
MAXIMIZE = {
    "f1_folded_gait_velocity": True,
    "f2_entropy": True,
    "f3_pheromone_yaw_response": True,
    "f4_pheromone_speed_response": True,
}

# main.py's PHEROMONE_RESPONSE_TYPE must be one of these.
PHEROMONE_RESPONSE_TYPES = ("attractive", "repulsive")

# Pre-renumbering (f1..f7) key names mapped onto current OBJECTIVE_NAMES - see
# migrate_legacy_objectives(). The old unused f1/f3/f4 placeholders have no entry.
_LEGACY_KEY_MAP = {
    "f2_forward_velocity_folded": "f1_folded_gait_velocity",
    "f5_entropy": "f2_entropy",
    "f6_pheromone_yaw_response": "f3_pheromone_yaw_response",
    "f7_pheromone_speed_response": "f4_pheromone_speed_response",
}


def migrate_legacy_objectives(objectives_dict):
    """Maps an older run's pre-renumbering objective keys onto the current
    OBJECTIVE_NAMES scheme. Safe to call unconditionally - a no-op if the
    dict is already in the current schema."""
    if any(name in objectives_dict for name in OBJECTIVE_NAMES):
        return objectives_dict
    migrated = dict(objectives_dict)
    for old_name, new_name in _LEGACY_KEY_MAP.items():
        if old_name in objectives_dict:
            migrated[new_name] = objectives_dict[old_name]
    return migrated


def configure_pheromone_response(mode):
    """Sets whether f3/f4 are optimized "attractive" (turn toward a light
    stimulus and speed up) or "repulsive" (turn away and slow down). Call
    once at process startup (main.py does, from PHEROMONE_RESPONSE_TYPE)
    before any compute_objectives() call."""
    if mode not in PHEROMONE_RESPONSE_TYPES:
        raise ValueError(f"Unknown pheromone_response_type {mode!r}; expected one of {PHEROMONE_RESPONSE_TYPES}")
    is_attractive = mode == "attractive"
    MAXIMIZE["f3_pheromone_yaw_response"] = is_attractive
    MAXIMIZE["f4_pheromone_speed_response"] = is_attractive


def _f1_folded_gait_velocity(stats):
    """Forward speed (m/s) of the evolved-gait rollout. 0.0 if the run failed."""
    if not stats.get("success", 0):
        return 0.0
    return float(stats.get("average_velocity_mmps", 0.0)) / 1000.0


def _f2_entropy(stats):
    """"Folding complexity gain": normalized 3D shape entropy minus normalized 2D."""
    return float(stats.get("shape_entropy_3d", 0.0)) - float(stats.get("shape_entropy_2d", 0.0))


def _f3_pheromone_yaw_response(stats):
    """Signed yaw turn (deg) under a one-sided light stimulus vs. no-light baseline
    (positive = toward the stimulus). 0.0 if the run failed."""
    if not stats.get("success", 0):
        return 0.0
    return float(stats.get("pheromone_yaw_response_deg", 0.0))


def _f4_pheromone_speed_response(stats):
    """Signed speed change under a full-width light stimulus vs. no-light baseline
    (positive = sped up). 0.0 if the run failed."""
    if not stats.get("success", 0):
        return 0.0
    return float(stats.get("pheromone_speed_response", 0.0))


_OBJECTIVE_FUNCS = {
    "f1_folded_gait_velocity": _f1_folded_gait_velocity,
    "f2_entropy": _f2_entropy,
    "f3_pheromone_yaw_response": _f3_pheromone_yaw_response,
    "f4_pheromone_speed_response": _f4_pheromone_speed_response,
}

# Running min/max per objective across every compute_objectives() call this run -
# scalarize()'s normalization reference. Persisted via get/set_normalization_state()
# so a checkpoint resume doesn't reset it.
_RUNNING_MIN = {}
_RUNNING_MAX = {}


def _update_running_range(objectives):
    for n in OBJECTIVE_NAMES:
        v = objectives[n]
        if n not in _RUNNING_MIN or v < _RUNNING_MIN[n]:
            _RUNNING_MIN[n] = v
        if n not in _RUNNING_MAX or v > _RUNNING_MAX[n]:
            _RUNNING_MAX[n] = v


def get_normalization_state():
    """For checkpoint.py: a plain-dict snapshot of the running min/max."""
    return dict(min=dict(_RUNNING_MIN), max=dict(_RUNNING_MAX))


def set_normalization_state(state):
    """For checkpoint.py: restores the running min/max on resume. `state` may
    be None or use older objective names - either way just rebuilds from
    this process's own evaluations instead of crashing."""
    global _RUNNING_MIN, _RUNNING_MAX
    if state:
        _RUNNING_MIN = {k: v for k, v in state.get("min", {}).items() if k in OBJECTIVE_NAMES}
        _RUNNING_MAX = {k: v for k, v in state.get("max", {}).items() if k in OBJECTIVE_NAMES}


def compute_objectives(stats):
    """stats: dict loaded from a roblet_simulator.py stats.json. Returns
    dict[name -> float] for the 4 core objectives plus shape_entropy_2d/3d
    (f2's raw components, logging only), and updates the running normalization range."""
    objectives = {name: fn(stats) for name, fn in _OBJECTIVE_FUNCS.items()}
    objectives["shape_entropy_2d"] = float(stats.get("shape_entropy_2d", 0.0))
    objectives["shape_entropy_3d"] = float(stats.get("shape_entropy_3d", 0.0))
    _update_running_range(objectives)
    return objectives


def to_minimization_vector(objectives_dict):
    """np.ndarray[len(OBJECTIVE_NAMES)] in pymoo's minimize-everything convention.
    Callers reading an older run's JSON should run it through
    migrate_legacy_objectives() first; a missing key reads as 0.0."""
    return np.array(
        [(-objectives_dict.get(n, 0.0) if MAXIMIZE[n] else objectives_dict.get(n, 0.0)) for n in OBJECTIVE_NAMES],
        dtype=float,
    )


def scalarize(objectives_dict):
    """Single 'higher is better' scalar: each objective is min-max normalized
    against its running observed range into [0, 1], then averaged with equal
    weight (objectives with no observed variation yet are excluded, falling
    back to a raw signed sum if none qualify). Used as the RL reward signal
    and for ranking individuals in logs/UI - not for NSGA-III selection
    itself, which uses the full objective vector."""
    contributions = []
    for n in OBJECTIVE_NAMES:
        lo, hi = _RUNNING_MIN.get(n), _RUNNING_MAX.get(n)
        if lo is None or (hi - lo) < 1e-9:
            continue
        norm = (objectives_dict.get(n, 0.0) - lo) / (hi - lo)
        contributions.append(norm if MAXIMIZE[n] else (1.0 - norm))

    if not contributions:
        return float(sum(
            objectives_dict.get(n, 0.0) if MAXIMIZE[n] else -objectives_dict.get(n, 0.0) for n in OBJECTIVE_NAMES
        ))
    return float(np.mean(contributions))


def collision_constraint(stats):
    """pymoo constraint convention: <= 0 is feasible. stats["success"] is
    0 whenever a MuJoCo warning (bad qpos/qvel/qacc) fired during the
    rollout, or when the graph couldn't even be built into a valid model
    (see moo_api.py's ModuleCollisionError handling)."""
    return 1.0 if not stats.get("success", 0) else -1.0
