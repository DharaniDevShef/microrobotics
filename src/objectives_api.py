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
    "f6_pheromone_yaw_response",
    "f7_pheromone_speed_response",
]

# True = higher is better (maximize), False = lower is better (minimize).
# pymoo/NSGA-III minimizes everything, so moo_api flips sign on the
# "maximize" objectives via to_minimization_vector() below.
#
# f6/f7's own entries here are just the "attractive" default - main.py
# calls configure_pheromone_response() once at startup (from its
# PHEROMONE_RESPONSE_TYPE switch) to flip them for a "repulsive" run - see
# that function's docstring for why both objectives read the SAME signed
# stats.json fields either way, with only the optimization direction
# changing between the two.
MAXIMIZE = {
    "f1_forward_velocity_flat": True,
    "f2_forward_velocity_folded": True,
    "f3_relative_yaw": True,
    "f4_gait_stability": False,
    "f5_entropy": True,
    "f6_pheromone_yaw_response": True,
    "f7_pheromone_speed_response": True,
}

# main.py's PHEROMONE_RESPONSE_TYPE must be one of these.
PHEROMONE_RESPONSE_TYPES = ("attractive", "repulsive")


def configure_pheromone_response(mode):
    """Sets whether this run's two pheromone objectives (f6_pheromone_
    yaw_response, f7_pheromone_speed_response) are optimized "attractive"
    (turn TOWARD a one-sided light stimulus and speed up under a full-width
    one - Reaction Primitives 2.1/3.2) or "repulsive" (turn away and slow
    down - RPs 2.2/3.1). Call once at process startup (main.py does, from
    its PHEROMONE_RESPONSE_TYPE constant) before any compute_objectives()
    call.

    Both objectives always read the SAME two signed stats.json fields
    (pheromone_yaw_response_deg, pheromone_speed_response -
    positive = toward the light / sped up, per
    roblet_simulator.run_headless_light_tests) regardless of mode - sharing
    one sign convention between the two runs is what keeps mirror symmetry
    (left vs. right turning) "free" within EACH run, per the design
    decided with the user: only the optimization DIRECTION flips here,
    never the metric's own sign. Sharing hinge_angle_on_light_detection as
    one scalar lever in two opposite directions within a SINGLE run would
    be a self-contradictory objective pair (push it up AND down at once) -
    that's exactly why "attractive" and "repulsive" are two separate
    evolution runs (main.py's OUTPUT_DIR already varies by this switch),
    not two objectives added to one run.
    """
    if mode not in PHEROMONE_RESPONSE_TYPES:
        raise ValueError(f"Unknown pheromone_response_type {mode!r}; expected one of {PHEROMONE_RESPONSE_TYPES}")
    is_attractive = mode == "attractive"
    MAXIMIZE["f6_pheromone_yaw_response"] = is_attractive
    MAXIMIZE["f7_pheromone_speed_response"] = is_attractive


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


def _f6_pheromone_yaw_response(stats):
    """Signed yaw rotation (deg) induced by a one-sided light stimulus
    (roblet_simulator.run_headless_light_tests' "left" stage), relative to
    this individual's own no-light baseline - positive = turned TOWARD the
    stimulus, negative = away. Read straight from stats.json's
    pheromone_yaw_response_deg. 0.0 whenever the run failed (light
    tests are skipped for a failed run - see save_simulation_stats'
    docstring) or include_light_tests wasn't requested (defaults to 0.0
    either way, so this degrades to "no signal" rather than crashing)."""
    if not stats.get("success", 0):
        return 0.0
    return float(stats.get("pheromone_yaw_response_deg", 0.0))


def _f7_pheromone_speed_response(stats):
    """Signed (avg_speed_under_stimulus - baseline) / baseline, from a
    full-width light stimulus (run_headless_light_tests' "front" stage) -
    positive = sped up, negative = slowed down. Read straight from
    stats.json's pheromone_speed_response (signed both ways - a
    "repulsive" run drives this negative, an "attractive" one positive,
    see objectives_api.configure_pheromone_response). 0.0 under the same
    conditions as f6 above."""
    if not stats.get("success", 0):
        return 0.0
    return float(stats.get("pheromone_speed_response", 0.0))


_OBJECTIVE_FUNCS = {
    "f1_forward_velocity_flat": _f1_forward_velocity_flat,
    "f2_forward_velocity_folded": _f2_forward_velocity_folded,
    "f3_relative_yaw": _f3_relative_yaw,
    "f4_gait_stability": _f4_gait_stability,
    "f5_entropy": _f5_entropy,
    "f6_pheromone_yaw_response": _f6_pheromone_yaw_response,
    "f7_pheromone_speed_response": _f7_pheromone_speed_response,
}

# Running (monotonically widening) min/max per objective, observed across
# every individual compute_objectives() has ever been called on this run -
# scalarize()'s normalization reference. Module-level (one process = one
# run) and persisted via get_normalization_state()/set_normalization_state()
# so a checkpoint resume doesn't reset it - see checkpoint.py.
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
    """For checkpoint.py: restores the running min/max on resume. `state`
    may be None (checkpoints saved before this existed) - leaves the
    (empty) running range to rebuild itself from this process's own
    evaluations instead of crashing."""
    global _RUNNING_MIN, _RUNNING_MAX
    if state:
        _RUNNING_MIN = dict(state.get("min", {}))
        _RUNNING_MAX = dict(state.get("max", {}))


def compute_objectives(stats):
    """stats: the dict loaded from a roblet_simulator.py stats.json (or an
    equivalent all-failed dict for a graph that couldn't even be built -
    see moo_api.py's ModuleCollisionError handling). Returns
    dict[name -> float] in natural ("MAXIMIZE says which way is good")
    units - the 5 core objectives (f1..f5) PLUS shape_entropy_2d/3d
    (f5's raw components, for AO-2 analysis only - see this module's
    docstring for why they're not separately optimized).

    Also feeds the new objectives into _update_running_range() - this is
    the one place every individual (parent or offspring) ever gets its
    objectives computed (see moo_api.evaluate_individual), so it's the
    natural hook for keeping scalarize()'s normalization reference
    current without touching any other call site."""
    objectives = {name: fn(stats) for name, fn in _OBJECTIVE_FUNCS.items()}
    objectives["shape_entropy_2d"] = float(stats.get("shape_entropy_2d", 0.0))
    objectives["shape_entropy_3d"] = float(stats.get("shape_entropy_3d", 0.0))
    _update_running_range(objectives)
    return objectives


def to_minimization_vector(objectives_dict):
    """np.ndarray[5] in pymoo's minimize-everything convention."""
    return np.array(
        [(-objectives_dict[n] if MAXIMIZE[n] else objectives_dict[n]) for n in OBJECTIVE_NAMES],
        dtype=float,
    )


def scalarize(objectives_dict):
    """Single 'higher is better' scalar: each objective is min-max
    normalized against its running observed range (_RUNNING_MIN/MAX,
    updated by every compute_objectives() call this run - see there) into
    a common [0, 1] "higher is better" scale, then averaged with equal
    weight. Used by moo_api.py as the RL reward signal (child improvement
    vs. parent), not for NSGA-III selection itself (which uses the full
    5-vector) - also what main.py's "Individual index" log line and the
    visualizer's Population-tab ranking use, so they all agree (see
    evolution_results_visualizer.py's _aggregate_fitness).

    Raw equal-COEFFICIENT summing (the previous behavior) is not actually
    equal WEIGHT when objectives live on very different scales - e.g. f2
    (~0.01-0.1 m/s) vs f5 (~0.001-0.01 entropy delta) - f2 would dominate
    every score regardless of f5's value. Normalizing first is what makes
    "equal weight" meaningful. An objective that hasn't shown any
    variation yet (zero range - this is what f1/f3/f4's dummy 0.0
    placeholders look like) is excluded from the average rather than
    forced to a fake 0/0 normalized value, so only genuinely "active"
    objectives share the weight; if NONE have shown variation yet (e.g.
    the very first individual ever scored this run), falls back to the
    old raw signed sum so this never divides by zero.

    Note: because the normalization range widens as the run progresses,
    replotting an earlier generation's scalarized score later in the same
    run can shift it slightly (it's normalized against the fuller range
    now known) - this is the standard tradeoff of online-normalized
    reward (e.g. RL's running reward normalization), not a bug.

    Iterates OBJECTIVE_NAMES explicitly, NOT objectives_dict.items() -
    compute_objectives() returns extra logging-only fields
    (shape_entropy_2d/3d, f5's raw components) alongside the 5 core
    objectives; summing "whatever's in the dict" would silently double
    that entropy delta's weight in every score this function drives."""
    contributions = []
    for n in OBJECTIVE_NAMES:
        lo, hi = _RUNNING_MIN.get(n), _RUNNING_MAX.get(n)
        if lo is None or (hi - lo) < 1e-9:
            continue
        norm = (objectives_dict[n] - lo) / (hi - lo)
        contributions.append(norm if MAXIMIZE[n] else (1.0 - norm))

    if not contributions:
        return float(sum(
            objectives_dict[n] if MAXIMIZE[n] else -objectives_dict[n] for n in OBJECTIVE_NAMES
        ))
    return float(np.mean(contributions))


def collision_constraint(stats):
    """pymoo constraint convention: <= 0 is feasible. stats["success"] is
    0 whenever a MuJoCo warning (bad qpos/qvel/qacc) fired during the
    rollout, or when the graph couldn't even be built into a valid model
    (see moo_api.py's ModuleCollisionError handling)."""
    return 1.0 if not stats.get("success", 0) else -1.0
