"""
MuJoCo API - Physics Simulation Engine.

Turns a genotype graph (see roblet_grammar.py) into an MJCF assembly via
helper_scripts/mjcf_generator.py::build_assembly, then runs two headless
rollouts under the same oscillating-magnetic-field gait used by
roblet_simulator.py, and returns raw trajectory data for
objectives_api.py to score.

Unlike roblet_simulator.py (which uses module-level globals and an
interactive viewer, fine for one manual session), every rollout here is
self-contained: no shared mutable state, so evaluate_graph() is safe to
call back-to-back across an EA population (and, later, from multiple
worker processes for the parallel evaluation the design doc calls for).
"""

import json
import os
import sys
import tempfile

import mujoco
import networkx as nx
import numpy as np

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER_SCRIPTS_DIR = os.path.join(_SRC_DIR, "..", "helper_scripts")
if _HELPER_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _HELPER_SCRIPTS_DIR)

from mjcf_generator import build_assembly  # noqa: E402

_MESHDIR = os.path.abspath(os.path.join(_SRC_DIR, "..", "meshes")).replace("\\", "/")

# Physics/actuation constants, mirrored from roblet_simulator.py so an
# evolved morphology reproduces the same magnetic-actuation gait used in
# interactive playback.
B_INTENSITY = 0.01     # Tesla
M_MOMENT = 6.5e-3       # A*m^2
TORQUE_MULTIPLIER = 10
TORQUE_RAMP_TIME = 2.0  # seconds; shortened vs. the 10s interactive demo so EA evaluation stays fast
FREQUENCY = 6.25        # Hz
TIMESTEP = 0.01         # seconds


class RolloutResult:
    """Raw per-step trajectory data for one gait rollout."""

    def __init__(self, times, com_pos, com_vel, collided):
        self.times = np.asarray(times, dtype=float)
        self.com_pos = np.asarray(com_pos, dtype=float).reshape(-1, 3)
        self.com_vel = np.asarray(com_vel, dtype=float).reshape(-1, 3)
        self.collided = collided


def _find_main_movable_parent(model, body_id):
    current_id = body_id
    while current_id != 0:
        parent_id = model.body_parentid[current_id]
        if parent_id == 0:
            return current_id
        current_id = parent_id
    return body_id


def _find_all_magnets(model):
    """body_id -> list[(geom_id, polarity_sign)], as in roblet_simulator.py."""
    magnet_map = {}
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and "magnet_" in geom_name:
            immediate_body_id = model.geom_bodyid[geom_id]
            movable_parent_id = _find_main_movable_parent(model, immediate_body_id)
            polarity_sign = 1.0 if "SGA" in geom_name else -1.0
            magnet_map.setdefault(movable_parent_id, []).append((geom_id, polarity_sign))
    return magnet_map


def _fold_joint_order(G):
    """Node ids in the same order mjcf_generator.build_assembly assigns
    actuators (sorted-by-id, foldable modules only) - so actuator index i
    (0-based) always corresponds to fold_joint_order(G)[i]."""
    nonrigid = [n for n in G.nodes if G.nodes[n]["module_type"] != "non-foldable"]
    return sorted(nonrigid, key=lambda x: int(x.split("_")[1]))


def _hinge_targets_rad(G, folded):
    """actuator_index -> target angle (radians), signed to match each
    joint's ctrlrange (valley fold: [0, +90deg], Mountain fold: [-90deg, 0])."""
    targets = {}
    for i, node in enumerate(_fold_joint_order(G)):
        if not folded:
            targets[i] = 0.0
            continue
        m_type = G.nodes[node]["module_type"]
        angle_deg = G.nodes[node]["hinge_angle"]
        sign = 1.0 if m_type == "valley fold" else -1.0
        targets[i] = sign * np.radians(angle_deg)
    return targets


def _make_control_callback(magnet_map, joint_targets_rad, torque_ramp_time):
    def callback(model, data):
        data.xfrc_applied.fill(0)

        ramp_progress = min(data.time / torque_ramp_time, 1.0) if torque_ramp_time > 0 else 1.0
        ramp_factor = ramp_progress * ramp_progress * (3 - 2 * ramp_progress)  # smoothstep
        effective_multiplier = TORQUE_MULTIPLIER * ramp_factor

        total_cycle = 1.0 / FREQUENCY
        half_cycle = total_cycle / 2.0
        t_in_cycle = data.time % total_cycle
        b_z = -B_INTENSITY if t_in_cycle < half_cycle else B_INTENSITY
        b_vector = np.array([0.0, 0.0, b_z])

        for body_id, magnets in magnet_map.items():
            accumulated_torque = np.zeros(3)
            for geom_id, polarity_sign in magnets:
                geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
                local_m = np.array([0.0, 0.0, 1.0]) * M_MOMENT * polarity_sign
                world_m = geom_mat.dot(local_m)
                accumulated_torque += np.cross(world_m, b_vector)
            data.xfrc_applied[body_id][3:6] = accumulated_torque * effective_multiplier

        if model.nu > 0:
            for act_id, target_rad in joint_targets_rad.items():
                data.ctrl[act_id] = target_rad

    return callback


def _run_single_rollout(model, magnet_map, joint_targets_rad, sim_seconds, torque_ramp_time):
    data = mujoco.MjData(model)
    callback = _make_control_callback(magnet_map, joint_targets_rad, torque_ramp_time)
    mujoco.set_mjcb_control(callback)

    times, com_pos, com_vel = [], [], []
    collided = False
    n_steps = max(1, int(sim_seconds / model.opt.timestep))
    try:
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            mujoco.mj_subtreeVel(model, data)

            # "Collision Detection via failure report from MuJoCo" (design
            # doc constraint): a bad contact/constraint configuration in a
            # generated morphology shows up as numerical blow-up rather
            # than a clean exception, so NaN/Inf state is our failure
            # signal - stop the rollout and flag the individual infeasible.
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                collided = True
                break

            times.append(data.time)
            com_pos.append(data.subtree_com[0].copy())
            com_vel.append(data.subtree_linvel[0].copy())
    finally:
        mujoco.set_mjcb_control(None)
        data.xfrc_applied.fill(0)

    return RolloutResult(times, com_pos, com_vel, collided)


def evaluate_graph(G, work_dir=None, sim_seconds=3.0, torque_ramp_time=TORQUE_RAMP_TIME, cleanup=True):
    """Builds an MJCF assembly for genotype `G` and runs both design-state
    rollouts, returning {"state1": RolloutResult, "state2": RolloutResult,
    "collided": bool} for objectives_api.compute_objectives().

    State 1 ("rest flat"): every foldable hinge held at 0 deg - baseline
    planar gait driven purely by the oscillating magnetic field.
    State 2 ("folded"): every foldable hinge driven to its own evolved
    `hinge_angle` design variable.

    (The design doc phrases State 1 as "all hinges flat except one" - a
    single dedicated gait hinge on the fixed hand-designed layout. RL-grown
    graphs don't have a fixed "the gait hinge", so this generalizes State 1
    to "all hinges flat" / State 2 to "all hinges at their evolved angle".
    Revisit if the original single-hinge semantics matter for your use case.)
    """
    work_dir = work_dir or tempfile.mkdtemp(prefix="roblet_eval_")
    os.makedirs(work_dir, exist_ok=True)
    graph_json_path = os.path.join(work_dir, "candidate_graph.json")
    xml_path = os.path.join(work_dir, "candidate_assembly.xml")

    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f)

    build_assembly(graph_json_path, xml_path, meshdir=_MESHDIR)

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = TIMESTEP
    magnet_map = _find_all_magnets(model)

    state1 = _run_single_rollout(model, magnet_map, _hinge_targets_rad(G, folded=False),
                                  sim_seconds, torque_ramp_time)
    state2 = _run_single_rollout(model, magnet_map, _hinge_targets_rad(G, folded=True),
                                  sim_seconds, torque_ramp_time)

    if cleanup:
        for path in (graph_json_path, xml_path):
            try:
                os.remove(path)
            except OSError:
                pass

    return {"state1": state1, "state2": state2, "collided": state1.collided or state2.collided}
