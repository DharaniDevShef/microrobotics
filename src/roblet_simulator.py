"""
Roblet Simulator - A simulation environment for testing self folding and
magnetic actuation of microrobots using MuJoCo physics engine.
"""

# pylint: disable=no-member

import argparse
import json
import logging
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack
import imageio
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

import entropy_api

# Global dictionaries mapping Parent Body ID -> List of (Magnet Geom ID, Polarity Sign)
parent_body_magnet_map = {}

# Vectorized-callback view of parent_body_magnet_map, rebuilt by
# _prepare_magnet_arrays() so the per-step hot loop doesn't walk the dict.
_magnet_geom_ids = np.zeros(0, dtype=np.int32)  # one row per (parent, magnet)
_magnet_moments = np.zeros(0, dtype=np.float64)  # M_MOMENT * polarity_sign, same rows
_magnet_owner = np.zeros(0, dtype=np.int32)  # row -> index into _magnet_parent_body_ids
_magnet_parent_body_ids = np.zeros(0, dtype=np.int32)  # one entry per parent body

# Torque tracking global variables
torque_history = []

# B-field tracking global variables (paired with torque_history, one entry per step)
b_field_history = []
time_history = []

# {actuator_idx: (last_angle_deg, last_time_s, last_setpoint_deg, reported)} -
# running state for light_sensitive_joint_angle_deg()'s exponential
# relaxation curve across set_angle_to_joint() calls. Cleared at the start
# of every run_headless/run_with_viewer/run_light_tests call.
_light_joint_state = {}

# {id(model): {body_id: frozenset(body ids in the same physical module)}} -
# lets get_light_sensor_values()'s occlusion raycast skip hits on the
# sensor's own module. Cleared alongside parent_body_magnet_map each run.
_module_self_body_cache = {}

logger = logging.getLogger(__name__)

# Physics constants
# Magnetic field intensity (Tesla)
B_INTENSITY = 0.025  # mT
# Candidate drive strengths for run_headless_b_sweep() - the field needed
# to walk (rather than stall or over-drive) is morphology-dependent.
B_SWEEP_VALUES = (0.001, 0.008, 0.025)
# 6.5 x 10^-3 Am^2 - 2mm x 2mm neodymium magnet cylinder (N42SH)
M_MOMENT = 6.5e-3
# Maximum torque multiplier
TORQUE_MULTIPLIER = 1
# Time taken to reach full torque
TORQUE_RAMP_TIME = 1  # seconds
FREQUENCY = 6.25  # Hz
TOTAL_CYCLE_TIME = 1.0 / FREQUENCY

# Rangefinder reading (m) below which a wall is considered "reached".
WALL_STOP_DISTANCE = 200  # 200 mm

# Offscreen screenshot/GIF render resolution.
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080

# run_light_tests()'s capture_video renderer is a separate diagnostic path,
# so it can use the full offscreen buffer for maximum video quality.
VIDEO_RENDER_WIDTH = 1920
VIDEO_RENDER_HEIGHT = 1080

# Pre-gait "settle" phase (see run_headless): lets hinge actuators reach
# their evolved target angles and the body settle before a screenshot.
SETTLE_MAX_STEPS = 300  # 10 steps * 0.01s timestep = 0.1s settle phase
SETTLE_VELOCITY_THRESHOLD = 1e-3  # rad/s or m/s - "basically stopped moving"

# 2D/3D shape entropy (entropy_api.py) window sizes, "radius r" hex-cell
# units, window_size = 2r + 1.
SHAPE_ENTROPY_WINDOW_SIZES = (3, 5, 7)  # r = 1, 2, 3

# Wall geoms are tagged group=1 in the model (see mjcf_generator.py) so this
# raycast can be filtered to see ONLY them.
_WALL_GEOMGROUP = np.zeros(6, dtype=np.uint8)
_WALL_GEOMGROUP[1] = 1

# The cosmetic `luminance_sheet` geom (group=2) would otherwise register as
# a giant occluder in mj_ray (which ignores contype/conaffinity), so the
# occlusion raycast uses "every group except 2" instead of wall-only.
_OCCLUSION_GEOMGROUP = np.ones(6, dtype=np.uint8)
_OCCLUSION_GEOMGROUP[2] = 0

# Duration (s) of magnetic actuation applied per run_light_tests() stage.
LIGHT_TEST_DEFAULT_DURATION = 10
# Duration (s) run_light_tests() moves the assembly with no light active,
# before the light stages start, to get a baseline avg linear velocity.
LIGHT_TEST_BASELINE_DURATION = 7.0
# Extra distance (m) the "left" stage's light is pushed beyond the left
# half-region's own centroid.
LIGHT_TEST_LEFT_EXTRA_OFFSET = 0.01
# Minimum net baseline (no-light) displacement (m) needed before the
# direction is trustworthy enough to reorient the assembly by.
LIGHT_TEST_MIN_BASELINE_TRAVEL = 0.0005
# How far (m) below the lowest light_sensor_* site pheromone_light_height()
# places the floor-level "luminance sheet" light.
LIGHT_TEST_PHEROMONE_MARGIN_M = 0.001
# World Z height (m) _show_luminance_sheet() renders the visible yellow
# patch at - just above the floor so it's never hidden behind it.
LUMINANCE_SHEET_VISUAL_HEIGHT_M = 0.0003
# How much bigger (X and Y) _show_luminance_sheet() draws the visible patch
# than the stage's own light_bounds rectangle - purely cosmetic.
LUMINANCE_SHEET_VISUAL_SCALE = 5.0
# Visible RGBA of the `luminance_sheet` placeholder geom - bright yellow,
# matching the UV-excited fluorescent pheromone trace it represents.
LUMINANCE_SHEET_RGBA = np.array([1.0, 0.85, 0.0, 1.0])
# run_light_tests() reactive fold: a joint whose light_sensor_joint_* reading
# exceeds this (lux) is driven to its own evolved light_ctrl_jointN angle.
LIGHT_TRIGGER_LUX_THRESHOLD = 10000.0
# Flat lux reading get_light_sensor_values() reports for any sensor inside a
# bounded light's footprint - fixed constant, deliberately kept separate
# from model.light_intensity (which is purely cosmetic).
LUMINANCE_SHEET_LUX = 20000.0
# Time constant tau_c (s) of a light-sensitive joint's exponential
# relaxation response - see light_sensitive_joint_angle_deg().
LIGHT_JOINT_TIME_CONSTANT_S = 1
# How close (deg) a light-sensitive joint's eased angle has to get to its
# setpoint before it's considered arrived, for the one-shot "reached"/
# "returned" log lines.
LIGHT_JOINT_REACHED_TOLERANCE_DEG = 0.5
# get_light_sensor_values()'s occlusion raycast: how many self-hit bounces
# off the sensor's own module it will skip before giving up.
_SELF_OCCLUSION_MAX_HITS = 6
# Distance (m) a skipped self-hit's ray origin is nudged forward before
# re-casting, so mj_ray doesn't immediately re-hit the same surface.
_SELF_OCCLUSION_RAY_EPS = 1e-6



def _prepare_magnet_arrays():
    """Flattens parent_body_magnet_map into the arrays magnetic_field_callback
    needs. Call once after each parent_body_magnet_map.clear()/.update()
    pair, not from inside the per-step callback itself."""
    global _magnet_geom_ids, _magnet_moments, _magnet_owner, _magnet_parent_body_ids
    parents = sorted(parent_body_magnet_map)
    geom_ids, moments, owner = [], [], []
    for idx, body_id in enumerate(parents):
        for geom_id, polarity_sign in parent_body_magnet_map[body_id]:
            geom_ids.append(geom_id)
            moments.append(M_MOMENT * polarity_sign)
            owner.append(idx)
    _magnet_geom_ids = np.array(geom_ids, dtype=np.int32)
    _magnet_moments = np.array(moments, dtype=np.float64)
    _magnet_owner = np.array(owner, dtype=np.int32)
    _magnet_parent_body_ids = np.array(parents, dtype=np.int32)


def magnetic_field_callback(model, data):
    """MuJoCo control callback: computes magnetic torque (Tau = M x B), applies
    it to parent module bodies, and records net applied torque magnitude.
    Vectorized over every magnet at once (see _prepare_magnet_arrays) since
    MuJoCo calls this once per physics step."""
    data.xfrc_applied.fill(0)
    ramp_progress = min(data.time / TORQUE_RAMP_TIME, 1.0)
    # Smoothstep: f(x)=x^2*(3-2x), 0 -> 1 with zero slope at start and end.
    ramp_factor = ramp_progress * ramp_progress * (3 - 2 * ramp_progress)
    effective_torque_multiplier = TORQUE_MULTIPLIER * ramp_factor

    # Oscillating magnetic field along Z-axis to induce walking motion: flips
    # sign at frequency f. Phase 1 (B=-z) pitches the robot forward about its
    # front foot; phase 2/3 (B=+z) lifts the front foot and lands on the rear.
    half_cycle_time = TOTAL_CYCLE_TIME / 2.0
    time_in_cycle = data.time % TOTAL_CYCLE_TIME

    b_z = -B_INTENSITY if time_in_cycle < half_cycle_time else B_INTENSITY

    n_parents = len(_magnet_parent_body_ids)
    if n_parents == 0:
        torque_history.append(0.0)
        b_field_history.append(b_z)
        time_history.append(data.time)
        return

    # World-frame dipole X/Y components for every magnet: indices 2 and 5 of
    # the flattened xmat are the local +Z axis in world coords, scaled by moment.
    geom_mats = data.geom_xmat[_magnet_geom_ids]
    dipole_x = geom_mats[:, 2] * _magnet_moments
    dipole_y = geom_mats[:, 5] * _magnet_moments

    # Tau = m x B, B=(0,0,b_z) -> (my*b_z, -mx*b_z, 0), summed per parent body.
    per_magnet_torque = np.stack(
        [dipole_y * b_z, -dipole_x * b_z, np.zeros_like(dipole_x)], axis=1
    )
    accumulated_torque = np.zeros((n_parents, 3))
    np.add.at(accumulated_torque, _magnet_owner, per_magnet_torque)

    ramped_torque = accumulated_torque * effective_torque_multiplier
    data.xfrc_applied[_magnet_parent_body_ids, 3:6] = ramped_torque

    # Record magnitude of total torque on whole body for this step
    torque_history.append(np.linalg.norm(ramped_torque.sum(axis=0)))
    b_field_history.append(b_z)
    time_history.append(data.time)


def find_main_movable_parent(model, body_id):
    """Traces up the MuJoCo body tree to find the main movable module body."""
    current_id = body_id
    while current_id != 0:
        parent_id = model.body_parentid[current_id]
        if parent_id == 0:
            return current_id
        current_id = parent_id
    return body_id


def _module_self_bodies(model):
    """{body_id: frozenset(body ids in the same physical module_N subtree)}
    for every body in `model`. Used by get_light_sensor_values() so its
    occlusion raycast can treat a hit on the sensor's own module as
    transparent rather than a real occluder."""
    children = [[] for _ in range(model.nbody)]
    for body_id in range(1, model.nbody):
        children[int(model.body_parentid[body_id])].append(body_id)

    def subtree(root):
        members = [root]
        for child in children[root]:
            members.extend(subtree(child))
        return members

    result = {}
    for root in children[0]:  # world's direct children = the module_N bodies
        members = frozenset(subtree(root))
        for body_id in members:
            result[body_id] = members
    return result


def find_all_magnets(model):
    """Finds all magnets in the model and maps parent body ID to magnets.
    Polarity is read from the mating connector mesh (SGA/SGB/SGX) at that
    site - SGB sites (mating partner of SGA) get the opposite sign."""
    magnet_map = {}
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and "magnet_" in geom_name:
            immediate_body_id = model.geom_bodyid[geom_id]
            movable_parent_id = find_main_movable_parent(model, immediate_body_id)

            sibling_geom_name = geom_name.replace("magnet_", "geom_", 1)
            sibling_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, sibling_geom_name)
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[sibling_id])
            polarity_sign = -1.0 if mesh_name == "connectorB" else 1.0

            if movable_parent_id not in magnet_map:
                magnet_map[movable_parent_id] = []
            magnet_map[movable_parent_id].append((geom_id, polarity_sign))
    return magnet_map


def find_module_labels(model):
    """Scans model's bodies to map top-level module IDs to their names."""
    module_labels = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        # Identify main module wrappers generated by the assembly builder script
        if body_name and body_name.startswith("module_"):
            module_labels[body_id] = body_name
    return module_labels


def _recolor_light_sensitive_module(model, rgba):
    """Recolors the bodyBase/bodyLink geoms of the light-sensitive module to
    `rgba` so it stands out in the viewer/screenshots/video. No-op if the
    model has no light-sensitive module."""
    module_ids = set()
    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if site_name and site_name.startswith("light_sensor_joint_"):
            module_ids.add(site_name[len("light_sensor_joint_"):])
    for module_id in module_ids:
        for geom_name in (f"geom_bodyBase_{module_id}", f"geom_bodyLink_{module_id}"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                model.geom_rgba[geom_id] = rgba


def _module_positions(data, module_labels):
    """{module_name: world position (m)} for every top-level module body,
    read from `data.xpos`. Caller must have called mj_forward/mj_step
    first to resolve the pose."""
    return {name: data.xpos[body_id].copy() for body_id, name in module_labels.items()}


def read_joint_target_angles_from_xml(model_path):
    """Read per-joint target angles from XML metadata by actuator name."""
    tree = ET.parse(model_path)
    root = tree.getroot()
    target_angles = {}
    for numeric in root.findall(".//custom/numeric"):
        name = numeric.get("name", "")
        if not name:
            continue
        if name.startswith("ctrl_joint"):
            target_angles[name] = float(numeric.get("data", "0.0"))
        elif name.startswith("joint_target_angle_"):
            target_angles[name] = float(numeric.get("data", "0.0"))
    return target_angles


def read_light_target_angles_from_xml(model_path):
    """Read per-joint light-triggered target angles (Design Variable 6,
    hinge_angle_on_light_detection) from XML metadata. Only present for
    light-sensitive joints, so may be empty."""
    tree = ET.parse(model_path)
    root = tree.getroot()
    light_target_angles = {}
    for numeric in root.findall(".//custom/numeric"):
        name = numeric.get("name", "")
        if name.startswith("light_ctrl_joint"):
            light_target_angles[name] = float(numeric.get("data", "0.0"))
    return light_target_angles


def light_sensitive_joint_angle_deg(theta_prev_deg, setpoint_deg, dt_s, tau_c=LIGHT_JOINT_TIME_CONSTANT_S):
    """First-order exponential relaxation of a light-sensitive hinge's angle
    toward `setpoint_deg` over one `dt_s`-second step:

        theta(t) = setpoint + (theta_prev - setpoint) * e^(-dt / tau_c)

    Same formula drives both directions: setpoint is the evolved
    light_ctrl_jointN angle on light entry, the baseline gait angle on
    exit. Called once per physics step, chaining theta_prev across calls
    to trace one continuous curve through on/off transitions."""
    if dt_s <= 0:
        return theta_prev_deg
    return setpoint_deg + (theta_prev_deg - setpoint_deg) * np.exp(-dt_s / tau_c)


def set_angle_to_joint(model, data, target_angle_deg, light_bounds=None, light_target_angles=None, frame=None):
    """Set actuator position targets from a scalar, ordered sequence, or name
    map. If `light_target_angles` is given, any joint with an evolved
    light_ctrl_jointN value eases (via light_sensitive_joint_angle_deg)
    toward that angle while lit, or back to its baseline angle once the
    light is lost, instead of jumping instantly. `frame`: optional
    (origin_xy, forward) forwarded to get_light_sensor_values()."""
    if model.nu == 0:
        return

    if isinstance(target_angle_deg, dict):
        target_values = []
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name is None:
                actuator_name = f"actuator_{i}"
            if actuator_name in target_angle_deg:
                target_values.append(float(target_angle_deg[actuator_name]))
            elif str(i + 1) in target_angle_deg:
                target_values.append(float(target_angle_deg[str(i + 1)]))
            else:
                target_values.append(float(target_angle_deg.get("default", 0.0)))
    elif isinstance(target_angle_deg, (list, tuple, np.ndarray)):
        if len(target_angle_deg) != model.nu:
            raise ValueError(
                f"Expected {model.nu} target angles but received {len(target_angle_deg)}"
            )
        target_values = [float(v) for v in target_angle_deg]
    else:
        target_values = [float(target_angle_deg)] * model.nu

    if light_bounds is not None:
        joint_to_actuator = {}
        for actuator in range(model.nu):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                           model.actuator_trnid[actuator, 0])
            if joint_name:
                joint_to_actuator[joint_name] = actuator

        readings = get_light_sensor_values(model, data, verbose=False, light_bounds=light_bounds, frame=frame)
        for site_name, lux in readings.items():
            actuator_idx = joint_to_actuator.get(site_name.replace("light_sensor_", "", 1))
            if actuator_idx is None:
                continue
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_idx)
            light_key = (actuator_name or "").replace("ctrl_joint", "light_ctrl_joint", 1)
            if not (light_target_angles and light_key in light_target_angles):
                # No evolved light_ctrl_jointN for this joint - leave its
                # target as the baseline pass set it; no reactive fold.
                continue

            baseline_deg = target_values[actuator_idx]
            lit = lux > LIGHT_TRIGGER_LUX_THRESHOLD
            setpoint_deg = light_target_angles[light_key] if lit else baseline_deg
            # First-seen joints default to baseline (not setpoint), so if
            # already lit on the first observation it still counts as a
            # setpoint change.
            theta_prev_deg, prev_time, prev_setpoint_deg, reported = _light_joint_state.get(
                actuator_idx, (baseline_deg, data.time, baseline_deg, True))
            eased_deg = light_sensitive_joint_angle_deg(
                theta_prev_deg, setpoint_deg, data.time - prev_time)

            _light_joint_state[actuator_idx] = (eased_deg, data.time, setpoint_deg, reported)
            target_values[actuator_idx] = eased_deg

    for i in range(model.nu):
        target_deg = float(target_values[i])
        target_rad = np.radians(abs(target_deg))
        lo, hi = model.actuator_ctrlrange[i]

        # Target falls inside positive range (Valley fold)
        if lo >= 0 and lo <= target_rad <= hi:
            data.ctrl[i] = target_rad
        # Target falls inside negative range (Mountain fold)
        elif hi <= 0 and lo <= -target_rad <= hi:
            data.ctrl[i] = -target_rad
        else:
            lo_deg, hi_deg = np.degrees(lo), np.degrees(hi)
            raise ValueError(
                f"Target {target_deg}° (or -{target_deg}°) is out of bounds "
                f"for actuator {i} range [{lo_deg:.1f}°, {hi_deg:.1f}°]"
            )


def distance_to_nearest_wall(model, data):
    """Minimum of the 4 wall-facing rangefinder rays (m), or None if none
    currently hit a wall. Uses a manual, geom-group-filtered mj_ray() rather
    than a native <rangefinder> sensor to avoid false hits on the robot's
    own nearby geometry."""
    geomid = np.zeros(1, dtype=np.int32)
    readings = []
    for wall_dir in ("north", "south", "east", "west"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"rf_{wall_dir}")
        pnt = data.site_xpos[site_id]
        vec = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        dist = mujoco.mj_ray(model, data, pnt, vec, _WALL_GEOMGROUP, True, -1, geomid)
        if dist >= 0:
            readings.append(dist)
    return min(readings) if readings else None


def _travel_frame_axes(forward):
    """(forward, right) unit vectors for a travel-direction-aligned local
    frame: `right` is `forward` rotated -90 deg (clockwise)."""
    forward = np.asarray(forward, dtype=float)
    return forward, np.array([forward[1], -forward[0]])


def world_to_local_xy(point_xy, origin_xy, forward):
    """World XY -> (local_x, local_y) in the travel-aligned frame anchored
    at origin_xy: local_y grows along `forward`, local_x grows to the
    right of it."""
    forward, right = _travel_frame_axes(forward)
    rel = np.asarray(point_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    return float(np.dot(rel, right)), float(np.dot(rel, forward))


def local_to_world_xy(local_xy, origin_xy, forward):
    """Inverse of world_to_local_xy() - places a real MuJoCo <light> (world
    coordinates only) at a position chosen in the local frame."""
    forward, right = _travel_frame_axes(forward)
    lx, ly = local_xy
    return np.asarray(origin_xy, dtype=float) + lx * right + ly * forward


def assembly_local_bounds(model, data, origin_xy, forward):
    """Axis-aligned bounding box, as (mins, maxs) each [x, y], of every robot
    geom (excludes the worldbody), expressed in the travel-aligned local
    frame instead of raw world XY - used to size/center a light's coverage
    footprint over a specific half of the assembly."""
    forward, right = _travel_frame_axes(forward)
    mins = np.array([np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf])
    origin_xy = np.asarray(origin_xy, dtype=float)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        rel = data.geom_xpos[g][:2] - origin_xy
        local = np.array([np.dot(rel, right), np.dot(rel, forward)])
        r = model.geom_rbound[g]
        mins = np.minimum(mins, local - r)
        maxs = np.maximum(maxs, local + r)
    return mins, maxs


def pheromone_light_height(model, data):
    """World Z height (m) to place the floor-level "luminance sheet" light
    at: LIGHT_TEST_PHEROMONE_MARGIN_M below the lowest of the model's own
    light_sensor_* sites, so even the closest-to-ground sensor gets a
    same-side, unoccluded reading. Clamped to never cross world Z=0 when
    every sensor already sits at or above the floor."""
    sensor_zs = [
        float(data.site_xpos[site_id][2])
        for site_id in range(model.nsite)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or "").startswith("light_sensor_")
    ]
    lowest = min(sensor_zs) if sensor_zs else 0.0
    height = lowest - LIGHT_TEST_PHEROMONE_MARGIN_M
    if lowest >= 0.0 and height < 0.0:
        height = lowest / 2.0  # stay on the same (non-negative) side of the floor as `lowest`
    return height


def _place_pheromone_light(model, light_id, center_xy, height):
    """Points the model's spotlight (light_id) straight down from `height`
    meters (see pheromone_light_height()) at `center_xy` (world XY) - the
    floor-level "luminance sheet" light. ambient/diffuse/specular are
    purely cosmetic (get_light_sensor_values() only reads light_intensity);
    light_type/light_pos staying SPOT and floor-level is what actually
    drives sensing."""
    model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_SPOT
    model.light_pos[light_id] = np.array([center_xy[0], center_xy[1], height])
    model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])
    model.light_diffuse[light_id] = np.array([1.0, 0.9, 0.3])
    model.light_ambient[light_id] = np.array([0.1, 0.1, 0.1])
    model.light_specular[light_id] = np.array([0.05, 0.05, 0.05])


def _scale_local_bounds(bounds, anchor_local_xy, scale):
    """Scales a (x_lo, x_hi, y_lo, y_hi) travel-aligned-local-frame rectangle
    by `scale`, growing each edge away from anchor_local_xy rather than the
    rectangle's own center. Shared by _show_luminance_sheet() (the visible
    patch) and the light_bounds passed into get_light_sensor_values() /
    set_angle_to_joint(), so what's rendered matches what's sensed."""
    x_lo, x_hi, y_lo, y_hi = bounds
    ax, ay = anchor_local_xy
    x_lo, x_hi = ax + scale * (x_lo - ax), ax + scale * (x_hi - ax)
    y_lo, y_hi = ay + scale * (y_lo - ay), ay + scale * (y_hi - ay)
    return (x_lo, x_hi, y_lo, y_hi)


def _show_luminance_sheet(model, sheet_geom_id, bounds, anchor_local_xy, origin_xy, forward):
    """Resizes, recolors and re-orients mjcf_generator.py's `luminance_sheet`
    placeholder geom into a visible yellow patch covering `bounds`, scaled
    up by LUMINANCE_SHEET_VISUAL_SCALE - the video-only counterpart to
    _place_pheromone_light(). Always sits at LUMINANCE_SHEET_VISUAL_HEIGHT_M
    above the floor so it renders visibly rather than behind the floor
    geom. anchor_local_xy: see _scale_local_bounds()."""
    x_lo, x_hi, y_lo, y_hi = _scale_local_bounds(bounds, anchor_local_xy, LUMINANCE_SHEET_VISUAL_SCALE)
    center_xy = local_to_world_xy(((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0), origin_xy, forward)

    fwd, right = _travel_frame_axes(forward)
    rot = np.array([
        right[0], fwd[0], 0.0,
        right[1], fwd[1], 0.0,
        0.0, 0.0, 1.0,
    ])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot)
    model.geom_quat[sheet_geom_id] = quat
    model.geom_pos[sheet_geom_id] = np.array([center_xy[0], center_xy[1], LUMINANCE_SHEET_VISUAL_HEIGHT_M])
    model.geom_size[sheet_geom_id] = np.array([
        max(abs(x_hi - x_lo), 1e-4) / 2.0,
        max(abs(y_hi - y_lo), 1e-4) / 2.0,
        0.0002,
    ])
    model.geom_rgba[sheet_geom_id] = LUMINANCE_SHEET_RGBA


def get_light_sensor_values(model, data, verbose=True, light_bounds=None, frame=None):
    """Illuminance (lux) at each on-body light sensor (`light_sensor_*`
    sites), summed over every <light> in the scene. MuJoCo has no native
    illuminance sensor, so this is a hand-rolled photometric estimate:

      lux = intensity * max(0, cos(theta)) / falloff   (0 if occluded)

    where theta is the angle between the sensor's outward normal and the
    direction to the light (Lambert's cosine law), and falloff is 1 for a
    directional light or distance^2 for a positional one. Occlusion uses a
    multi-hit raycast that skips hits on the sensor's own module
    (_module_self_bodies) before counting real occlusion.

    verbose=False skips the print. light_bounds: optional
    {light_id: (x_lo, x_hi, y_lo, y_hi)} restricts a light to a rectangular
    footprint (e.g. the luminance sheet) - inside reads a flat `intensity`
    lux regardless of angle/occlusion, outside reads 0. frame: optional
    (origin_xy, forward) to interpret light_bounds in the travel-aligned
    local frame instead of raw world XY."""
    self_bodies_map = _module_self_body_cache.get(id(model))
    if self_bodies_map is None:
        self_bodies_map = _module_self_bodies(model)
        _module_self_body_cache[id(model)] = self_bodies_map

    geomid = np.zeros(1, dtype=np.int32)
    readings = {}
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not name or not name.startswith("light_sensor_"):
            continue
        pnt = data.site_xpos[site_id]
        body_id = model.site_bodyid[site_id]
        normal = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        self_bodies = self_bodies_map.get(body_id, frozenset((body_id,)))

        lux = 0.0
        for light_id in range(model.nlight):
            if not model.light_active[light_id]:
                continue
            intensity = model.light_intensity[light_id]
            if intensity <= 0:
                continue

            bounds = light_bounds.get(light_id) if light_bounds else None
            if bounds is not None:
                x_lo, x_hi, y_lo, y_hi = bounds
                px, py = (world_to_local_xy(pnt[:2], frame[0], frame[1]) if frame is not None
                          else (pnt[0], pnt[1]))
                if not (x_lo <= px <= x_hi and y_lo <= py <= y_hi):
                    continue
                # Luminance sheet: a floor-level puddle, not a directional
                # beam - flat lux inside the footprint regardless of angle
                # or occlusion, 0 outside.
                lux += LUMINANCE_SHEET_LUX
                continue

            if model.light_type[light_id] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                to_light = -data.light_xdir[light_id]
                light_dist = np.inf
                falloff = 1.0
            else:
                offset = data.light_xpos[light_id] - pnt
                light_dist = np.linalg.norm(offset)
                if light_dist == 0:
                    continue
                to_light = offset / light_dist
                falloff = light_dist ** 2

            cos_theta = np.dot(normal, to_light)
            if cos_theta <= 0:
                continue  # light is behind the sensor panel

            ray_origin = pnt
            remaining_dist = light_dist
            occluded = False
            for _ in range(_SELF_OCCLUSION_MAX_HITS):
                hit_dist = mujoco.mj_ray(model, data, ray_origin, to_light, _OCCLUSION_GEOMGROUP, True, -1, geomid)
                if hit_dist < 0 or hit_dist >= remaining_dist:
                    break  # nothing solid before the light - unoccluded
                if int(model.geom_bodyid[geomid[0]]) in self_bodies:
                    # Own module's geometry touching itself at the fold -
                    # not a real occluder; nudge past it and keep looking.
                    step = hit_dist + _SELF_OCCLUSION_RAY_EPS
                    ray_origin = ray_origin + to_light * step
                    remaining_dist -= step
                    continue
                occluded = True
                break
            if occluded:
                continue  # occluded before reaching the light

            lux += intensity * cos_theta / falloff

        readings[name] = lux

    # if verbose:
    #     print("Light sensor readings (lux):")
    #     for name, lux in readings.items():
    #         print(f"  {name}: {lux:.1f}")

    return readings


def get_com_position(data):
    """World-frame center of mass position of the whole model (m)."""
    return data.subtree_com[0].copy()


def get_com_velocity(data):
    """World-frame linear velocity of the whole model's center of mass (m/s).

    subtree_linvel is not populated by mj_step on its own - it requires an
    explicit mj_subtreeVel(model, data) call each step (unless a
    subtreelinvel sensor is defined in the XML).
    """
    return data.subtree_linvel[0].copy()


def dynamic_text_rendering(viewer, data, module_labels):
    """
    Dynamically renders text labels for each module body in the simulation.
    """
    # Reset custom user scene geoms at the start of each frame
    viewer.user_scn.ngeom = 0

    for body_id, label_text in module_labels.items():
        # Fetch position of the module body
        pos = data.xpos[body_id].copy()

        # Height offset: 0.0003 m (0.3 mm) so the label floats clearly above the module
        pos[2] += 0.0003

        # Fetch reference to current visual geom slot
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        # Call mjv_initGeom using positional arguments
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.array([0.005, 0.005, 0.005], dtype=np.float64),
            pos.astype(np.float64),
            np.eye(3).flatten().astype(np.float64),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # White label text
        )

        # Assign label string (displays "1", "2", "3", etc.)
        display_tag = label_text.split("_")[-1]
        geom.label = display_tag

        viewer.user_scn.ngeom += 1


def _detect_instability(window_velocities, cv_threshold=0.60):
    """Flags a "jumping/rolling" gait from its per-second windowed speed
    samples: any window of net backward COM motion, or a speed spread
    (stdev/mean) above `cv_threshold`, means it isn't walking steadily."""
    if len(window_velocities) < 2:
        return False
    arr = np.array(window_velocities, dtype=float)
    if np.any(arr < 0):
        return True
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return mean > 1e-9 and (std / mean) > cv_threshold


def save_simulation_stats(model, avg_velocity, total_distance, module_labels,
                           filename="simulation_stats.json", physics_ok=True, is_stable=True,
                           b_intensity=None, shape_entropy_2d=0.0, shape_entropy_3d=0.0,
                           pheromone_yaw_response_deg=0.0, pheromone_speed_response=0.0):
    """Calculates masses and average torque, then saves to a JSON file.

    physics_ok=False means a MuJoCo physics warning fired during the
    rollout, so every stat is written as zero instead of a partial value.
    is_stable=False means physics completed but the gait looked like
    jumping/rolling (see _detect_instability); velocity/distance are still
    real but not a meaningful "how well does this walk" signal. `success`
    (written to the JSON) is physics_ok AND is_stable. pheromone_yaw/speed
    response default to 0.0 and stay 0.0 when success is 0 (light tests
    aren't run for a failed/unstable rollout)."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not physics_ok:
        stats = {
            "success": 0,
            "physics_ok": 0,
            "is_stable": 0,
            "B_intensity_T": round(b_intensity, 6) if b_intensity is not None else 0,
            "total_mass_mg": 0,
            "average_torque_Nm": 0,
            "total_simulated_steps": 0,
            "average_velocity_mmps": 0,
            "total_distance_mm": 0,
            "shape_entropy_2d": round(shape_entropy_2d, 6),
            "shape_entropy_3d": round(shape_entropy_3d, 6),
            "pheromone_yaw_response_deg": 0,
            "pheromone_speed_response": 0,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        logger.info(
            "Simulation Results: \n(success=%d, physics_ok=%d, is_stable=%d, avg_velocity=%.2f mm/s, total_distance=%.2f mm)",
            stats["success"], stats["physics_ok"], stats["is_stable"], stats["average_velocity_mmps"], stats["total_distance_mm"],
        )
        return

    total_mass = float(mujoco.mj_getTotalmass(model))
    avg_torque = float(np.mean(torque_history)) if torque_history else 0.0

    stats = {
        "success": 1 if physics_ok and is_stable else 0,
        "physics_ok": 1 if physics_ok else 0,
        "is_stable": 1 if is_stable else 0,
        "B_intensity_T": round(b_intensity, 6) if b_intensity is not None else 0,
        "total_mass_mg": round(total_mass * 1e6, 4),  # Convert to milligrams
        "average_torque_Nm": round(avg_torque, 4),  # Convert to Newton-meters
        "total_simulated_steps": len(torque_history),
        "average_velocity_mmps": round(avg_velocity * 1000, 4),
        "total_distance_mm": round(total_distance * 1000, 4),
        "shape_entropy_2d": round(shape_entropy_2d, 6),
        "shape_entropy_3d": round(shape_entropy_3d, 6),
        "pheromone_yaw_response_deg": round(pheromone_yaw_response_deg, 4),
        "pheromone_speed_response": round(pheromone_speed_response, 6),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    logger.info(
        "Simulation Results: \n(success=%d, physics_ok=%d, is_stable=%d, avg_velocity=%.2f mm/s, total_distance=%.2f mm)",
         stats["success"], stats["physics_ok"], stats["is_stable"], stats["average_velocity_mmps"], stats["total_distance_mm"],
    )
    logger.debug(
        "Total mass=%.2f mg, average_torque=%.2e Nm, last_torque=%.2e Nm, steps=%d",
        total_mass * 1e6, avg_torque, torque_history[-1] if torque_history else 0.0, len(torque_history),
    )


def plot_b_field_history(filename="../output/b_field_plot.png"):
    """Plots the applied magnetic field B (z-component) over simulation time and saves it to file."""
    if not time_history:
        return

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(time_history[0:120], b_field_history[0:120], drawstyle="steps-post", color="#2a78d6", linewidth=2)

    ax.set_xlabel("Time (s)", color="#52514e")
    ax.set_ylabel("Applied B field, z-component (T)", color="#52514e")
    logger.info("Saved B-field plot to '%s'", filename)
    ax.set_title("Applied Magnetic Field Over Time", color="#0b0b0b")

    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    fig.tight_layout()
    fig.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Saved B-field plot to '%s'", filename)


def _offscreen_camera(distance=0.25, lookat=(0, 0, 0)):
    """MjvCamera matching the live viewer's default zoom/lookat, for
    screenshots/video captured via mujoco.Renderer (no GUI window needed)."""
    camera = mujoco.MjvCamera()
    camera.distance = distance
    camera.lookat[:] = lookat

    # Top view: looking straight down along -Z
    # camera.azimuth = 90
    camera.elevation = -90

    return camera


def run_headless_light_tests(
    model, target_angles, module1_qpos_adr,
    light_test_duration=LIGHT_TEST_DEFAULT_DURATION,
    light_target_angles=None,
):
    """Headless companion to run_light_tests(): computes
    pheromone_yaw_response_deg and pheromone_speed_response for
    run_headless()'s include_light_tests=True path.

    Runs a dedicated no-light baseline stage first, then defines
    "left"/"front" relative to that stage's own measured travel direction
    (light-patch bounds are expressed in a travel-aligned local frame; the
    robot's own qpos is never rotated). `model` is reused as-is; a fresh
    MjData starts a clean simulation state. light_target_angles forwards
    each joint's own evolved trigger angle into set_angle_to_joint().

    Returns {"pheromone_yaw_response_deg": ..., "pheromone_speed_response":
    ...} - yaw response is yaw(left stage) - yaw(baseline); speed response
    is (speed(front) - speed(baseline)) / speed(baseline), 0.0 if baseline
    speed is 0."""
    fallback = {"pheromone_yaw_response_deg": 0.0, "pheromone_speed_response": 0.0}

    data = mujoco.MjData(model)
    model.opt.timestep = 0.01

    if model.nlight == 0:
        logger.warning("Model has no <light> for run_headless_light_tests() to repoint - skipping.")
        return fallback
    light_id = 0

    def module1_yaw_deg():
        qw, qx, qy, qz = data.qpos[module1_qpos_adr + 3: module1_qpos_adr + 7]
        return float(np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))

    # ---- settle to the folded pose ----
    for _ in range(SETTLE_MAX_STEPS):
        set_angle_to_joint(model, data, target_angle_deg=target_angles)
        mujoco.mj_step(model, data)
        if int(np.sum(data.warning.number)) > 0:
            logger.warning("MuJoCo warning during run_headless_light_tests() settle - skipping light tests.")
            return fallback
        if np.max(np.abs(data.qvel)) < SETTLE_VELOCITY_THRESHOLD:
            break
    data.time = 0.0
    initial_com = get_com_position(data)
    initial_qpos = data.qpos.copy()

    def reset_to_initial_pose():
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0
        data.time = 0.0
        mujoco.mj_forward(model, data)

    def run_stage(duration, light_bounds, track_angular, frame=None):
        com_start = get_com_position(data)
        yaw_start = module1_yaw_deg() if track_angular else None
        mujoco.set_mjcb_control(magnetic_field_callback)
        while data.time < duration:
            set_angle_to_joint(model, data, target_angle_deg=target_angles, light_bounds=light_bounds,
                                light_target_angles=light_target_angles, frame=frame)
            mujoco.mj_step(model, data)
            mujoco.mj_subtreeVel(model, data)
            if int(np.sum(data.warning.number)) > 0:
                logger.warning("MuJoCo warning during run_headless_light_tests() stage - stopping it early.")
                break
        mujoco.set_mjcb_control(None)
        data.xfrc_applied.fill(0)
        elapsed = data.time if data.time > 0 else duration
        avg_linear_mps = float(np.linalg.norm(get_com_position(data) - com_start)) / elapsed if elapsed > 0 else 0.0
        result = {"avg_linear_mps": avg_linear_mps}
        if track_angular:
            result["yaw_rotation_deg"] = float(((module1_yaw_deg() - yaw_start + 180) % 360) - 180)
        return result

    # Dedicated no-light baseline stage (light_bounds=None disables the
    # reactive-fold branch regardless of the scene <light>'s own state).
    reset_to_initial_pose()
    baseline_result = run_stage(LIGHT_TEST_BASELINE_DURATION, light_bounds=None, track_angular=True)

    # Travel-aligned local frame from the baseline stage's own measured
    # direction (falls back to world +Y if too small to trust). The
    # robot's qpos is not rotated; only the light-patch bounds are.
    baseline_travel_xy = get_com_position(data)[:2] - initial_com[:2]
    travel_dist = float(np.linalg.norm(baseline_travel_xy))
    if travel_dist >= LIGHT_TEST_MIN_BASELINE_TRAVEL:
        forward = baseline_travel_xy / travel_dist
    else:
        logger.debug(
            "run_headless_light_tests(): baseline stage's travel distance too small (%.4f mm) - "
            "defaulting light-patch frame to world +Y.",
            travel_dist * 1000,
        )
        forward = np.array([0.0, 1.0])
    origin_xy = initial_com[:2]

    reset_to_initial_pose()

    xy_min, xy_max = assembly_local_bounds(model, data, origin_xy, forward)
    mid_x = (xy_min[0] + xy_max[0]) / 2.0
    mid_y = (xy_min[1] + xy_max[1]) / 2.0

    def _pheromone_light(center_local_xy):
        # MuJoCo only understands world coordinates, so the local-frame
        # center is converted via local_to_world_xy() for this light's
        # placement.
        center_xy = local_to_world_xy(center_local_xy, origin_xy, forward)
        _place_pheromone_light(model, light_id, center_xy, pheromone_light_z)
        # Forward kinematics must re-run so data.light_xpos reflects the new
        # model.light_pos before run_stage()'s first iteration reads it.
        mujoco.mj_forward(model, data)

    frame = (origin_xy, forward)
    pheromone_light_z = pheromone_light_height(model, data)

    # "left" stage
    reset_to_initial_pose()
    left_bounds = (xy_min[0], mid_x, xy_min[1], xy_max[1])
    _pheromone_light(((xy_min[0] + mid_x) / 2.0 - LIGHT_TEST_LEFT_EXTRA_OFFSET, mid_y))
    # Scaled the same way run_light_tests() scales its visible patch, so
    # the two tools' trigger/release boundary stays identical.
    left_sensed_bounds = _scale_local_bounds(left_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
    left_result = run_stage(light_test_duration, {light_id: left_sensed_bounds}, track_angular=True, frame=frame)

    # "front" stage: lit patch covers the whole body footprint (centered on
    # it), so both sensors are lit simultaneously from the start.
    reset_to_initial_pose()
    front_bounds = (xy_min[0], xy_max[0], xy_min[1], xy_max[1])
    _pheromone_light((mid_x, mid_y))
    front_sensed_bounds = _scale_local_bounds(front_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
    front_result = run_stage(light_test_duration, {light_id: front_sensed_bounds}, track_angular=False, frame=frame)

    mujoco.set_mjcb_control(None)

    baseline_avg_velocity = baseline_result["avg_linear_mps"]
    # Re-wrap the difference of two already-wrapped angles into (-180, 180].
    pheromone_yaw_response_deg = (
        (left_result["yaw_rotation_deg"] - baseline_result["yaw_rotation_deg"] + 180) % 360
    ) - 180
    if baseline_avg_velocity:
        pheromone_speed_response = (
            (front_result["avg_linear_mps"] - baseline_avg_velocity) / baseline_avg_velocity
        )
    else:
        pheromone_speed_response = 0.0

    return {
        "pheromone_yaw_response_deg": float(pheromone_yaw_response_deg),
        "pheromone_speed_response": float(pheromone_speed_response),
    }


def run_headless(
    model_path, stats_output_path, max_sim_time=None,
    capture_img=False, capture_gif=False, media_dir="../output", gif_fps=15,
    model=None, target_angles=None, light_target_angles=None,
    include_light_tests=False, compute_shape_entropy_3d=False,
    light_module_rgba=None,
):
    """Runs one closed-loop simulation to completion with no viewer and no
    real-time pacing. Designed to be called once per OS process (via
    parallel_main) so N models can run concurrently on N cores; clears
    module-level tracking state at the start for worker-process reuse.

    If capture_img/capture_gif is True, saves a final-pose screenshot and/or
    a GIF via an offscreen mujoco.Renderer. `model`/`target_angles` let a
    caller running the same model_path multiple times (run_headless_b_sweep)
    compile and parse target angles once instead of per-call. A fresh MjData
    is always created so per-run state never leaks across calls.

    include_light_tests=True additionally runs run_headless_light_tests()
    after this run finishes; left False by default so run_headless_b_sweep's
    thrown-away per-candidate-B trials don't pay for it, and skipped
    entirely when this run's own success is 0.

    compute_shape_entropy_3d=True runs the settle-to-folded-pose pass and
    computes shape_entropy_3d even when capture_img is False (otherwise it
    stays at its 0.0 default). Same False-by-default reasoning as
    include_light_tests.

    light_module_rgba (default None) recolors the light-sensitive module for
    spotting it in a screenshot/GIF - purely cosmetic, no physics effect."""
    parent_body_magnet_map.clear()
    torque_history.clear()
    b_field_history.clear()
    time_history.clear()
    _light_joint_state.clear()
    _module_self_body_cache.clear()

    if model is None:
        model = mujoco.MjModel.from_xml_path(model_path)
    if light_module_rgba is not None:
        _recolor_light_sensitive_module(model, light_module_rgba)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.01

    if target_angles is None:
        target_angles = read_joint_target_angles_from_xml(model_path)
    if not target_angles:
        target_angles = [45.0] * model.nu
    if light_target_angles is None:
        light_target_angles = read_light_target_angles_from_xml(model_path)

    parent_body_magnet_map.update(find_all_magnets(model))
    _prepare_magnet_arrays()
    module_labels = find_module_labels(model)

    # module_1 is this design's designated sensor/control module - forwarded
    # to run_headless_light_tests() as the reference body for yaw tracking.
    module1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_module_1")
    module1_qpos_adr = model.jnt_qposadr[module1_joint_id] if module1_joint_id >= 0 else None

    # 2D shape entropy from the model's flat/unfolded layout - a single
    # forward-kinematics pass at the default qpos already gives correct
    # flat positions, no settling needed.
    mujoco.mj_forward(model, data)
    flat_positions = _module_positions(data, module_labels)
    entropy_cell_size = entropy_api.adaptive_cell_size(flat_positions)
    shape_entropy_2d = entropy_api.multiscale_shape_entropy(
        [pos[:2] for pos in flat_positions.values()], dims=2,
        window_sizes=SHAPE_ENTROPY_WINDOW_SIZES, cell_size=entropy_cell_size,
    )
    shape_entropy_3d = 0.0  # filled in below once the folded pose is settled

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    renderer = None
    camera = None
    gif_frames = []
    # GIF frames are captured every gif_frame_stride steps so playback speed
    # approximates real time: gif_fps output frames per second of sim time.
    gif_frame_stride = max(1, round(1.0 / (gif_fps * model.opt.timestep)))
    if capture_img or capture_gif:
        try:
            renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
            camera = _offscreen_camera()
        except Exception:
            # A renderer/GPU failure should cost only the screenshot, not
            # the rest of this run's physics result.
            logger.exception(
                "Could not create offscreen renderer for '%s' - continuing without screenshot/GIF capture.",
                model_name,
            )
            renderer = None

    avg_velocity = 0.0
    initial_com = None
    displacement = 0.0
    steady_state_displacement = None
    magnets_active = True
    physics_ok = True
    step_count = 0
    # Raycasting the rangefinders every 0.01s step is wasted work at mm/s
    # speeds; every 10 steps (0.1s) cuts the cost 10x with no real loss.
    WALL_CHECK_STRIDE = 10

    # Per-second windowed speed samples (post torque-ramp), for
    # _detect_instability().
    last_window_second = -1
    window_velocity_history = []
    window_displacement = None
    window_time = None

    try:
        # Settle to the folded pose, used for both the screenshot and 3D
        # shape entropy below - gated on either flag needing it.
        if capture_img or compute_shape_entropy_3d:
            for _ in range(SETTLE_MAX_STEPS):
                set_angle_to_joint(model, data, target_angle_deg=target_angles)
                get_light_sensor_values(model, data)
                mujoco.mj_step(model, data)
                mujoco.mj_subtreeVel(model, data)

                if int(np.sum(data.warning.number)) > 0:
                    physics_ok = False
                    magnets_active = False
                    break
                if np.max(np.abs(data.qvel)) < SETTLE_VELOCITY_THRESHOLD:
                    break

            # Reset simulated time so the torque ramp / max_sim_time cutoff /
            # displacement tracking below all start fresh.
            data.time = 0.0

            # 3D shape entropy from this now-settled folded pose - needs the
            # real stepped-and-settled `data`, not just forward kinematics,
            # since modules are only rigidly linked via weld constraints.
            if physics_ok:
                folded_positions = _module_positions(data, module_labels)
                shape_entropy_3d = entropy_api.multiscale_shape_entropy(
                    list(folded_positions.values()), dims=3,
                    window_sizes=SHAPE_ENTROPY_WINDOW_SIZES, cell_size=entropy_cell_size,
                )

            if capture_img and physics_ok and renderer is not None:
                try:
                    os.makedirs(media_dir, exist_ok=True)
                    renderer.update_scene(data, camera=camera)
                    screenshot = renderer.render()
                    if screenshot is not None:
                        Image.fromarray(screenshot).save(
                            os.path.join(media_dir, f"screenshot_{model_name}.png")
                        )
                    else:
                        logger.warning("Renderer returned no image for '%s' - no screenshot saved.", model_name)
                except Exception:
                    # A failed screenshot save shouldn't cost this run its
                    # otherwise-valid physics results.
                    logger.exception("Failed to render/save screenshot for '%s' - continuing without it.", model_name)

        mujoco.set_mjcb_control(magnetic_field_callback)

        while magnets_active:
            set_angle_to_joint(model, data, target_angle_deg=target_angles)
            get_light_sensor_values(model, data)
            mujoco.mj_step(model, data)
            mujoco.mj_subtreeVel(model, data)
            step_count += 1

            # A MuJoCo warning means the physics is no longer trustworthy -
            # stop immediately rather than spin through garbage state.
            if int(np.sum(data.warning.number)) > 0:
                physics_ok = False
                magnets_active = False
                break

            com = get_com_position(data)
            if initial_com is None:
                initial_com = com.copy()
            displacement = float(np.linalg.norm(com - initial_com))

            if steady_state_displacement is None and data.time >= TORQUE_RAMP_TIME:
                steady_state_displacement = displacement
                window_displacement = displacement
                window_time = data.time
            if steady_state_displacement is not None:
                steady_elapsed = data.time - TORQUE_RAMP_TIME
                avg_velocity = (
                    (displacement - steady_state_displacement) / steady_elapsed
                    if steady_elapsed > 0 else 0.0
                )

                current_second = int(data.time)
                if current_second != last_window_second:
                    last_window_second = current_second
                    interval = data.time - window_time
                    if interval > 0:
                        window_velocity_history.append((displacement - window_displacement) / interval)
                        window_displacement = displacement
                        window_time = data.time

            if (
                capture_gif and renderer is not None and data.time >= TORQUE_RAMP_TIME
                and step_count % gif_frame_stride == 0
            ):
                renderer.update_scene(data, camera=camera)
                gif_frames.append(renderer.render().copy())

            if step_count % WALL_CHECK_STRIDE == 0:
                wall_dist = distance_to_nearest_wall(model, data)
                if wall_dist is not None:
                    wall_dist *= 1000  # Convert to mm
                if wall_dist is not None and wall_dist < WALL_STOP_DISTANCE:
                    magnets_active = False
            if max_sim_time is not None and data.time >= max_sim_time:
                magnets_active = False

        # Save GIF if requested and frames were captured
        if capture_gif and gif_frames:
            os.makedirs(media_dir, exist_ok=True)
            frame_duration_ms = round(1000 / gif_fps)
            frames = [Image.fromarray(f) for f in gif_frames]
            frames[0].save(
                os.path.join(media_dir, f"movement_{model_name}.gif"),
                save_all=True, append_images=frames[1:],
                duration=frame_duration_ms, loop=0,
            )

    finally:
        if renderer is not None:
            renderer.close()

    mujoco.set_mjcb_control(None)
    data.xfrc_applied.fill(0)

    is_stable = not _detect_instability(window_velocity_history) if physics_ok else False
    success = physics_ok and is_stable

    # pheromone_yaw_response_deg / pheromone_speed_response: only meaningful
    # relative to a real "no light" baseline, which a failed/unstable run
    # doesn't have.
    pheromone_yaw_response_deg = 0.0
    pheromone_speed_response = 0.0
    if include_light_tests and success:
        light_test_results = run_headless_light_tests(
            model, target_angles, module1_qpos_adr,
            light_target_angles=light_target_angles,
        )
        pheromone_yaw_response_deg = light_test_results["pheromone_yaw_response_deg"]
        pheromone_speed_response = light_test_results["pheromone_speed_response"]

    save_simulation_stats(model, avg_velocity, displacement, module_labels,
                           filename=stats_output_path, physics_ok=physics_ok, is_stable=is_stable,
                           b_intensity=B_INTENSITY, shape_entropy_2d=shape_entropy_2d,
                           shape_entropy_3d=shape_entropy_3d,
                           pheromone_yaw_response_deg=pheromone_yaw_response_deg,
                           pheromone_speed_response=pheromone_speed_response)

    return {
        "model": model_name,
        "physics_ok": physics_ok,
        "success": success,
        "is_stable": is_stable,
        "B_intensity_T": B_INTENSITY,
        "avg_velocity_mmps": avg_velocity * 1000 if success else 0.0,
        "displacement_mm": displacement * 1000 if success else 0.0,
        "sim_time_s": data.time,
        "pheromone_yaw_response_deg": pheromone_yaw_response_deg,
        "pheromone_speed_response": pheromone_speed_response,
    }


def run_headless_b_sweep(
    model_path, stats_output_path, b_values=B_SWEEP_VALUES, max_sim_time=None,
    capture_img=False, capture_gif=False, media_dir="../output", gif_fps=15,
    include_light_tests=True, light_module_rgba=None,
):
    """Runs one fast (no media) run_headless() rollout per candidate B in
    b_values, picks the one with the highest average velocity, then
    re-runs just that winning B for real (with the caller's actual
    capture_img/capture_gif and always compute_shape_entropy_3d=True) so
    media/entropy is never spent on a discarded candidate.

    Winner selection: prefer the highest avg_velocity_mmps among
    physics_ok=True, is_stable=True runs; fall back to physics_ok=True
    regardless of is_stable; if every B failed, report the last attempted
    B's failed (all-zero) result.

    Mutates the module-level B_INTENSITY for the duration of the sweep;
    always restored afterward, even on error. include_light_tests (default
    True here) is forwarded only to the final winner re-run, at the actual
    winning field strength. light_module_rgba is applied once, up front,
    since every candidate shares the same compiled `model` object."""
    global B_INTENSITY
    original_b = B_INTENSITY

    # Only B_INTENSITY differs between candidates, so compile the model
    # once and reuse it across every run_headless call.
    model = mujoco.MjModel.from_xml_path(model_path)
    if light_module_rgba is not None:
        _recolor_light_sensitive_module(model, light_module_rgba)
    target_angles = read_joint_target_angles_from_xml(model_path)
    light_target_angles = read_light_target_angles_from_xml(model_path)

    results = []  # (result_dict, b_value)
    tmp_paths = []
    try:
        for b in b_values:
            B_INTENSITY = b
            fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="b_sweep_")
            os.close(fd)
            tmp_paths.append(tmp_path)
            result = run_headless(
                model_path, tmp_path, max_sim_time=max_sim_time,
                capture_img=False, capture_gif=False, media_dir=media_dir, gif_fps=gif_fps,
                model=model, target_angles=target_angles, light_target_angles=light_target_angles,
            )
            results.append((result, b))
            logger.debug(
                "[B sweep] B=%s T -> physics_ok=%d is_stable=%d velocity=%.2f mm/s",
                b, int(result['physics_ok']), int(result['is_stable']), result['avg_velocity_mmps'],
            )
    finally:
        B_INTENSITY = original_b
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    # Prefer runs where the physics completed and the gait was stable.
    pool = [(r, b) for r, b in results if r["physics_ok"] and r["is_stable"]]
    if pool:
        winner_result, winner_b = max(pool, key=lambda rb: rb[0]["avg_velocity_mmps"])
    else:
        winner_b = results[-1][1] if results else 0
        last_model = results[-1][0].get("model", os.path.basename(model_path)) if results else os.path.basename(model_path)
        winner_result = {
            "model": last_model,
            "success": 0,
            "physics_ok": 0,
            "is_stable": 0,
            "B_intensity_T": round(winner_b, 6),
            "avg_velocity_mmps": 0,
            "displacement_mm": 0,
            "sim_time_s": 0.0,
            "pheromone_yaw_response_deg": 0.0,
            "pheromone_speed_response": 0.0,
        }

    logger.info(
        "[B sweep] picked B=%.6f T (velocity=%.2f mm/s, physics_ok=%d, is_stable=%d)",
        winner_b, winner_result['avg_velocity_mmps'], int(winner_result['physics_ok']), int(winner_result['is_stable']),
    )

    # Re-run the winner for real, at the caller's requested stats path and
    # media flags, only if the physics completed successfully and the gait
    # wasn't flagged unstable.
    if winner_result["physics_ok"] and winner_result["is_stable"]:
        B_INTENSITY = winner_b
        try:
            final_result = run_headless(
                model_path, stats_output_path, max_sim_time=max_sim_time,
                capture_img=capture_img, capture_gif=capture_gif,
                media_dir=media_dir, gif_fps=gif_fps,
                model=model, target_angles=target_angles, light_target_angles=light_target_angles,
                include_light_tests=include_light_tests, compute_shape_entropy_3d=True,
            )
        finally:
            B_INTENSITY = original_b
    else:
        # No successful B found, write a zeroed stats file to the caller's
        # requested path. pheromone_yaw_response_deg/pheromone_speed_response
        # default to 0.0 in save_simulation_stats() -- no light tests run here either.
        save_simulation_stats(
            model=None, avg_velocity=0.0, total_distance=0.0, module_labels={},
            filename=stats_output_path, physics_ok=False, is_stable=False,
            b_intensity=winner_b,
        )
        final_result = winner_result

    return final_result


def run_with_viewer(model_path, stats_output_path, max_sim_time=None, light_module_rgba=None):
    """Same closed-loop simulation as run_headless, but with the live passive
    viewer and real-time pacing so you can watch it run.

    max_sim_time is in seconds of simulated time (data.time), same as
    run_headless - once reached, magnet actuation stops (same as reaching a
    wall) but the viewer stays open so you can still inspect the final pose.

    light_module_rgba (default None = leave the normal body-blue color
    alone): see run_headless()'s docstring.
    """
    if not os.path.exists(model_path):
        logger.error("Could not find '%s'", model_path)
        return
    logger.info("Loading model: %s...", model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    if light_module_rgba is not None:
        _recolor_light_sensitive_module(model, light_module_rgba)
    data = mujoco.MjData(model)

    target_angles = read_joint_target_angles_from_xml(model_path)
    if not target_angles:
        target_angles = [45.0] * model.nu

    if isinstance(target_angles, dict):
        logger.debug("Loaded actuator target angles from XML:")
        for actuator_name, angle_deg in sorted(target_angles.items()):
            if actuator_name.startswith("ctrl_joint"):
                logger.debug(" - %s: %.2f°", actuator_name, angle_deg)
    else:
        logger.debug("Loaded actuator target angles from XML as ordered values:")
        for i, angle_deg in enumerate(target_angles):
            logger.debug(" - actuator %d: %.2f°", i, angle_deg)

    # Simulation timestep
    model.opt.timestep = 0.01  # 10 milliseconds

    # Find all magnets
    parent_body_magnet_map.clear()
    parent_body_magnet_map.update(find_all_magnets(model))
    _prepare_magnet_arrays()
    _light_joint_state.clear()
    _module_self_body_cache.clear()

    # Identify module bodies and assign their text labels
    module_labels = find_module_labels(model)

    # Register callback
    mujoco.set_mjcb_control(magnetic_field_callback)

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.3  # zoom
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.elevation = -90
        last_print = -1
        avg_velocity = 0.0
        initial_com = None
        displacement = 0.0
        # Displacement captured the instant the torque ramp finishes, so the
        # ramp's slow-moving first TORQUE_RAMP_TIME seconds can be excluded
        # from the average velocity below.
        steady_state_displacement = None
        magnets_active = True
        physics_ok = True
        # Windowed velocity: displacement since the last 1-second print,
        # reflecting actual per-interval speed rather than the cumulative
        # average-since-ramp-end.
        window_velocity = 0.0
        window_displacement = None
        window_time = None
        # Same per-second samples as run_headless, for _detect_instability().
        window_velocity_history = []

        while viewer.is_running():
            step_start = time.time()
            set_angle_to_joint(model, data, target_angle_deg=target_angles)

            mujoco.mj_step(model, data)
            mujoco.mj_subtreeVel(model, data)

            # A MuJoCo warning means the physics is no longer trustworthy -
            # stop feeding the viewer immediately.
            if int(np.sum(data.warning.number)) > 0:
                physics_ok = False
                mujoco.set_mjcb_control(None)
                data.xfrc_applied.fill(0)
                viewer.close()
                break

            # Gated on magnets_active so these freeze at the wall-stop
            # instant instead of drifting during post-locomotion idle time.
            if magnets_active:
                com = get_com_position(data)
                if initial_com is None:
                    initial_com = com.copy()

                displacement = float(np.linalg.norm(com - initial_com)) #Euclidean distance

                # Average velocity is calculated from the displacement after the torque ramp finishes.
                if steady_state_displacement is None and data.time >= TORQUE_RAMP_TIME:
                    steady_state_displacement = displacement
                    window_displacement = displacement
                    window_time = data.time
                if steady_state_displacement is not None:
                    steady_elapsed = data.time - TORQUE_RAMP_TIME
                    avg_velocity = (
                        (displacement - steady_state_displacement) / steady_elapsed
                        if steady_elapsed > 0 else 0.0
                    )

            # Stop actuating once close to a wall, judged from the on-body
            # rangefinders - mirrors a real obstacle-avoidance cutoff.
            if magnets_active:
                wall_dist = distance_to_nearest_wall(model, data)
                if wall_dist is not None:
                    wall_dist *= 1000  # Convert to mm
                if wall_dist is not None and wall_dist < WALL_STOP_DISTANCE:
                    mujoco.set_mjcb_control(None)
                    # xfrc_applied persists across steps; unregistering the
                    # callback alone wouldn't clear the last applied torque.
                    data.xfrc_applied.fill(0)
                    magnets_active = False
                    logger.info("Time: %.2fs | Wall reached (%d mm) - magnetic field stopped.", data.time, int(wall_dist))

            if max_sim_time is not None and data.time >= max_sim_time:
                logger.info("Time: %.2fs | max_sim_time reached - closing viewer.", data.time)
                viewer.close()
                break

            # Render dynamic text labels for each module
            #dynamic_text_rendering(viewer, data, module_labels)

            viewer.sync()

            # Print torque ramp progress and body velocity every second
            current_second = int(data.time)
            if current_second != last_print and magnets_active :
                last_print = current_second
                get_light_sensor_values(model, data)
                ramp = min(data.time / TORQUE_RAMP_TIME, 1.0)
                if ramp < 1.0:
                    logger.info("Time: %.2fs | Torque ramp: %d%%", data.time, int(ramp * 100))
                else:
                    interval = data.time - window_time
                    window_velocity = (displacement - window_displacement) / interval if interval > 0 else 0.0
                    window_displacement = displacement
                    window_time = data.time
                    window_velocity_history.append(window_velocity)
                    logger.info(
                        "Velocity: %.2f mm/s | Avg velocity: %.2f mm/s | Displacement: %.2f mm",
                        window_velocity * 1000, avg_velocity * 1000, displacement * 1000,
                    )

            # Maintain real-time velocity
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Unregister callback and save JSON data on exit
    mujoco.set_mjcb_control(None)
    is_stable = not _detect_instability(window_velocity_history) if physics_ok else False
    save_simulation_stats(model, avg_velocity, displacement, module_labels,
                           filename=stats_output_path, physics_ok=physics_ok, is_stable=is_stable,
                           b_intensity=B_INTENSITY)
    #plot_b_field_history(filename="../output/b_field_plot.png")


def run_light_tests(model_path, stats_output_path, light_test_duration=LIGHT_TEST_DEFAULT_DURATION,
                     capture_video=False, media_dir="../output", video_fps=30, light_module_rgba=None):
    """Light-response test: same live-viewer setup as run_with_viewer, but
    runs a fixed sequence instead of one continuous gait: settle to folded
    pose -> baseline (no light) -> stage "left" -> stage "front".

    Stages "left"/"front" use the same floor-level "luminance sheet"
    fixture (the model's <light> repositioned via _place_pheromone_light(),
    plus a visible emissive patch via _show_luminance_sheet()); only the
    rectangle each stage covers (assembly_local_bounds()) differs - "left"
    covers the left half, "front" covers the whole footprint so both
    sensors light up simultaneously.

    Metrics per stage: avg linear velocity, and (baseline/left only) net
    yaw rotation of module_1's freejoint. The final printed summary reports
    pheromone_yaw_response_deg/pheromone_speed_response the same way as
    run_headless_light_tests(), against this run's own baseline stage.

    capture_video=True additionally records every stage through a second,
    independent offscreen mujoco.Renderer, streamed to
    f"{media_dir}/light_tests_{model_name}.mp4" via an imageio ffmpeg
    writer rather than buffered in memory. A renderer/writer failure only
    costs the video, not the physics/summary. light_module_rgba: see
    run_headless()'s docstring."""
    if not os.path.exists(model_path):
        logger.error("Could not find '%s'", model_path)
        return
    logger.info("Loading model: %s...", model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    if light_module_rgba is not None:
        _recolor_light_sensitive_module(model, light_module_rgba)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.01

    if model.nlight == 0:
        logger.error("Model has no <light> for run_light_tests() to repoint.")
        return
    light_id = 0
    sheet_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "luminance_sheet")

    # module_1 is this design's designated sensor/control module, used here
    # as the reference body for yaw tracking.
    module1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_module_1")
    if module1_joint_id < 0:
        logger.warning("No 'free_module_1' joint found - yaw tracking will be skipped.")
        module1_qpos_adr = None
    else:
        module1_qpos_adr = model.jnt_qposadr[module1_joint_id]

    def module1_yaw_deg():
        qw, qx, qy, qz = data.qpos[module1_qpos_adr + 3: module1_qpos_adr + 7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(np.degrees(yaw))

    target_angles = read_joint_target_angles_from_xml(model_path)
    if not target_angles:
        target_angles = [45.0] * model.nu
    light_target_angles = read_light_target_angles_from_xml(model_path)

    parent_body_magnet_map.clear()
    parent_body_magnet_map.update(find_all_magnets(model))
    _prepare_magnet_arrays()
    _light_joint_state.clear()
    _module_self_body_cache.clear()

    dt = model.opt.timestep

    # Offscreen video capture, independent of the live viewer window. A
    # renderer/writer construction failure only costs the video.
    video_renderer = None
    video_writer = None
    video_frame_stride = max(1, round(1.0 / (video_fps * dt)))
    video_step_count = 0
    if capture_video:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        try:
            video_renderer = mujoco.Renderer(model, height=VIDEO_RENDER_HEIGHT, width=VIDEO_RENDER_WIDTH)
            # distance=0.3 matches the live viewer's own zoom (viewer.cam.distance below).
            video_camera = _offscreen_camera(distance=0.3)
            os.makedirs(media_dir, exist_ok=True)
            video_path = os.path.join(media_dir, f"light_tests_{model_name}.mp4")
            video_writer = imageio.get_writer(
                video_path, fps=video_fps, codec="libx264", quality=10,
                pixelformat="yuv420p", macro_block_size=1,
            )
        except Exception:
            logger.exception(
                "Could not create offscreen video renderer/writer for '%s' - "
                "continuing without video capture.", model_path,
            )
            if video_renderer is not None:
                video_renderer.close()
            video_renderer = None
            video_writer = None

    def capture_video_frame():
        nonlocal video_step_count
        video_step_count += 1
        if video_renderer is None or video_writer is None:
            return
        if video_step_count % video_frame_stride != 0:
            return
        video_renderer.update_scene(data, camera=video_camera)
        video_writer.append_data(video_renderer.render())

    with ExitStack() as video_stack:
        # ExitStack cleans these up on every exit path, including the early
        # returns in the settle loop below, without wrapping the whole
        # function in try/finally.
        if video_writer is not None:
            video_stack.callback(video_writer.close)
        if video_renderer is not None:
            video_stack.callback(video_renderer.close)
        viewer = video_stack.enter_context(mujoco.viewer.launch_passive(model, data))
        viewer.cam.distance = 0.3
        viewer.cam.lookat[:] = [0, 0, 0]

        # ---- set hinge angle: settle to the folded pose before anything else ----
        for _ in range(SETTLE_MAX_STEPS):
            if not viewer.is_running():
                return
            step_start = time.time()
            set_angle_to_joint(model, data, target_angle_deg=target_angles)
            mujoco.mj_step(model, data)
            viewer.sync()
            if int(np.sum(data.warning.number)) > 0:
                logger.error("MuJoCo warning during hinge settle - aborting light tests.")
                return
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            if np.max(np.abs(data.qvel)) < SETTLE_VELOCITY_THRESHOLD:
                break
        data.time = 0.0

        # ---- get initial com of the assembly ----
        initial_com = get_com_position(data)
        initial_qpos = data.qpos.copy()
        logger.info("Initial COM: %s", initial_com)

        # Luminance sheet light height, computed once from this settled
        # pose, same for every stage.
        pheromone_light_z = pheromone_light_height(model, data)
        logger.info("Pheromone light height (world Z): %.4f mm.", pheromone_light_z * 1000)

        def reset_to_initial_pose():
            data.qpos[:] = initial_qpos
            data.qvel[:] = 0
            data.time = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()

        def run_stage(stage_name, duration, light_bounds=None, track_angular=False, track_acceleration=False,
                      v_start_override=None, frame=None):
            """Runs `duration` seconds of magnetic actuation (light_bounds=
            None means no reactive fold). Returns avg linear velocity; with
            track_angular=True and a resolved module_1 joint, also net yaw
            rotation. track_acceleration=True additionally returns
            avg_accel_y_mps2 = (v_end_y - v_start_y) / elapsed; pass
            v_start_override to use a caller-supplied v_start_y instead of
            the just-reset (zero-velocity) state."""
            logger.info("Light test stage '%s': %.1fs of magnetic actuation...", stage_name, duration)
            last_print = -1
            com_start = get_com_position(data)
            track_angular = track_angular and module1_qpos_adr is not None
            yaw_start = module1_yaw_deg() if track_angular else None
            if track_acceleration:
                if v_start_override is not None:
                    v_start_y = float(v_start_override)
                else:
                    mujoco.mj_subtreeVel(model, data)
                    v_start_y = float(get_com_velocity(data)[1])

            mujoco.set_mjcb_control(magnetic_field_callback)
            while viewer.is_running() and data.time < duration:
                step_start = time.time()
                set_angle_to_joint(model, data, target_angle_deg=target_angles, light_bounds=light_bounds,
                                    light_target_angles=light_target_angles, frame=frame)
                mujoco.mj_step(model, data)
                mujoco.mj_subtreeVel(model, data)

                if int(np.sum(data.warning.number)) > 0:
                    logger.error("MuJoCo warning during '%s' stage - stopping it early.", stage_name)
                    break

                current_second = int(data.time)
                if current_second != last_print:
                    last_print = current_second
                    if light_bounds is not None:
                        get_light_sensor_values(model, data, light_bounds=light_bounds, frame=frame)

                capture_video_frame()

                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            mujoco.set_mjcb_control(None)
            data.xfrc_applied.fill(0)

            elapsed = data.time if data.time > 0 else duration
            displacement_xy = get_com_position(data)[:2] - com_start[:2]
            displacement = float(np.linalg.norm(get_com_position(data) - com_start))
            avg_linear_mps = displacement / elapsed if elapsed > 0 else 0.0
            logger.info("Light test stage '%s' done. Avg linear velocity: %.3f mm/s",
                        stage_name, avg_linear_mps * 1000)
            # Net XY displacement vector, not just speed magnitude, so the
            # caller can check whether the robot moved toward or away from
            # the light, independent of module_1's own yaw rotation.
            result = {"avg_linear_mps": avg_linear_mps, "displacement_xy": displacement_xy}

            if track_angular:
                yaw_diff = ((module1_yaw_deg() - yaw_start + 180) % 360) - 180
                result["yaw_rotation_deg"] = float(yaw_diff)
                logger.info("Light test stage '%s': net yaw rotation %.2f deg",
                            stage_name, result["yaw_rotation_deg"])

            if track_acceleration:
                v_end_y = float(get_com_velocity(data)[1])
                result["v_start_y_mps"] = v_start_y
                result["v_end_y_mps"] = v_end_y
                result["avg_accel_y_mps2"] = (v_end_y - v_start_y) / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Light test stage '%s': v_start_y %.3f mm/s, v_end_y %.3f mm/s, avg accel_y %.3f mm/s^2",
                    stage_name, v_start_y * 1000, v_end_y * 1000, result["avg_accel_y_mps2"] * 1000,
                )

            return result

        results = {}

        # Baseline: move with the scene's normal (directional, non-spot)
        # light on. Its avg_linear_mps becomes the front stage's
        # v_start_override, so that stage's acceleration is measured from
        # actual cruising speed rather than an artificial at-rest 0.
        model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        model.light_active[light_id] = 1
        reset_to_initial_pose()
        results["baseline"] = run_stage("baseline (no light)", LIGHT_TEST_BASELINE_DURATION, track_angular=True)

        # Travel-aligned local frame from the baseline stage's own measured
        # direction (falls back to world +Y if too small to trust). The
        # robot's qpos is never rotated; only the light-patch bounds below
        # are expressed in this frame.
        baseline_end_com = get_com_position(data)
        travel_xy = baseline_end_com[:2] - initial_com[:2]
        travel_dist = float(np.linalg.norm(travel_xy))
        if travel_dist < LIGHT_TEST_MIN_BASELINE_TRAVEL:
            logger.warning(
                "Baseline moved only %.4f mm (< %.4f mm) - direction of travel isn't "
                "well-defined, defaulting light-patch frame to world +Y.",
                travel_dist * 1000, LIGHT_TEST_MIN_BASELINE_TRAVEL * 1000,
            )
            forward = np.array([0.0, 1.0])
        else:
            forward = travel_xy / travel_dist

            # "Head" = whichever module ends up furthest along the baseline
            # direction of travel, not a fixed module_1 -- purely for
            # visibility in the log.
            head_name, head_proj = None, -np.inf
            for body_id, name in find_module_labels(model).items():
                proj = float(np.dot(data.xpos[body_id][:2] - initial_com[:2], forward))
                if proj > head_proj:
                    head_name, head_proj = name, proj
            logger.info("Baseline travel dist %.3f mm, head module '%s'.", travel_dist * 1000, head_name)

        origin_xy = initial_com[:2]
        frame = (origin_xy, forward)

        # Purely cosmetic: point the viewer camera so the robot appears to
        # walk toward it - only moves the camera, never the robot's pose.
        viewer.cam.lookat[:2] = origin_xy
        viewer.cam.azimuth = float(np.degrees(np.arctan2(-forward[1], -forward[0])))

        reset_to_initial_pose()

        # Stage bounds all come from the same settled-pose bounding box;
        # only which half/edge each stage targets differs.
        xy_min, xy_max = assembly_local_bounds(model, data, origin_xy, forward)
        mid_x = (xy_min[0] + xy_max[0]) / 2.0
        mid_y = (xy_min[1] + xy_max[1]) / 2.0

        def _pheromone_sheet(center_local_xy, bounds, anchor_local_xy):
            center_xy = local_to_world_xy(center_local_xy, origin_xy, forward)
            _place_pheromone_light(model, light_id, center_xy, pheromone_light_z)
            _show_luminance_sheet(model, sheet_geom_id, bounds, anchor_local_xy, origin_xy, forward)
            # Forward kinematics must re-run so data.light_xpos reflects the
            # new model.light_pos before run_stage()'s first iteration.
            mujoco.mj_forward(model, data)
            return center_xy

        # Stage 1: luminance sheet covering only the left half of the body.
        reset_to_initial_pose()
        left_bounds = (xy_min[0], mid_x, xy_min[1], xy_max[1])  # left half in X, full depth in Y
        left_center = ((xy_min[0] + mid_x) / 2.0 - LIGHT_TEST_LEFT_EXTRA_OFFSET, mid_y)
        # Pin the inner edge (mid_x) so a bigger scale only grows the patch
        # further left, never toward the robot's center.
        left_light_xy = _pheromone_sheet(left_center, left_bounds, (mid_x, mid_y))
        # Trigger/release check uses the same scaled-up rectangle
        # _show_luminance_sheet() rendered, not the smaller unscaled bounds.
        left_sensed_bounds = _scale_local_bounds(left_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
        results["left"] = run_stage("left", light_test_duration, light_bounds={light_id: left_sensed_bounds},
                                     track_angular=True, frame=frame)

        # Stage 2: same luminance sheet, covering the whole body footprint
        # (centered on it) instead of a patch ahead of it - both sensors
        # light up simultaneously from the start.
        reset_to_initial_pose()
        front_bounds = (xy_min[0], xy_max[0], xy_min[1], xy_max[1])  # the whole body footprint
        front_center = (mid_x, mid_y)
        front_light_xy = _pheromone_sheet(front_center, front_bounds, (mid_x, mid_y))
        front_sensed_bounds = _scale_local_bounds(front_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
        results["front"] = run_stage("front", light_test_duration, light_bounds={light_id: front_sensed_bounds},
                                      track_acceleration=True,
                                      v_start_override=results["baseline"]["avg_linear_mps"], frame=frame)

    mujoco.set_mjcb_control(None)

    # Computed the same way as objectives_api.py's f6/f7, but against this
    # run's own freshly-measured no-light baseline stage.
    baseline, left, front = results["baseline"], results["left"], results["front"]
    baseline_yaw = baseline.get("yaw_rotation_deg", 0.0)
    # Re-wrap the difference of two already-wrapped angles into (-180, 180].
    pheromone_yaw_response_deg = ((left["yaw_rotation_deg"] - baseline_yaw + 180) % 360) - 180
    baseline_v = baseline["avg_linear_mps"]
    pheromone_speed_response = (
        (front["avg_linear_mps"] - baseline_v) / baseline_v if baseline_v else 0.0
    )

    # Sign conventions match objectives_api.py's f6/f7: positive yaw
    # response = turned TOWARD the stimulus, negative = away; positive
    # speed response = sped up, negative = slowed down.
    if pheromone_yaw_response_deg > 0:
        turn_desc = "rotated TOWARD the light"
    elif pheromone_yaw_response_deg < 0:
        turn_desc = "rotated AWAY from the light"
    else:
        turn_desc = "no net rotation"
    if pheromone_speed_response > 0:
        speed_desc = "ACCELERATED"
    elif pheromone_speed_response < 0:
        speed_desc = "DECELERATED"
    else:
        speed_desc = "no change"

    # Net translation toward/away from the light, independent of module_1's
    # own yaw rotation (the body can rotate to face the light while its net
    # translation still drifts the other way).
    def _translation_desc(displacement_xy, light_xy):
        direction_to_light = np.asarray(light_xy) - origin_xy
        if np.linalg.norm(direction_to_light) < 1e-9:
            return "light at robot's own position - undefined"
        moved = np.dot(displacement_xy, direction_to_light)
        if moved > 0:
            return "moved TOWARD the light"
        if moved < 0:
            return "moved AWAY from the light"
        return "no net translation toward/away"

    left_translation_desc = _translation_desc(left["displacement_xy"], left_light_xy)
    front_translation_desc = _translation_desc(front["displacement_xy"], front_light_xy)

    print("\n=== run_light_tests summary ===")
    print(f"No light - avg velocity: {baseline_v * 1000:.3f} mm/s")
    print(f"Left stage - pheromone_yaw_response_deg: {pheromone_yaw_response_deg:.2f} deg "
          f"({turn_desc}); {left_translation_desc}")
    print(f"Front stage - avg velocity: {front['avg_linear_mps'] * 1000:.3f} mm/s, "
          f"pheromone_speed_response: {pheromone_speed_response:.4f} ({speed_desc}); {front_translation_desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        #"--m", type=str, default="../models/assembly.xml",
        #"--m", type=str, default="..\\output\\evolution_run\\generation_9\\ind5_assembly.xml",
        "--m", type=str, default="../models/light_phermone_attracted_assembly.xml",
        #"--m", type=str, default="D:\\microrobotics\\output\\evolution_run\\generation_59\\ind5_assembly.xml",
        help="MJCF model path to run in the live viewer",
    )

    parser.add_argument(
        "--o", type=str, default="../output/simulation_stats.json",
        help="Output path for simulation statistics JSON file",
    )

    parser.add_argument(
        "--max_sim_time", type=float, default=7.0,
        help="Maximum simulation time in seconds (for headless runs)",
    )

    parser.add_argument(
        "--headless", action="store_true",
        help="Run without the live viewer (default: show the live viewer)",
    )

    parser.add_argument(
        "--capture_img", action="store_true",
        help="With --headless, also save a final-pose screenshot PNG",
    )

    parser.add_argument(
        "--capture_gif", action="store_true",
        help="With --headless, also save a movement GIF (post torque-ramp)",
    )

    parser.add_argument(
        "--sweep_b", action="store_true",
        help="With --headless, sweep B_intensity over --b_values and keep the "
             "highest-velocity success+stable run (see run_headless_b_sweep)",
    )

    parser.add_argument(
        "--b_values", type=str, default=None,
        help="Comma-separated Tesla values for --sweep_b "
             f"(default: {','.join(str(b) for b in B_SWEEP_VALUES)})",
    )

    parser.add_argument(
        "--light_tests", action="store_true",
        help="Run the two-stage light-response test (see run_light_tests) in the "
             "live viewer instead of the normal free-roam gait.",
    )
    parser.add_argument(
        "--light_test_duration", type=float, default=LIGHT_TEST_DEFAULT_DURATION,
        help=f"With --light_tests, seconds of magnetic actuation per stage "
             f"(default: {LIGHT_TEST_DEFAULT_DURATION}).",
    )
    parser.add_argument(
        "--capture_video", action="store_true",
        help="With --light_tests, also save a high-quality MP4 of every stage "
             "(baseline/left/front), captured via an offscreen renderer independent "
             "of the live viewer window.",
    )
    parser.add_argument(
        "--video_fps", type=int, default=30,
        help="With --light_tests --capture_video, output frames per second (default: 30).",
    )
    parser.add_argument(
        "--light_module_color", type=str, default=None,
        help="Comma-separated RGBA (e.g. '0,1,0,1') to recolor whichever module "
             "carries the light-sensitive joint, so it's easy to spot against the "
             "rest of the (uniformly blue) assembly in the viewer/screenshots/video. "
             "Default: leave its normal body color alone.",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Optional log file path to write simulator logs to.",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Optional log level for simulator (DEBUG, INFO, WARNING, ERROR).",
    )

    args = parser.parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    level = logging.INFO
    if args.log_level:
        try:
            level = getattr(logging, args.log_level.upper(), logging.INFO)
        except Exception:
            level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    light_module_rgba = (
        tuple(float(v) for v in args.light_module_color.split(","))
        if args.light_module_color else None
    )

    if args.light_tests:
        run_light_tests(
            args.m, args.o, light_test_duration=args.light_test_duration,
            capture_video=args.capture_video, video_fps=args.video_fps,
            media_dir=os.path.dirname(args.o) or ".",
            light_module_rgba=light_module_rgba,
        )
    elif args.headless:
        if args.sweep_b:
            b_values = (
                tuple(float(v) for v in args.b_values.split(","))
                if args.b_values else B_SWEEP_VALUES
            )
            run_headless_b_sweep(
                args.m, args.o, b_values=b_values, max_sim_time=args.max_sim_time,
                capture_img=args.capture_img, capture_gif=args.capture_gif,
                media_dir=os.path.dirname(args.o) or ".",
                light_module_rgba=light_module_rgba,
            )
        else:
            run_headless(
                args.m, args.o, max_sim_time=args.max_sim_time,
                capture_img=args.capture_img, capture_gif=args.capture_gif,
                media_dir=os.path.dirname(args.o) or ".",
                light_module_rgba=light_module_rgba,
            )
    else:
        run_with_viewer(args.m, args.o, max_sim_time=args.max_sim_time, light_module_rgba=light_module_rgba)