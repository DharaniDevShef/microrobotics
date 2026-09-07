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
# _prepare_magnet_arrays() alongside every parent_body_magnet_map.update()
# call -- see magnetic_field_callback for why the per-step hot loop needs
# this instead of walking the dict itself.
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
# the running state light_sensitive_joint_angle_deg()'s exponential
# relaxation needs to trace a continuous curve across set_angle_to_joint()'s
# reactive-fold calls, one per physics step, through any number of light
# on/off transitions. last_setpoint_deg/reported track whether this joint's
# arrival at its CURRENT setpoint (the evolved light_ctrl_jointN angle while
# lit, the baseline gait angle once released) has already been logged, so
# set_angle_to_joint() prints "reached"/"returned" once per excursion
# instead of every step - see LIGHT_JOINT_REACHED_TOLERANCE_DEG. Cleared
# alongside parent_body_magnet_map at the start of every
# run_headless/run_with_viewer/run_light_tests call, same reasoning as
# torque_history above - this module reuses the same process across many
# independent simulation runs.
_light_joint_state = {}

# {id(model): {body_id: frozenset(body ids in the same physical module)}} -
# get_light_sensor_values()'s multi-hit occlusion raycast uses this to skip
# past hits on the sensor's OWN module (see _module_self_bodies) rather than
# treating them as real occluders. Cleared alongside parent_body_magnet_map
# at the start of every run_headless/run_with_viewer/run_light_tests call -
# a fresh mujoco.MjModel is loaded each call, and id() can be reused once
# the previous one is garbage-collected, so a stale entry could otherwise
# apply to the wrong model.
_module_self_body_cache = {}

logger = logging.getLogger(__name__)

# Physics constants
# Magnetic field intensity (Tesla)
B_INTENSITY = 0.025  # mT
# Candidate drive strengths for run_headless_b_sweep() -- the field a
# morphology needs to overcome stiction and walk (rather than stall, or
# over-drive into a rolling/tumbling gait) is morphology-dependent, so
# this is swept per run rather than assumed fixed.
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

# Offscreen screenshot/GIF render resolution. The visualizer only ever
# displays these at a couple hundred px wide (SCREENSHOT_CARD_WIDTH=260 in
# evolution_results_visualizer.py), so 1400x1080 was still far more than
# needed - 480x370 (same ~1.3 aspect ratio) is still ~1.8x that card width
# (retina-sharp at 2x), while cutting renderer-construction + render time
# roughly 30% (profiled: 0.48s -> 0.34s per screenshot) and the GPU/EGL
# context memory each of sim_executor.py's parallel subprocesses holds -
# a plausible source of the occasional renderer failure some individuals
# were hitting.
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080

# run_light_tests()'s capture_video renderer is a separate, single-run
# diagnostic path (not part of sim_executor.py's parallel evolution pool
# that the RENDER_WIDTH/HEIGHT comment above warns about), so it can afford
# to use the full offscreen buffer the model provisions (assembly.xml's
# <global offwidth="1920" offheight="1080"/>) for maximum video quality.
VIDEO_RENDER_WIDTH = 1920
VIDEO_RENDER_HEIGHT = 1080

# Pre-gait "settle" phase (see run_headless): how long to let the hinge
# position actuators reach their evolved target angles - and the body
# settle under gravity - before a requested screenshot is taken, and the
# hard cap on that in case a configuration never quite stops jittering.
SETTLE_MAX_STEPS = 300  # 10 steps * 0.01s timestep = 0.1s settle phase
SETTLE_VELOCITY_THRESHOLD = 1e-3  # rad/s or m/s - "basically stopped moving"

# 2D/3D shape entropy (entropy_api.py) window sizes, in "radius r" hex-cell
# units per the reference thesis - window_size = 2r + 1. Kept small since
# assemblies here are modest-sized (a handful to a few dozen modules), so
# a window has to stay small to see multiple independent samples of it.
SHAPE_ENTROPY_WINDOW_SIZES = (3, 5, 7)  # r = 1, 2, 3

# Wall geoms are tagged group=1 in the model (see mjcf_generator.py) so this
# raycast can be filtered to see ONLY them.
_WALL_GEOMGROUP = np.zeros(6, dtype=np.uint8)
_WALL_GEOMGROUP[1] = 1

# mjcf_generator.py's `luminance_sheet` geom is tagged group=2 - it's a
# purely cosmetic patch _show_luminance_sheet() can size to cover an
# entire stage region right at sensor/light height, so
# get_light_sensor_values()'s occlusion raycast (mj_ray, which -- unlike
# contact detection -- ignores contype/conaffinity and would otherwise
# treat the sheet itself as a giant real occluder) filters it out via this
# "every group except 2" mask rather than the "wall-only" one above.
_OCCLUSION_GEOMGROUP = np.ones(6, dtype=np.uint8)
_OCCLUSION_GEOMGROUP[2] = 0

# Duration (s) of magnetic actuation applied per run_light_tests() stage.
LIGHT_TEST_DEFAULT_DURATION = 10
# How long (s) run_light_tests() moves the assembly with no light active at
# all, before the light stages start, to get a baseline avg linear velocity.
LIGHT_TEST_BASELINE_DURATION = 7.0
# Extra distance (m) the "left" stage's light is pushed beyond the left
# half-region's own centroid, further out to that side.
LIGHT_TEST_LEFT_EXTRA_OFFSET = 0.01
# Minimum net baseline (no-light) displacement (m) run_light_tests() needs
# before its direction is trustworthy enough to reorient the assembly by --
# below this, "which way did it travel" is dominated by settle/numerical
# noise, not a real heading, so the reorientation is skipped.
LIGHT_TEST_MIN_BASELINE_TRAVEL = 0.0005
# How far (m) below the lowest light_sensor_* site pheromone_light_height()
# places the floor-level "luminance sheet" light - see there for why it's
# relative to the sensors' own settled height rather than a fixed distance
# from the floor.
LIGHT_TEST_PHEROMONE_MARGIN_M = 0.001
# World Z height (m) _show_luminance_sheet() renders the visible yellow
# patch at - just above the floor (a real MuJoCo <geom type="plane">,
# opaque, at world Z=0) so it's never hidden behind it, unlike
# pheromone_light_height()'s own (often negative) height for the actual
# light - see _show_luminance_sheet()'s docstring.
LUMINANCE_SHEET_VISUAL_HEIGHT_M = 0.0003
# How much bigger (both X and Y) _show_luminance_sheet() draws the visible
# patch than the stage's own light_bounds rectangle it's centered on -
# purely cosmetic (get_light_sensor_values()'s actual sensed region is
# still exactly `bounds`, unaffected by this).
LUMINANCE_SHEET_VISUAL_SCALE = 5.0
# Visible RGBA of mjcf_generator.py's `luminance_sheet` placeholder geom
# while _show_luminance_sheet() has it covering a stage's region - bright
# yellow, matching the UV-excited fluorescent pheromone trace it represents
# (see the wireless-pheromone-robot paper this models). Paired with the
# "luminance_glow" material's high emission (mjcf_generator.py) so the
# sheet reads as actually luminous/glowing on video, not just flat-colored.
LUMINANCE_SHEET_RGBA = np.array([1.0, 0.85, 0.0, 1.0])
# run_light_tests() reactive fold: a joint whose own light_sensor_joint_*
# reading exceeds this (lux) gets driven to ITS OWN evolved light_ctrl_jointN
# angle (read_light_target_angles_from_xml) -- every other joint (including
# one with no such metadata at all) is left exactly as it was.
LIGHT_TRIGGER_LUX_THRESHOLD = 10000.0
# Flat lux reading get_light_sensor_values() reports for any sensor
# physically inside a bounded light's footprint (the luminance sheet) -
# see the light_bounds docstring there. Deliberately a fixed constant
# safely above LIGHT_TRIGGER_LUX_THRESHOLD rather than model.light_intensity
# itself: light_intensity also drives the light's actual rendered
# brightness (cosmetic), and under the old physically-modeled Lambert's-law
# + inverse-square-falloff treatment a modest intensity only ever crossed
# the trigger threshold because of that falloff amplifying it at
# millimeter-scale sensor distances - a plain "puddle" model has no
# falloff to amplify anything, so reusing light_intensity directly would
# almost never trigger. Keeping this separate preserves both directions of
# the existing rule that cosmetic and sensing values never affect each
# other.
LUMINANCE_SHEET_LUX = 20000.0
# Time constant tau_c (s) of a light-sensitive joint's exponential
# relaxation response - see light_sensitive_joint_angle_deg(). A joint
# doesn't jump between its evolved light_ctrl_jointN angle and its
# baseline gait angle instantly; it eases toward whichever one currently
# applies (light_ctrl_jointN while lit, baseline once the light is lost)
# with this same time constant on both the entry and exit transition.
LIGHT_JOINT_TIME_CONSTANT_S = 1
# set_angle_to_joint()'s reactive fold: how close (deg) a light-sensitive
# joint's eased angle has to get to its current setpoint (the evolved
# light_ctrl_jointN angle while lit, the baseline gait angle once released)
# before it's considered to have actually arrived there, for the one-shot
# "reached"/"returned" log lines - see light_sensitive_joint_angle_deg(),
# whose exponential curve only ever asymptotically approaches its setpoint
# and never exactly equals it.
LIGHT_JOINT_REACHED_TOLERANCE_DEG = 0.5
# get_light_sensor_values()'s multi-hit occlusion raycast: how many bounces
# off the sensor's own module (_module_self_bodies) it will skip past
# before giving up and treating the ray as occluded. A folded module's own
# geometry only overlaps itself in a couple of places near the hinge, so
# this only ever needs to be a handful, not a large search.
_SELF_OCCLUSION_MAX_HITS = 6
# Distance (m) a skipped self-hit's ray origin is nudged forward by before
# re-casting, so mj_ray doesn't immediately re-hit the exact same surface
# (a hit at t=0 would otherwise re-trigger the same skip forever).
_SELF_OCCLUSION_RAY_EPS = 1e-6



def _prepare_magnet_arrays():
    """Flattens parent_body_magnet_map into the arrays magnetic_field_callback
    needs, in one shot. Call this once right after every
    parent_body_magnet_map.clear()/.update(find_all_magnets(...)) pair (the
    3 run_headless/run_with_viewer/run_light_tests call sites) -- NOT from
    inside the callback itself, which runs once per physics step and can't
    afford to re-walk the dict (and re-build lists of Python (geom_id,
    polarity_sign) tuples) 700+ times per run just to read it."""
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
    """
    MuJoCo Control Callback.
    Calculates magnetic torque:
        Tau = M x B
    Applies torque to parent module bodies and records net applied torque magnitude.

    Vectorized over every magnet at once via _magnet_geom_ids/_magnet_moments/
    _magnet_owner (see _prepare_magnet_arrays) instead of a per-magnet Python
    loop -- this is the dominant per-step cost of a headless run (profiling
    showed it costing ~5x the underlying mj_step itself), since MuJoCo calls
    this once per physics step. The closed form below exploits b_vector
    always being Z-only (per the oscillating-field scheme a few lines down):
    for world dipole m=(mx,my,mz) and B=(0,0,b_z),
        Tau = m x B = (my*b_z, -mx*b_z, 0)
    so only the world-frame X/Y dipole components are needed -- i.e. just
    the [2] and [5] entries (the local-Z-axis column) of each magnet geom's
    flattened 3x3 xmat, not a full matrix multiply + 3D cross product per
    magnet. If b_vector is ever generalized away from Z-only (see the
    commented-out XZ-plane scheme above), this closed form must go back to
    a real per-magnet cross product.
    """
    # Clear previous external forces
    data.xfrc_applied.fill(0)
    # Smooth torque ramp
    ramp_progress = min(data.time / TORQUE_RAMP_TIME, 1.0)
    # Smoothstep: f(x)=x^2*(3−2x)
    # 0 -> 1 with zero slope at start and end
    ramp_factor = ramp_progress * ramp_progress * (3 - 2 * ramp_progress)
    effective_torque_multiplier = TORQUE_MULTIPLIER * ramp_factor

    # # Oscillating magnetic field settings
    # time_in_cycle = data.time % TOTAL_CYCLE_TIME

    # # The magnetic field is controlled to roll forward to -50◦in 0.9 s,
    # # and then backward to 50◦in 0.1 s, allowing the robot to
    # # slowly tilt down and quickly tilt up to perform the stick-slip motion

    # # Change from (-50 + 100*alpha)to (50 - 100*alpha)
    # if time_in_cycle < 0.9:
    #     alpha = time_in_cycle / 0.9
    #     theta = np.radians(-50.0 + (100.0 * alpha))
    # else:
    #     alpha = (time_in_cycle - 0.9) / 0.1
    #     theta = np.radians(50.0 - (100.0 * alpha))

    # # Magnetic field in XZ plane
    # b_vector = np.array([np.cos(theta), 0.0, np.sin(theta)]) * B_INTENSITY



    # Oscillating magnetic field along Z-axis (vertical) to induce walking motion.
    # It flips sign along a single fixed vertical axis at frequency f:
    #   Phase 1 (B = -z): torque pitches the robot forward about its front foot.
    #   Phase 2/3 (B = +z): torque lifts the front foot, robot rotates about its
    #   COM and lands back on its rear foot.
    # Half-period = 1/(2f) per the paper's t1 = 1/(2f).
    half_cycle_time = TOTAL_CYCLE_TIME / 2.0
    time_in_cycle = data.time % TOTAL_CYCLE_TIME

    b_z = -B_INTENSITY if time_in_cycle < half_cycle_time else B_INTENSITY

    n_parents = len(_magnet_parent_body_ids)
    if n_parents == 0:
        torque_history.append(0.0)
        b_field_history.append(b_z)
        time_history.append(data.time)
        return

    # World-frame dipole X/Y components for every magnet at once: column 2
    # (indices 2, 5 of the flattened row-major 3x3) of each magnet geom's
    # xmat is its local +Z axis in world coordinates, scaled by that
    # magnet's (signed) moment.
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
    for every body in `model` - every bodyBase_N/bodyLink_N/bodyRigid_N/
    connectorN_N etc. under the same top-level module_N body (found via
    find_main_movable_parent) maps to the identical frozenset, itself
    included. get_light_sensor_values() uses this so its occlusion
    raycast can treat a hit on the sensor's OWN module as transparent:
    bodyBase_N and bodyLink_N are, by construction, touching/overlapping
    right at the fold's own hinge (that's where the sensor sits), which
    isn't a real occluder the way a neighboring module or the floor is."""
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

    Polarity comes from which connector mesh (SGA/SGB/SGX) is mounted at
    that site -- see doc/design_info.txt ("N or S pole facing outward").
    SGB sites are the mating partner of SGA (magnets attract when
    complementary, per the smart-glue design) so they get the opposite
    sign; SGX (unmated/free) sites keep the default pole. That mesh name
    lives on the sibling "geom_connectorN_..." geom on the same body, not
    on this magnet's own "magnet_connectorN_..." name.
    """
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
    """Recolors the bodyBase/bodyLink geoms of whichever module carries the
    light-sensitive joint (the one wearing a light_sensor_joint_* site -
    see mjcf_generator.py's is_light_sensitive) to `rgba`, so it stands out
    from the rest of the (uniformly blue) assembly in the viewer/
    screenshots/video. A no-op if this model has no such module (e.g. a
    genotype whose light-sensitive design variable didn't select any
    foldable module)."""
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
    read from `data.xpos` as of whatever pose `data` currently holds (the
    caller is responsible for having called mj_forward/mj_step to resolve
    it first). Each module is its own freejoint-rooted body - see
    mjcf_generator.py - so this position already reflects wherever the
    weld-constraint solver has settled that module, not just a fold
    joint's local rotation."""
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
    hinge_angle_on_light_detection - mjcf_generator.py's
    "light_ctrl_jointN" custom numerics, one per light-sensitive joint,
    named to match its "ctrl_jointN" baseline counterpart) from XML
    metadata. Only present for joints that are actually light_sensitive -
    see _write_joint_target_angles()'s docstring - so this dict is a
    strict subset of read_joint_target_angles_from_xml()'s keys, possibly
    empty if the graph has no light-sensitive joints at all."""
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

    The SAME formula and time constant drive both directions: on light
    entry, setpoint is the joint's own evolved light_ctrl_jointN angle
    (Design Variable 6, hinge_angle_on_light_detection, range [0, 45] - see
    roblet_grammar.py) and theta eases UP toward it from wherever it was;
    on light exit, setpoint reverts to the joint's baseline (non-light)
    gait angle and theta eases back DOWN toward that instead. Starting from
    theta_prev=0 with a constant setpoint this reduces to the textbook
    charging curve theta(t) = setpoint * (1 - e^(-t/tau_c)); starting from
    theta_prev=setpoint and driving setpoint to 0 gives the matching
    discharge curve theta(t) = theta_prev * e^(-t/tau_c). set_angle_to_joint()
    calls this once per physics step with dt_s = that step's elapsed time
    and feeds each return value back in as the next call's theta_prev_deg
    (see _light_joint_state), tracing one continuous curve through any
    number of light on/off transitions rather than just those two
    closed-form special cases. tau_c defaults to LIGHT_JOINT_TIME_CONSTANT_S
    (1s)."""
    if dt_s <= 0:
        return theta_prev_deg
    return setpoint_deg + (theta_prev_deg - setpoint_deg) * np.exp(-dt_s / tau_c)


def set_angle_to_joint(model, data, target_angle_deg, light_bounds=None, light_target_angles=None, frame=None):
    """Set actuator position targets from a scalar, ordered sequence, or name map.

    light_target_angles: optional {"light_ctrl_jointN": angle_deg} map
    (roblet_simulator.read_light_target_angles_from_xml) - the evolved
    Design Variable 6 (hinge_angle_on_light_detection) for each
    light-sensitive joint. For any joint that has an evolved value here,
    its actuator target continuously eases toward whichever setpoint
    currently applies - THIS joint's own evolved value while its
    light_sensor_* reading exceeds LIGHT_TRIGGER_LUX_THRESHOLD, or back to
    the baseline (non-light) gait angle once it drops below - via
    light_sensitive_joint_angle_deg()'s exponential relaxation (same
    tau_c = LIGHT_JOINT_TIME_CONSTANT_S on entry and exit), rather than
    jumping instantly either way. A joint with no evolved metadata at all
    (e.g. a hand-built model missing that custom numeric) is left at
    whatever the baseline pass already set, with no reactive fold in
    either direction. _light_joint_state records, per actuator, the last
    eased angle and the data.time it was computed at, so repeated calls
    trace one continuous curve across steps instead of restarting it every
    time.

    Each light-sensitive joint also gets exactly one logger.info() line per
    excursion (not a per-step print): one when its eased angle first comes
    within LIGHT_JOINT_REACHED_TOLERANCE_DEG of its evolved light_ctrl_jointN
    angle after triggering, and one when it comes back within that same
    tolerance of its baseline angle after the light is lost. Tracked via
    _light_joint_state's own `reported` flag, reset whenever the setpoint
    itself changes (light on -> off or off -> on), so repeated calls while
    already settled at the current setpoint stay silent.

    frame: optional (origin_xy, forward) forwarded to
    get_light_sensor_values() - see there - so light_bounds can be
    expressed in a travel-aligned local frame instead of raw world XY."""
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
                # this joint has no evolved light_ctrl_jointN of its own
                # (e.g. a hand-built model missing that metadata) - leave
                # its target exactly as the baseline pass already set it,
                # rather than guessing a made-up trigger angle. No
                # reactive fold happens for it, in either direction, on
                # either the interactive or headless path.
                continue

            baseline_deg = target_values[actuator_idx]
            lit = lux > LIGHT_TRIGGER_LUX_THRESHOLD
            setpoint_deg = light_target_angles[light_key] if lit else baseline_deg
            # Default prev_setpoint_deg is baseline_deg (NOT setpoint_deg) -
            # a joint seen for the first time is treated as having been at
            # rest at baseline "forever", so if it's ALREADY lit on this
            # very first observation, that still counts as a setpoint
            # change and gets its own "reached" line once it settles,
            # rather than being silently swallowed by looking like no
            # change had happened at all.
            theta_prev_deg, prev_time, prev_setpoint_deg, reported = _light_joint_state.get(
                actuator_idx, (baseline_deg, data.time, baseline_deg, True))
            eased_deg = light_sensitive_joint_angle_deg(
                theta_prev_deg, setpoint_deg, data.time - prev_time)

            # if not np.isclose(setpoint_deg, prev_setpoint_deg):
            #     reported = False  # setpoint just changed - allow one fresh "reached" line
            # if not reported and abs(eased_deg - setpoint_deg) <= LIGHT_JOINT_REACHED_TOLERANCE_DEG:
            #     reported = True
            #     if lit:
            #         logger.info(
            #             "Joint '%s' light-triggered: reached full angle %.2f deg (lux=%.0f).",
            #             actuator_name, eased_deg, lux,
            #         )
            #     else:
            #         logger.info(
            #             "Joint '%s' light released: returned to original angle %.2f deg.",
            #             actuator_name, eased_deg,
            #         )

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
    """Minimum of the 4 wall-facing rangefinder rays (m), or None if none of
    them currently hit a wall.

    Uses a manual, geom-group-filtered mj_ray() rather than a native
    <rangefinder> sensor: the native sensor only excludes the ray site's own
    body, so as module_1 pitches through the flip gait it would "see" the
    floor or a neighboring welded module (only ~8mm away) as a hit. Filtering
    to geom group 1 (the walls only) makes those false positives impossible.
    """
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
    frame: `right` is `forward` rotated -90 deg (clockwise), so when
    forward already equals world +Y this reduces to (world +Y, world +X) -
    i.e. every local-frame helper below is a strict generalization of the
    old world-axis-aligned bounds, not a behavior change for that case."""
    forward = np.asarray(forward, dtype=float)
    return forward, np.array([forward[1], -forward[0]])


def world_to_local_xy(point_xy, origin_xy, forward):
    """World XY -> (local_x, local_y) in the travel-aligned frame anchored
    at origin_xy: local_y grows along `forward` (e.g. front of travel),
    local_x grows to the right of it (mirroring how, e.g., world +X was
    "right" once the whole robot used to be physically rotated so its own
    forward pointed along world +Y - see _travel_frame_axes)."""
    forward, right = _travel_frame_axes(forward)
    rel = np.asarray(point_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    return float(np.dot(rel, right)), float(np.dot(rel, forward))


def local_to_world_xy(local_xy, origin_xy, forward):
    """Inverse of world_to_local_xy() - used to place a real MuJoCo <light>
    (which only understands world coordinates) at a position chosen in the
    travel-aligned local frame."""
    forward, right = _travel_frame_axes(forward)
    lx, ly = local_xy
    return np.asarray(origin_xy, dtype=float) + lx * right + ly * forward


def assembly_local_bounds(model, data, origin_xy, forward):
    """Axis-aligned bounding box, as (mins, maxs) each [x, y], of every
    robot geom -- i.e. every geom whose body is *not* the worldbody (body
    0, which is where mjcf_generator.py declares the floor/perimeter walls
    directly, with no wrapping <body>) -- expressed in the travel-aligned
    local frame (world_to_local_xy) instead of raw world XY. So "left
    half"/"front half" of the returned box actually mean left/front of the
    robot's OWN measured travel direction, without ever having to
    physically rotate the robot's (contact-consistent, already-settled)
    pose to make world axes line up with it. Used to size/center a ceiling
    light's square coverage footprint over a specific half of the
    assembly."""
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
    """The world Z height (m) run_light_tests()/run_headless_light_tests()
    position their floor-level "luminance sheet" light at: LIGHT_TEST_
    PHEROMONE_MARGIN_M below the LOWEST of the model's own light_sensor_*
    sites in `data`'s current pose.

    Why the lowest one, not a fixed height: light_sensor_joint_* sites now
    face -Z (mjcf_generator.py), so get_light_sensor_values() only lights
    one up when the light sits BELOW it on the same side of the floor
    plane (a real MuJoCo <geom type="plane">, which occludes a ray that
    crosses it) - and different joints settle at slightly different
    heights, some a hair above world Z=0, some a hair below. Placing the
    light below every sensor at once (rather than some fixed offset from
    the floor) is the only way to guarantee the closest-to-the-ground
    sensors - the entire point of mounting them at the backside/bottom of
    the joint - actually register a same-side, unoccluded reading; a
    sensor that settles well above this run's lowest one may still miss
    it, the same way a real photodiode's exact mounting height affects
    what it can see of a ground-level light.

    Subtracting the margin is clamped to never cross world Z=0 itself when
    `lowest` is already non-negative: if every sensor sits at or above the
    floor (lowest - margin would otherwise land below it), blindly
    subtracting would put the light on the OTHER side of the floor plane
    from every single sensor - occluding all of them at once, including
    the lowest one a same-side height would still have lit.
    """
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
    meters (see pheromone_light_height()) at `center_xy` (world XY) - a
    floor-level "luminance sheet" light (the UV-excited fluorescent
    pheromone trace from the wireless-pheromone-robot paper this models),
    not the old overhead ceiling spotlight. light_dir doesn't affect
    get_light_sensor_values()'s own lux math for a non-directional light
    (only light_pos does - see there), so "-1" here is purely cosmetic,
    matching how a real fixture over the sheet would be aimed.

    ambient/specular are pinned to mjcf_generator.py's own <light
    ambient="0.1 0.1 0.1" ... specular="0.05 0.05 0.05" .../> - the exact
    same values run_light_tests()'s "baseline" stage sees (it only sets
    light_type/light_active, never touches these, so it's just reading the
    model's own compiled defaults) and a normal run_headless()/
    run_with_viewer() call sees too. Those values are deliberately low -
    see LIGHT_AMBIENT/LIGHT_DIFFUSE/LIGHT_SPECULAR in mjcf_generator.py -
    because this light is DIRECTIONAL during "baseline" (no distance
    falloff - covers the whole floor at full strength) and the viewer's
    own camera headlight always adds on top by default; at old, higher
    cosmetic values the two together blew the floor out to solid white in
    ANY run, not just a light-test one. diffuse is intentionally NOT
    matched to baseline the same way - a distinct warm tint here is a
    deliberate cosmetic identity for the pheromone light specifically, not
    an oversight. light_type/light_pos staying SPOT and floor-level
    (rather than baseline's own DIRECTIONAL/overhead) is a difference that
    can't be removed at all: it's what pheromone_light_height() and
    get_light_sensor_values() actually need for the reactive fold to sense
    the sheet in the first place. Purely cosmetic otherwise -
    get_light_sensor_values() never reads light_ambient/diffuse/specular,
    only light_intensity, so none of this affects lux/sensing."""
    model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_SPOT
    model.light_pos[light_id] = np.array([center_xy[0], center_xy[1], height])
    model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])
    model.light_diffuse[light_id] = np.array([1.0, 0.9, 0.3])
    model.light_ambient[light_id] = np.array([0.1, 0.1, 0.1])
    model.light_specular[light_id] = np.array([0.05, 0.05, 0.05])


def _scale_local_bounds(bounds, anchor_local_xy, scale):
    """Scales a (x_lo, x_hi, y_lo, y_hi) travel-aligned-local-frame
    rectangle by `scale`, growing each edge away from anchor_local_xy
    instead of the rectangle's own center - e.g. "left" passes (mid_x,
    mid_y) so the inner edge at mid_x (the boundary against the robot's
    OTHER half) stays put and only the outer edge moves further left as
    the scale grows, rather than growing symmetrically and eating into the
    robot's center/other half; "front" pins its own near edge (closest to
    the robot) the same way. Passing bounds' own center reproduces plain
    symmetric growth in both directions.

    Shared by _show_luminance_sheet() (the VISIBLE patch) and every
    left_bounds/front_bounds passed as light_bounds into get_light_sensor_
    values()/set_angle_to_joint() (the actual trigger/release check) -
    using the same helper (and the same LUMINANCE_SHEET_VISUAL_SCALE, same
    anchor) for both keeps them from ever drifting apart again: what's
    rendered as the sheet is exactly what's sensed, not a bigger-looking
    patch sitting on top of a smaller invisible one."""
    x_lo, x_hi, y_lo, y_hi = bounds
    ax, ay = anchor_local_xy
    x_lo, x_hi = ax + scale * (x_lo - ax), ax + scale * (x_hi - ax)
    y_lo, y_hi = ay + scale * (y_lo - ay), ay + scale * (y_hi - ay)
    return (x_lo, x_hi, y_lo, y_hi)


def _show_luminance_sheet(model, sheet_geom_id, bounds, anchor_local_xy, origin_xy, forward):
    """Resizes, recolors and re-orients mjcf_generator.py's `luminance_sheet`
    placeholder geom into a visible yellow patch covering `bounds` (the
    same travel-aligned-local-frame rectangle passed as this stage's
    light_bounds), scaled up by LUMINANCE_SHEET_VISUAL_SCALE via
    _scale_local_bounds() - run_light_tests()'s video-only counterpart to
    _place_pheromone_light(). Callers scale their own light_bounds by the
    same factor/anchor (see _scale_local_bounds()) before passing them to
    get_light_sensor_values()/set_angle_to_joint(), so what's rendered here
    and what's actually sensed are the same rectangle, not a bigger-looking
    patch sitting on top of a smaller invisible one.

    Always sits at LUMINANCE_SHEET_VISUAL_HEIGHT_M, a small height ABOVE
    the floor - deliberately NOT pheromone_light_height()'s own (often
    negative, "below the lowest sensor") height, which would frequently
    place it behind the floor's own opaque geom and render as invisible.

    anchor_local_xy: see _scale_local_bounds()."""
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
    """Illuminance (lux) at each on-body light sensor (the `light_sensor_*`
    sites from mjcf_generator.py), summed over every <light> in the scene.

    verbose=False skips the print (e.g. when this is called every step for
    up-to-date values but only needs to print occasionally).

    light_bounds: optional {light_id: (x_lo, x_hi, y_lo, y_hi)} restricting
    a light to a rectangular footprint -- e.g. the floor-level luminance
    sheet that only covers part of the assembly (see run_light_tests). A
    bounded light is treated as a uniform puddle rather than a directional
    beam: any sensor inside the footprint reads a flat `intensity` lux
    from it regardless of the sensor's own angle or any occlusion, and a
    sensor outside reads exactly 0 -- so entering/leaving the footprint is
    the only thing that ever changes the reading, never a joint's own fold
    or a neighboring module's geometry. Lights not present in the dict
    keep the full physically-modeled Lambert's-law + occlusion treatment
    below.

    frame: optional (origin_xy, forward) - when given, each sensor's world
    XY is transformed via world_to_local_xy() before the bounds check
    above, so `light_bounds` is interpreted in that travel-aligned local
    frame instead of raw world XY (see assembly_local_bounds). None (the
    default) keeps the original world-XY bounds check.

    MuJoCo has no native illuminance sensor/API -- <light> is a
    rendering-only construct, so this is a hand-rolled photometric estimate
    built from MuJoCo's own light data (mj_ray for occlusion, plus each
    light's geometry and its `intensity` field, read straight from the
    model rather than a duplicated Python constant -- see
    LIGHT_INTENSITY_LUX in mjcf_generator.py):

      lux = intensity * max(0, cos(theta)) / falloff   (0 if occluded)

    where theta is the angle between the sensor panel's own outward normal
    (the SITE's own orientation, its local +Z axis rotated into world frame
    -- mjcf_generator.py orients each light_sensor_joint_N site to face
    whichever way that joint's sensor is actually mounted, e.g. downward
    for the backside-of-the-joint placement, not just the owning body's own
    Z axis) and the direction to the light (Lambert's cosine law), and
    falloff is 1 for a directional light (parallel rays, no distance
    attenuation) or distance^2 for a positional light (inverse-square law).

    Occlusion uses a multi-hit raycast (up to _SELF_OCCLUSION_MAX_HITS
    bounces) that skips past any hit on a body in the sensor's own physical
    module (_module_self_bodies) before checking real occlusion: a folded
    module's own bodyBase_N/bodyLink_N are, by construction, touching or
    overlapping right at the hinge the sensor sits on, so a naive single-hit
    raycast would treat the sensor's own module as permanently self-blocking
    (this is why a single bodyexclude -- which only ever covers the site's
    own body, not its sibling bodies -- isn't enough on its own). A hit on
    any OTHER body (a neighboring module, the floor, a wall) still counts as
    real occlusion.
    """
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
                # Luminance sheet: a floor-level chemical/pheromone puddle,
                # not a directional beam - any sensor physically inside its
                # footprint reads the same fixed lux no matter which way it
                # faces or what else is folded nearby, and reads exactly 0
                # the instant it's outside. This keeps entry and exit
                # symmetric and immune to a joint's own fold changing its
                # own sensor's angle, or a neighboring module swinging into
                # the line of sight - only position in/out of the sheet
                # matters, never angle or occlusion.
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
                    # Own module's own geometry, touching itself right at
                    # the fold - not a real occluder. Nudge past it and
                    # keep looking from there.
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
    samples (post torque-ramp): a steady walker's window-to-window speed
    stays in a fairly narrow, consistently-forward band, while a robot
    that's tumbling/bouncing instead of walking shows big swings -- most
    tellingly, whole one-second windows of *net backward* COM motion (a
    real stick-slip gait can micro-slip backward within a step, but
    shouldn't lose ground over a full second), or a speed spread so wide
    (stdev over mean) that "average velocity" isn't a meaningful summary.
    """
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

    physics_ok=False means a MuJoCo physics warning (bad qpos/qvel/qacc, a
    diverging/colliding model, ...) fired during the rollout, so every
    stat is meaningless -- write zeros for all of them instead of whatever
    partial numbers had accumulated up to the point of failure.

    is_stable=False means the run completed (physics-wise) but its gait
    looked like jumping/rolling rather than walking -- see
    _detect_instability(). The velocity/distance numbers are still real,
    just not a meaningful "how well does this walk" signal.

    success (written to the JSON) is physics_ok AND is_stable.

    b_intensity is the magnetic field strength (Tesla) this particular run
    used -- recorded so a B sweep (see run_headless_b_sweep()) can tell
    which candidate B produced the reported stats.

    shape_entropy_2d/3d (entropy_api.py) are the flat/folded normalized
    shape-entropy scalars computed in run_headless - objectives_api.py's
    f5 is their delta. Defaulted to 0.0 (not None) here so a failed run's
    JSON always has the same schema as a successful one.

    pheromone_yaw_response_deg / pheromone_speed_response come
    from run_headless_light_tests() (see run_headless()'s
    include_light_tests) -- both default to 0.0, both left at 0.0 whenever
    success is 0 (a failed/unstable run has no meaningful "no light"
    baseline to compare a light response against, so light tests aren't
    run for it at all -- see run_headless())."""

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

    # Total mass of the whole model
    total_mass = float(mujoco.mj_getTotalmass(model))

    # # Mass of each individual module body tree
    # module_masses = []
    # for body_id in sorted(module_labels.keys()):
    #     # body_subtreemass includes the body and all child geoms/bodies attached to it
    #     module_mass = float(model.body_subtreemass[body_id])
    #     module_masses.append({
    #         "module_id": module_labels[body_id],
    #         "mass_mg": round(module_mass * 1e6, 4)  # Convert to milligrams
    #     })

    # Average torque acting on whole body across all physics steps
    avg_torque = float(np.mean(torque_history)) if torque_history else 0.0

    stats = {
        "success": 1 if physics_ok and is_stable else 0,
        "physics_ok": 1 if physics_ok else 0,
        "is_stable": 1 if is_stable else 0,
        "B_intensity_T": round(b_intensity, 6) if b_intensity is not None else 0,
        "total_mass_mg": round(total_mass * 1e6, 4),  # Convert to milligrams
        # "module_masses_mg": module_masses,
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

    Both "left" and "front" are lit by the same floor-level "luminance
    sheet" mechanism as run_light_tests() (pheromone_light_height() +
    _place_pheromone_light(), via the _pheromone_light() closure below) -
    light_sensor_joint_* sites face -Z (mjcf_generator.py) specifically so
    they can register it.

    Runs its OWN dedicated no-light baseline stage first (LIGHT_TEST_
    BASELINE_DURATION seconds, light_bounds=None so no reactive fold can
    trigger regardless of the scene light), then defines "left"/"front"
    relative to THAT stage's own measured travel direction - not the
    primary gait rollout's net displacement over its whole (possibly
    curved/wandering) multi-second run, which turned out to be an
    unreliable heading, nor by physically rotating the settled pose to
    face any reference direction (world axis, camera, ...), which turned
    out to risk tipping over an already-marginal design before the light
    stages even start (a rigid Z-rotation of a contact-settled pose is not
    guaranteed to still be as stable). Instead the light-patch bounds
    themselves are expressed in a travel-aligned LOCAL frame (see
    assembly_local_bounds/world_to_local_xy) while the robot's own
    qpos is never touched after settling - see run_light_tests() for the
    live-viewer sibling using the identical approach.

    `model` is reused as-is (no reload from disk -- the caller already
    paid the compile cost); a fresh MjData is created so this starts from
    a clean simulation state regardless of whatever run_headless()'s own
    gait left the model in.

    light_target_angles (roblet_simulator.read_light_target_angles_from_xml)
    is this model's own evolved Design Variable 6
    (hinge_angle_on_light_detection) per light-sensitive joint -- forwarded
    into set_angle_to_joint()'s reactive-fold branch below so a triggered
    joint moves to ITS OWN evolved trigger angle (or doesn't trigger at all
    if this model has none recorded for it - see set_angle_to_joint).

    Returns {"pheromone_yaw_response_deg": ..., "pheromone_speed_response": ...}
    -- pheromone_yaw_response_deg = yaw_rotation(left stage) -
    yaw_rotation(baseline stage) (degrees); pheromone_speed_response =
    (avg_linear_mps(front stage) - avg_linear_mps(baseline stage)) /
    avg_linear_mps(baseline stage) (dimensionless; negative = slower than
    baseline (deceleration), positive = faster (acceleration)). If the
    baseline stage's avg_linear_mps is 0, pheromone_speed_response is
    reported as 0.0 rather than dividing by zero.
    """
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

    # ---- dedicated no-light baseline stage (light_bounds=None already
    # fully disables the reactive-fold branch in set_angle_to_joint,
    # regardless of the scene <light>'s own active/inactive state) ----
    reset_to_initial_pose()
    baseline_result = run_stage(LIGHT_TEST_BASELINE_DURATION, light_bounds=None, track_angular=True)

    # ---- travel-aligned local frame from the baseline stage's OWN
    # measured direction (falls back to world +Y if that travel was too
    # small to be a trustworthy heading) - the robot's qpos is NOT
    # rotated; only the light-patch bounds/positions below are expressed
    # in this frame (see assembly_local_bounds/world_to_local_xy/
    # local_to_world_xy) ----
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
        # Floor-level "luminance sheet" light (see pheromone_light_height()/
        # run_light_tests()'s identical mechanism) - light_sensor_joint_*
        # sites face -Z (mjcf_generator.py) specifically so they can see a
        # ground-level fixture like this one. MuJoCo only understands world
        # coordinates, so the chosen local-frame center is converted back
        # via local_to_world_xy() just for this light's placement.
        center_xy = local_to_world_xy(center_local_xy, origin_xy, forward)
        _place_pheromone_light(model, light_id, center_xy, pheromone_light_z)
        # data.light_xpos (what get_light_sensor_values() reads) is derived
        # kinematics - won't reflect this new model.light_pos until forward
        # kinematics runs again, so without this, run_stage()'s very first
        # iteration would read the PREVIOUS stage's light position.
        mujoco.mj_forward(model, data)

    frame = (origin_xy, forward)
    pheromone_light_z = pheromone_light_height(model, data)

    # ---- "left" stage ----
    reset_to_initial_pose()
    left_bounds = (xy_min[0], mid_x, xy_min[1], xy_max[1])
    _pheromone_light(((xy_min[0] + mid_x) / 2.0 - LIGHT_TEST_LEFT_EXTRA_OFFSET, mid_y))
    # Scaled by the same LUMINANCE_SHEET_VISUAL_SCALE/anchor run_light_tests()
    # renders its VISIBLE patch with (see _scale_local_bounds()), even though
    # this headless path never renders anything - keeps the two tools'
    # actual trigger/release boundary identical rather than this one
    # silently using the smaller, unscaled rectangle.
    left_sensed_bounds = _scale_local_bounds(left_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
    left_result = run_stage(light_test_duration, {light_id: left_sensed_bounds}, track_angular=True, frame=frame)

    # ---- "front" stage: lit patch covers the WHOLE body footprint
    # (centered on it), not a patch ahead of it - so both sensors are lit
    # simultaneously from the start ("both sensors detecting pheromone"),
    # matching what this stage's own pheromone_speed_response is actually
    # meant to measure, rather than a "walk toward a light ahead" scenario.
    # Same floor-level luminance sheet mechanism as "left" above. ----
    reset_to_initial_pose()
    front_bounds = (xy_min[0], xy_max[0], xy_min[1], xy_max[1])
    _pheromone_light((mid_x, mid_y))
    front_sensed_bounds = _scale_local_bounds(front_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
    front_result = run_stage(light_test_duration, {light_id: front_sensed_bounds}, track_angular=False, frame=frame)

    mujoco.set_mjcb_control(None)

    baseline_avg_velocity = baseline_result["avg_linear_mps"]
    # Re-wrap the difference of two already-wrapped angles into (-180, 180]
    # - without this, a pair straddling the wraparound boundary (e.g.
    # left=+170, baseline=-170) reports +340 instead of the true -20.
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
    real-time pacing, so it steps as fast as the CPU allows.

    Designed to be called once per OS process (via parallel_main) so that N
    different XML models can be simulated concurrently on N cores. Clears
    the module-level tracking state at the start so this is also safe if a
    process pool ever reuses a worker process across more than one model.

    If capture_media is True, saves a final-pose screenshot PNG and a GIF of
    the movement from TORQUE_RAMP_TIME (once the magnetic field is at full
    strength) to the end of the run, via mujoco.Renderer - an offscreen
    renderer that needs no visible window, so this works in a headless
    worker process just like the rest of this function.

    `model`/`target_angles` let a caller that's about to run this SAME
    model_path multiple times (run_headless_b_sweep's 3 B candidates + 1
    winner re-run) compile the MJCF and parse its target angles ONCE and
    pass them in here, instead of paying that cost 4x for byte-identical
    XML - MjModel compilation is the dominant fixed cost for a model this
    small, often more than the actual mj_step loop below. A fresh MjData
    is always created regardless, so per-run state never leaks even when
    `model` is reused across calls.
    include_light_tests=True additionally runs run_headless_light_tests()
    (see its docstring) after this run finishes, reusing this run's own
    already-loaded model. That function measures its own dedicated no-light
    baseline stage internally (reusing this run's avg_velocity/yaw as a
    baseline turned out to reorient the light rig by an unreliable
    direction - see run_headless_light_tests' docstring), so this adds a
    baseline + "left" + "front" stage on top, never a redundant repeat of
    the primary gait itself. Left False by default so run_headless_b_sweep()'s
    per-candidate-B trial runs (which get thrown away except for their
    avg_velocity) don't pay for it -- only its final re-run of the winning
    B does. Skipped entirely (both new stats left at 0.0) whenever this
    run's own success is 0 -- see below -- since there's no meaningful "no
    light" baseline to compare a light response against otherwise. Uses
    whatever the module-level B_INTENSITY currently is, same as this run's
    own gait (the global default for a standalone call, or the sweep's
    winning B when called from run_headless_b_sweep()'s final re-run).

    compute_shape_entropy_3d=True runs the settle-to-folded-pose pass (see
    the "3D shape entropy" block below) and computes shape_entropy_3d even
    when capture_img is False - separated from capture_img (which used to
    be the ONLY thing gating that pass) because sim_executor.py always
    calls run_headless_b_sweep() with capture_img=False for speed, which
    silently meant shape_entropy_3d was NEVER computed during a normal
    evolutionary run - stuck at its 0.0 initial value for every individual,
    every generation, collapsing objectives_api's f2_entropy =
    shape_entropy_3d - shape_entropy_2d into just -shape_entropy_2d. Left
    False by default, same reasoning as include_light_tests:
    run_headless_b_sweep()'s per-candidate-B trial runs get thrown away
    except for their avg_velocity, so they shouldn't pay for a settle pass
    whose entropy result would never be read - only its final re-run of
    the winning B passes this as True.

    light_module_rgba (default None = leave the normal body-blue color
    alone) recolors whichever module carries the light-sensitive joint -
    see _recolor_light_sensitive_module() - purely cosmetic, for spotting
    that module in a screenshot/GIF; has no effect on the physics. Applied
    here regardless of whether `model` was just loaded or passed in by the
    caller - run_headless_b_sweep() instead recolors its own shared
    `model` once, up front, rather than passing this into every
    per-candidate run_headless() call.
    """
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

    # module_1 is this design's designated sensor/control module (see
    # build_module_element()'s IMU/rangefinder comment in mjcf_generator.py)
    # -- forwarded to run_headless_light_tests() below as the reference
    # body for its own yaw tracking.
    module1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_module_1")
    module1_qpos_adr = model.jnt_qposadr[module1_joint_id] if module1_joint_id >= 0 else None

    # 2D shape entropy (entropy_api.py): from the model's own flat/
    # unfolded layout. mjcf_generator.py places every module body's XML
    # pos/quat to already satisfy the weld-mate geometry at zero fold, so
    # a single forward-kinematics pass at the default qpos (no stepping/
    # settling needed - unlike the folded pose below, nothing has to be
    # solved into place) gives correct flat positions.
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
            # A renderer/GPU context failure should cost this individual
            # its screenshot, not its whole physics result - the rest of
            # run_headless below still runs and still writes real stats.
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
    # Raycasting the 4 rangefinders every single 0.01s step is wasted work:
    # the robot moves on the order of mm/s, so it cannot close the 200mm
    # WALL_STOP_DISTANCE gap between one step and the next. Checking every
    # 10 steps (0.1s) is still far tighter than needed and cuts this cost 10x.
    WALL_CHECK_STRIDE = 10

    # Per-second windowed speed samples (post torque-ramp), for
    # _detect_instability() -- same windowing as run_with_viewer's printed
    # "Velocity: ..." line, just collected silently instead of printed.
    last_window_second = -1
    window_velocity_history = []
    window_displacement = None
    window_time = None

    try:
        # Settle to the folded pose (used for BOTH the screenshot and 3D
        # shape entropy below) - gated on capture_img OR compute_shape_entropy_3d
        # (either alone is enough to need it), not capture_img alone - see
        # compute_shape_entropy_3d's docstring for why that used to silently
        # skip shape_entropy_3d for every normal evolutionary run. Screenshot
        # rendering below stays its own separate `capture_img and renderer is
        # not None` check, not nested under this one, so a renderer/GPU
        # failure only costs the screenshot, never the entropy computation
        # (they need the same settled pose, but are otherwise independent).
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

            # The settle phase has its own simulated-time cost; reset it
            # so the torque ramp / max_sim_time cutoff / displacement
            # tracking below all start fresh, exactly as if the settle
            # phase had never happened.
            data.time = 0.0

            # 3D shape entropy (entropy_api.py): from this now-settled
            # folded pose - the weld-constraint solver has actually
            # repositioned every module relative to its neighbors as the
            # fold hinges rotated (modules are only rigidly linked via weld
            # equality constraints, not a kinematic tree, so this couldn't
            # be read off from a plain forward-kinematics pass - it needs
            # the real stepped-and-settled `data` the screenshot also uses).
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
                    # Same principle as the renderer-construction guard
                    # above: a failed screenshot save shouldn't cost this
                    # individual its (otherwise valid) physics results.
                    logger.exception("Failed to render/save screenshot for '%s' - continuing without it.", model_name)

        # Set the MuJoCo control callback to apply magnetic torque each step
        mujoco.set_mjcb_control(magnetic_field_callback)

        while magnets_active:
            set_angle_to_joint(model, data, target_angle_deg=target_angles)
            get_light_sensor_values(model, data)
            mujoco.mj_step(model, data)
            mujoco.mj_subtreeVel(model, data)
            step_count += 1

            # A MuJoCo warning (bad qpos/qvel/qacc from a diverging or
            # colliding model, ...) means the physics is no longer
            # trustworthy - stop immediately rather than let a headless run
            # spin through every remaining step (or hang, if the resulting
            # NaNs make the wall-distance check never trip) on garbage state.
            if int(np.sum(data.warning.number)) > 0:
                # print(f"Time: {data.time:.2f}s | MuJoCo warning triggered - aborting run.")
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

    # pheromone_yaw_response_deg / pheromone_speed_response: only
    # meaningful relative to a real "no light" baseline, which a failed or
    # unstable run doesn't have -- see include_light_tests' docstring.
    # run_headless_light_tests() measures its own dedicated baseline stage
    # internally now (see its docstring for why reusing this primary run's
    # own travel/yaw used to give an unreliable reorientation), so nothing
    # from this run's own trajectory needs to be passed in beyond the model
    # itself and where module_1's freejoint lives.
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
    b_values, picks the one with the highest average velocity among those
    that both succeeded and weren't flagged unstable, then re-runs just
    that winning B for real (with the caller's actual capture_img/
    capture_gif, and ALWAYS compute_shape_entropy_3d=True regardless of
    capture_img - see run_headless's docstring for why that's decoupled
    from capture_img now) so media/entropy is never spent on a discarded
    candidate, but shape_entropy_3d always gets computed for the one
    result that's actually kept and scored.

    Winner selection:
      1. Prefer the highest avg_velocity_mmps among physics_ok=True,
         is_stable=True runs.
      2. If none qualify, fall back to the highest avg_velocity_mmps among
         physics_ok=True runs regardless of is_stable (a sweep should never
         come back with nothing just because every candidate rocked).
      3. If every B outright failed (a MuJoCo warning fired), report the
         last attempted B's failed (all-zero) result.

    Mutates the module-level B_INTENSITY for the duration of the sweep
    (magnetic_field_callback reads it directly); always restored
    afterward, even on error.

    include_light_tests (default True, unlike run_headless()'s own default
    of False) is forwarded only to the final re-run of the winning B below
    -- never to the per-candidate sweep trials above, which are thrown away
    except for their avg_velocity/stability and shouldn't pay for it. This
    is "B picked from run_headless_b_sweep (preferred)": by the time that
    re-run happens, B_INTENSITY is already set to winner_b, so
    run_headless_light_tests() runs at the actual winning field strength,
    not the module-level default.

    light_module_rgba (default None = leave the normal body-blue color
    alone): see run_headless()'s docstring. Applied ONCE here, right after
    this function's own model load, since every candidate B AND the final
    winner re-run below all share this same compiled `model` object -
    geom_rgba is a static model property, unaffected by B_INTENSITY/
    stepping, so one recolor covers the whole sweep.
    """
    global B_INTENSITY
    original_b = B_INTENSITY

    # Every candidate B (and the winner re-run below) simulates the exact
    # same XML - only B_INTENSITY (a runtime/callback parameter, never
    # baked into the compiled model) differs between them - so compile
    # once here and hand this same `model`/`target_angles` into every
    # run_headless call instead of re-parsing the MJCF 4 times over.
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
        # Windowed velocity: displacement covered since the last 1-second
        # print, so the printout reflects actual per-interval speed instead
        # of the cumulative average-since-ramp-end (which can only creep
        # slowly toward its converged value and hides real fluctuation).
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

            # A MuJoCo warning (bad qpos/qvel/qacc from a diverging or
            # colliding model, ...) means the physics is no longer
            # trustworthy - stop feeding it to the viewer immediately
            # instead of continuing to sync() frames from a state that's
            # about to (or already did) blow up.
            if int(np.sum(data.warning.number)) > 0:
                #print(f"Time: {data.time:.2f}s | MuJoCo warning triggered - aborting run.")
                physics_ok = False
                mujoco.set_mjcb_control(None)
                data.xfrc_applied.fill(0)
                viewer.close()
                break

            # Ground-truth morphology performance metrics, tracked from the
            # whole-model subtree COM (body 0 = worldbody subtree = everything).
            # Gated on magnets_active so these freeze at the wall-stop instant
            # instead of draining toward 0 during whatever idle time passes
            # in the viewer after locomotion has actually ended.
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

            # Stop actuating once the assembly gets close to a wall, judged
            # from the on-body rangefinders (not a ground-truth position
            # check) - mirrors a real obstacle-avoidance cutoff.
            if magnets_active:
                wall_dist = distance_to_nearest_wall(model, data)
                if wall_dist is not None:
                    wall_dist *= 1000  # Convert to mm
                if wall_dist is not None and wall_dist < WALL_STOP_DISTANCE:
                    mujoco.set_mjcb_control(None)
                    # xfrc_applied is a persistent array, not reset by
                    # unregistering the callback - without this the last
                    # applied torque would keep acting on every future step.
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
    """Light-response test: same live-viewer setup as run_with_viewer (load
    model, register the magnetic-field callback, real-time-paced mj_step
    loop, per-second logging), but instead of one continuous free-roam gait
    it runs this fixed sequence:

        load xml -> settle to the folded hinge angle -> record the settled
        pose's COM as `initial_com` -> baseline (no light) -> stage "left"
        -> stage "front"

    Baseline: LIGHT_TEST_BASELINE_DURATION seconds of magnetic actuation
    with the scene's normal (directional, non-spot) light on -- not the
    fixture used by stages 1-2 -- so there's a no-light reference avg
    linear velocity to compare the lit stages against.

    Stages 1-2 both use the same floor-level "luminance sheet" fixture --
    the model's <light> repositioned via _place_pheromone_light() to
    pheromone_light_height() (just below the lowest light_sensor_* site in
    this run's own settled pose - see there for why a fixed height doesn't
    work), plus a visible bright-yellow, emissive patch (mjcf_generator.py's
    `luminance_sheet` geom + its "luminance_glow" material, shown via
    _show_luminance_sheet()) covering the same rectangle for the video.
    light_sensor_joint_* sites face -Z (mjcf_generator.py) specifically so
    they can see this ground-level fixture, matching the paper's UV-excited
    fluorescent pheromone trace this represents. Only the rectangle each
    stage covers (assembly_local_bounds()) differs:

      Stage "left": covers the left half in X, full depth in Y (symmetric
        front-to-back).

      Stage "front": covers the WHOLE body footprint, centered on it - both
        sensors lit simultaneously from the start ("both sensors detecting
        pheromone"), matching what pheromone_speed_response actually
        measures, rather than a patch ahead of the robot it has to walk
        toward.

    model.vis.headlight stays at its normal default (active) here - same as
    any other run - rather than being disabled the way an earlier version
    of this function did. mjcf_generator.py's own <light> ambient/diffuse/
    specular values are deliberately low (see LIGHT_INTENSITY_LUX's own
    comment there) specifically so headlight + this DIRECTIONAL light (no
    distance falloff, so it covers the WHOLE floor) don't add up to blow
    the floor out to solid white during "baseline" - that combination
    overexposes at the XML's old, higher cosmetic values regardless of
    run_light_tests() at all (confirmed with a plain run_headless()/
    run_with_viewer()-equivalent render, no light-test code involved), so
    the fix had to live in the model's own light values, not here.
    light_intensity itself (what get_light_sensor_values() actually reads
    for lux) is untouched either way - see LIGHT_INTENSITY_LUX.

    get_light_sensor_values()'s `light_bounds` for each stage is the SAME
    LUMINANCE_SHEET_VISUAL_SCALE-enlarged rectangle _show_luminance_sheet()
    renders (via _scale_local_bounds(), same anchor) rather than the
    smaller unscaled bounds - so the visible yellow patch IS the actual
    trigger/release cutoff, not an approximation of it.

    Every stage (baseline included) resets the assembly back to the settled
    pose (the full qpos, so shape and orientation reset too, not only
    position) before running, so each is an independent trial from the same
    starting state rather than picking up wherever the previous one left
    off.

    MuJoCo's light count is fixed at compile time, so the light stages
    repoint the model's one existing <light> rather than adding a second.

    Metrics collected per stage (see run_stage()):
      - Every stage: avg linear velocity (net COM displacement / elapsed
        stage time).
      - "baseline" and "left": net yaw rotation of module_1's own freejoint
        (module_1 is this design's designated sensor/control module -- see
        its rangefinder/IMU comment further down) -- skipped with a warning
        if the model has no "free_module_1" joint.
    The final summary (printed, not written to stats_output_path) reports
    pheromone_yaw_response_deg/pheromone_speed_response computed the same
    way as objectives_api.py's f6/f7 (see run_headless_light_tests), but
    against this run's own freshly-measured baseline stage.

    get_light_sensor_values() is called once per simulated second in each
    stage (prints only -- per-step light-sensor readings are not written to
    stats_output_path).

    capture_video=True additionally records every stage (baseline, "left",
    "front") through a SECOND, offscreen mujoco.Renderer -- same approach as
    run_headless()'s GIF capture, just independent of the live viewer window
    (and its real-time pacing/pause/close) so recording never depends on
    what's on screen. Frames are captured at video_fps and streamed straight
    to disk via an imageio ffmpeg writer (H.264, yuv420p, quality=10 -- the
    top of imageio's 0-10 scale, which maps to -crf 0, i.e. lossless) rather
    than buffered in memory, since a VIDEO_RENDER_WIDTH x VIDEO_RENDER_HEIGHT
    run of this length would otherwise be several GB of raw frames. Saved to
    f"{media_dir}/light_tests_{model_name}.mp4". A renderer/writer failure
    (e.g. no GPU/EGL context available) only costs the video, exactly like
    run_headless()'s screenshot/GIF guard -- the physics and printed summary
    below are unaffected.

    light_module_rgba (default None = leave the normal body-blue color
    alone): see run_headless()'s docstring - handy here specifically,
    since this is the test that's actually exercising the light-sensitive
    joint, so knowing which module that is at a glance in the viewer/video
    is often the point of watching this run.
    """
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

    # module_1 is this design's designated sensor/control module (see
    # build_module_element()'s IMU/rangefinder comment in mjcf_generator.py)
    # -- used here as the reference body for yaw tracking.
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

    # ---- offscreen video capture (independent of the live viewer window -
    # see docstring). A renderer/writer construction failure only costs the
    # video, never the light tests themselves. ----
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
        # Closing the video writer/renderer via ExitStack (rather than a
        # try/finally around the whole function) means they get cleaned up
        # on every exit path out of this "with" -- including the early
        # `return`s in the settle loop below -- without having to wrap (and
        # re-indent) the entire viewer session.
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

        # Floor-level "luminance sheet" light height for "left"/"front"
        # below (see pheromone_light_height()) - computed once from this
        # settled pose's own light_sensor_* heights, same for every stage.
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
            """Runs `duration` seconds of magnetic actuation (light_bounds
            None means no light-triggered reactive fold, and if the scene
            light is also inactive -- see baseline below -- this is a
            genuine no-light run). Always returns avg linear velocity (net
            COM displacement / elapsed stage time); with track_angular=True
            and a resolved module_1 joint, also returns module_1's avg
            angular speed and net yaw rotation over the stage.

            track_acceleration=True additionally returns avg_accel_y_mps2 =
            (v_end_y - v_start_y) / elapsed, where v_end_y is the signed
            Y-velocity (+Y is "front", see the module docstring) at stage
            end. reset_to_initial_pose() zeros qvel, so without
            v_start_override this stage's own v_start_y would trivially be
            0 -- pass v_start_override to use that as v_start_y instead of
            capturing it fresh from the current (just-reset, at-rest) state.
            The "front" stage call below passes the baseline (no-light)
            stage's own avg_linear_mps for this -- note that's an unsigned
            speed (net displacement / time), not a signed Y-velocity like
            v_end_y, a deliberate choice: it's the assembly's actual
            lightless cruising speed, standing in for "however fast it was
            already going" the way avg_accel_y_mps2 is meant to be read.
            """
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
            # Net XY displacement vector (not just speed magnitude) - lets
            # the caller check whether the robot actually moved TOWARD or
            # AWAY from the light's own position, independent of module_1's
            # own yaw rotation (a body can rotate to face the light while
            # its net translation still drifts the other way, e.g. from a
            # non-driving-wheel-like stick-slip gait).
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

        # ---- Baseline: move with the scene's normal (directional, non-spot)
        # light on, before any spotlight stage ----
        # Its avg_linear_mps becomes the front stage's v_start_override
        # below, so that stage's acceleration is measured from the
        # assembly's actual no-spotlight cruising speed, not from an
        # artificial at-rest 0 (reset_to_initial_pose() zeros qvel).
        model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        model.light_active[light_id] = 1
        reset_to_initial_pose()
        results["baseline"] = run_stage("baseline (no light)", LIGHT_TEST_BASELINE_DURATION, track_angular=True)

        # ---- Travel-aligned local frame from the baseline stage's own
        # measured direction (falls back to world +Y if that travel was too
        # small to trust). The robot's qpos is NEVER rotated to face
        # anything - physically rotating an already-settled, contact-
        # consistent pose turned out to risk tipping a marginal design over
        # before the light stages even start. Only the light-patch bounds/
        # positions below are expressed in this frame (see
        # assembly_local_bounds/world_to_local_xy/local_to_world_xy) -
        # run_headless_light_tests() uses the identical approach.
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

        # Purely cosmetic: point the VIEWER CAMERA so the robot appears to
        # walk toward it during "left" below - this only moves the
        # camera, never the robot's own pose (unlike the old camera-facing
        # reorientation this replaced, which rotated the robot instead and
        # risked tipping a marginal design over). MuJoCo's free-camera
        # convention: campos = lookat - distance*(cos(el)cos(az), cos(el)sin(az), sin(el)),
        # so the floor-projected viewing direction (camera -> lookat) is
        # proportional to (cos(el)cos(az), cos(el)sin(az)) - azimuth is
        # chosen so that direction is -forward (i.e. the robot's travel
        # direction points from the scene toward the camera).
        viewer.cam.lookat[:2] = origin_xy
        viewer.cam.azimuth = float(np.degrees(np.arctan2(-forward[1], -forward[0])))

        # Back onto the (untouched) settled pose before measuring bounds,
        # so every stage below is sized/positioned against the same
        # reference pose it will actually reset to.
        reset_to_initial_pose()

        # Stage bounds all come from the same settled-pose bounding box;
        # only which half/edge each stage targets differs.
        xy_min, xy_max = assembly_local_bounds(model, data, origin_xy, forward)
        mid_x = (xy_min[0] + xy_max[0]) / 2.0
        mid_y = (xy_min[1] + xy_max[1]) / 2.0

        def _pheromone_sheet(center_local_xy, bounds, anchor_local_xy):
            # Floor-level "luminance sheet" (real light repositioned via
            # _place_pheromone_light() + the visible yellow patch shown via
            # _show_luminance_sheet()) - used by every stage below, see
            # pheromone_light_height(). anchor_local_xy: see
            # _show_luminance_sheet()'s docstring - the point LUMINANCE_
            # SHEET_VISUAL_SCALE grows the visible patch away from, so it
            # never bleeds into the robot's other half/its own body.
            center_xy = local_to_world_xy(center_local_xy, origin_xy, forward)
            _place_pheromone_light(model, light_id, center_xy, pheromone_light_z)
            _show_luminance_sheet(model, sheet_geom_id, bounds, anchor_local_xy, origin_xy, forward)
            # data.light_xpos (what get_light_sensor_values() actually
            # reads) is a *derived* kinematic quantity - it won't reflect
            # this new model.light_pos until forward kinematics runs again,
            # so without this, run_stage()'s very first iteration would
            # read the PREVIOUS stage's light position for one step.
            mujoco.mj_forward(model, data)
            return center_xy

        # ---- Stage 1: luminance sheet covering only the left half of the body, symmetrically ----
        reset_to_initial_pose()
        left_bounds = (xy_min[0], mid_x, xy_min[1], xy_max[1])  # left half in X, full depth in Y
        left_center = ((xy_min[0] + mid_x) / 2.0 - LIGHT_TEST_LEFT_EXTRA_OFFSET, mid_y)
        # Pin the inner edge (mid_x, the boundary against the robot's own
        # right half) so a bigger LUMINANCE_SHEET_VISUAL_SCALE only grows
        # the patch further left, never toward/past the robot's center.
        left_light_xy = _pheromone_sheet(left_center, left_bounds, (mid_x, mid_y))
        # The actual trigger/release check gets the SAME scaled-up
        # rectangle _show_luminance_sheet() just rendered (same anchor, same
        # LUMINANCE_SHEET_VISUAL_SCALE via _scale_local_bounds()) rather
        # than the smaller unscaled `left_bounds` - otherwise a sensor could
        # visibly still be inside the yellow patch while the (much smaller)
        # numeric bounds had already released it.
        left_sensed_bounds = _scale_local_bounds(left_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
        results["left"] = run_stage("left", light_test_duration, light_bounds={light_id: left_sensed_bounds},
                                     track_angular=True, frame=frame)

        # ---- Stage 2: same luminance sheet, covering the WHOLE body
        # footprint (centered on it) instead of a patch ahead of it - both
        # sensors are lit simultaneously from the start ("both sensors
        # detecting pheromone"), matching what pheromone_speed_response is
        # actually meant to measure, rather than a "walk toward a light
        # ahead" scenario. ----
        reset_to_initial_pose()
        front_bounds = (xy_min[0], xy_max[0], xy_min[1], xy_max[1])  # the whole body footprint
        front_center = (mid_x, mid_y)
        # anchor == bounds' own center here (the body's own center) since
        # there's no "inner edge" to protect anymore - a bigger
        # LUMINANCE_SHEET_VISUAL_SCALE just grows outward on all sides.
        front_light_xy = _pheromone_sheet(front_center, front_bounds, (mid_x, mid_y))
        front_sensed_bounds = _scale_local_bounds(front_bounds, (mid_x, mid_y), LUMINANCE_SHEET_VISUAL_SCALE)
        results["front"] = run_stage("front", light_test_duration, light_bounds={light_id: front_sensed_bounds},
                                      track_acceleration=True,
                                      v_start_override=results["baseline"]["avg_linear_mps"], frame=frame)

    mujoco.set_mjcb_control(None)

    # pheromone_yaw_response_deg / pheromone_speed_response, computed the
    # same way as objectives_api.py's f6/f7 (see run_headless_light_tests),
    # but against THIS run's own freshly-measured no-light baseline stage
    # rather than the primary gait rollout's baseline stand-in - so these
    # numbers are the more direct measurement, not expected to match
    # ind{i}_stats.json's f6/f7 exactly (see run_headless_light_tests'
    # docstring for why it cuts a dedicated baseline stage there).
    baseline, left, front = results["baseline"], results["left"], results["front"]
    baseline_yaw = baseline.get("yaw_rotation_deg", 0.0)
    # Re-wrap the difference of two already-wrapped angles into (-180, 180]
    # - without this, a pair straddling the wraparound boundary (e.g.
    # left=+170, baseline=-170) reports +340 instead of the true -20.
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

    # Net TRANSLATION toward/away from the light, independent of
    # module_1's own yaw rotation - the two can disagree (e.g. the body
    # rotates to face the light while its net stick-slip drift still
    # carries it the other way), which is exactly what yaw alone can't
    # catch. dot(displacement, direction-to-light) > 0 means the stage's
    # net COM movement had a component toward the light, not just that the
    # body's orientation ended up facing it.
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