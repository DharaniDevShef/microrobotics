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
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

import entropy_api

# Global dictionaries mapping Parent Body ID -> List of (Magnet Geom ID, Polarity Sign)
parent_body_magnet_map = {}

# Torque tracking global variables
torque_history = []

# B-field tracking global variables (paired with torque_history, one entry per step)
b_field_history = []
time_history = []

logger = logging.getLogger(__name__)

# Physics constants
# Magnetic field intensity (Tesla)
B_INTENSITY = 0.05  # 10 mT
# Candidate drive strengths for run_headless_b_sweep() -- the field a
# morphology needs to overcome stiction and walk (rather than stall, or
# over-drive into a rolling/tumbling gait) is morphology-dependent, so
# this is swept per run rather than assumed fixed.
B_SWEEP_VALUES = (0.001, 0.008, 0.05)
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
# displays these at a couple hundred px wide (see SCREENSHOT_CARD_WIDTH in
# evolution_results_visualizer.py), so 1920x1080 was pure waste -
# needlessly heavy GPU/EGL context memory per renderer, which under
# sim_executor.py's parallel subprocesses meant several large contexts
# competing at once - a plausible source of the occasional renderer
# failure some individuals were hitting.
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

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

# Duration (s) of magnetic actuation applied per run_light_tests() stage.
LIGHT_TEST_DEFAULT_DURATION = 10
# Default standoff distance (m) for the light source in each run_light_tests()
# stage (how far to the left / how far above the initial COM it's placed).
LIGHT_TEST_DEFAULT_DISTANCE = 0.1
# How long (s) run_light_tests() moves the assembly with no light active at
# all, before the light stages start, to get a baseline avg linear velocity.
LIGHT_TEST_BASELINE_DURATION = 7.0
# Extra distance (m) each stage's light is pushed beyond its half-region's
# own centroid, further out to that side -- e.g. "left" stage's light sits
# this much further left (more negative X) than the left half's midpoint.
LIGHT_TEST_LEFT_EXTRA_OFFSET = 0.01
LIGHT_TEST_FRONT_EXTRA_OFFSET = 0
# Minimum net baseline (no-light) displacement (m) run_light_tests() needs
# before its direction is trustworthy enough to reorient the assembly by --
# below this, "which way did it travel" is dominated by settle/numerical
# noise, not a real heading, so the camera-facing reorientation is skipped.
LIGHT_TEST_MIN_BASELINE_TRAVEL = 0.0005
# run_light_tests() reactive fold: a joint whose own light_sensor_joint_*
# reading exceeds this (lux) gets driven to LIGHT_TRIGGER_ANGLE_DEG -- every
# other joint is left exactly as it was.
LIGHT_TRIGGER_LUX_THRESHOLD = 10000.0
LIGHT_TRIGGER_ANGLE_DEG = 60.0


def magnetic_field_callback(model, data):
    """
    MuJoCo Control Callback.
    Calculates magnetic torque:
        Tau = M x B
    Applies torque to parent module bodies and records net applied torque magnitude.
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

    # Magnetic field along Z only
    b_vector = np.array([0.0, 0.0, b_z])

    step_total_torque = 0.0

    # Apply magnetic torque
    for parent_body_id, magnet_list in parent_body_magnet_map.items():
        accumulated_torque = np.zeros(3)
        for geom_id, polarity_sign in magnet_list:
            # Magnet world orientation
            geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
            # Magnet dipole along local Z-axis
            local_m = np.array([0.0, 0.0, 1.0]) * M_MOMENT * polarity_sign
            # Convert dipole to world frame
            world_m = geom_mat.dot(local_m)
            # Tau = M x B
            torque_vector = np.cross(world_m, b_vector)
            accumulated_torque += torque_vector

        ramped_torque = accumulated_torque * effective_torque_multiplier
        data.xfrc_applied[parent_body_id][3:6] = ramped_torque
        step_total_torque += ramped_torque

    # Record magnitude of total torque on whole body for this step
    torque_history.append(np.linalg.norm(step_total_torque))
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


def set_angle_to_joint(model, data, target_angle_deg, light_bounds=None, light_target_angles=None):
    """Set actuator position targets from a scalar, ordered sequence, or name map.

    light_target_angles: optional {"light_ctrl_jointN": angle_deg} map
    (roblet_simulator.read_light_target_angles_from_xml) - the evolved
    Design Variable 6 (hinge_angle_on_light_detection) for each
    light-sensitive joint. When a joint's own light_sensor_* reading
    exceeds LIGHT_TRIGGER_LUX_THRESHOLD, its actuator target is overridden
    to THIS joint's own evolved value if present, falling back to the flat
    LIGHT_TRIGGER_ANGLE_DEG default otherwise (e.g. a hand-built model with
    no evolved metadata at all)."""
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

        readings = get_light_sensor_values(model, data, verbose=False, light_bounds=light_bounds)
        for site_name, lux in readings.items():
            actuator_idx = joint_to_actuator.get(site_name.replace("light_sensor_", "", 1))
            if actuator_idx is not None and lux > LIGHT_TRIGGER_LUX_THRESHOLD:
                trigger_angle = LIGHT_TRIGGER_ANGLE_DEG
                if light_target_angles:
                    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_idx)
                    light_key = (actuator_name or "").replace("ctrl_joint", "light_ctrl_joint", 1)
                    trigger_angle = light_target_angles.get(light_key, LIGHT_TRIGGER_ANGLE_DEG)
                target_values[actuator_idx] = trigger_angle

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


def assembly_xy_bounds(model, data):
    """Axis-aligned world-XY bounding box, as (mins, maxs) each [x, y], of
    every robot geom -- i.e. every geom whose body is *not* the worldbody
    (body 0). The floor and perimeter walls are declared directly under
    <worldbody> in mjcf_generator.py (no wrapping <body>), so they're body 0
    too and get excluded automatically; every module_* geom is nested under
    its own body, so it's included. Used to size/center a ceiling light's
    square coverage footprint over a specific half of the assembly.
    """
    mins = np.array([np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf])
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        pos_xy = data.geom_xpos[g][:2]
        r = model.geom_rbound[g]
        mins = np.minimum(mins, pos_xy - r)
        maxs = np.maximum(maxs, pos_xy + r)
    return mins, maxs


def get_light_sensor_values(model, data, verbose=True, light_bounds=None):
    """Illuminance (lux) at each on-body light sensor (the `light_sensor_*`
    sites from mjcf_generator.py), summed over every <light> in the scene.

    verbose=False skips the print (e.g. when this is called every step for
    up-to-date values but only needs to print occasionally).

    light_bounds: optional {light_id: (x_lo, x_hi, y_lo, y_hi)} restricting
    a light to a rectangular world-XY footprint -- e.g. a square ceiling
    panel that only covers one half of the assembly (see run_light_tests).
    A sensor outside that footprint reads 0 lux from that light regardless
    of angle or occlusion. Lights not present in the dict are unrestricted.

    MuJoCo has no native illuminance sensor/API -- <light> is a
    rendering-only construct, so this is a hand-rolled photometric estimate
    built from MuJoCo's own light data (mj_ray for occlusion, plus each
    light's geometry and its `intensity` field, read straight from the
    model rather than a duplicated Python constant -- see
    LIGHT_INTENSITY_LUX in mjcf_generator.py):

      lux = intensity * max(0, cos(theta)) / falloff   (0 if occluded)

    where theta is the angle between the sensor panel's own outward normal
    (its owning body's local Z axis -- the site's Z is a cosmetic rotation
    used only to lay its marker geom flat, see below) and the direction to
    the light (Lambert's cosine law), and falloff is 1 for a directional
    light (parallel rays, no distance attenuation) or distance^2 for a
    positional light (inverse-square law). bodyexclude on the occlusion ray
    keeps a sensor from immediately re-hitting the surface it's mounted on.
    """
    geomid = np.zeros(1, dtype=np.int32)
    readings = {}
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not name or not name.startswith("light_sensor_"):
            continue
        pnt = data.site_xpos[site_id]
        body_id = model.site_bodyid[site_id]
        normal = data.xmat[body_id].reshape(3, 3)[:, 2]

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
                if not (x_lo <= pnt[0] <= x_hi and y_lo <= pnt[1] <= y_hi):
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

            hit_dist = mujoco.mj_ray(model, data, pnt, to_light, None, True, body_id, geomid)
            if hit_dist >= 0 and hit_dist < light_dist:
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
    return camera


def run_headless_light_tests(
    model, target_angles, travel_xy, module1_qpos_adr, module1_dof_adr,
    baseline_avg_velocity, baseline_yaw_rotation_deg,
    light_test_duration=LIGHT_TEST_DEFAULT_DURATION, light_distance=LIGHT_TEST_DEFAULT_DISTANCE,
    light_target_angles=None,
):
    """Headless companion to run_light_tests(): computes
    pheromone_yaw_response_deg and pheromone_speed_response for
    run_headless()'s include_light_tests=True path.

    Two things are cut relative to the interactive run_light_tests() to
    keep this cheap enough to call from every successful evaluation in an
    evolutionary run:

      - No separate no-light "baseline" stage, and no "right" stage --
        only "left" and "front" run. The "no light" reference these two
        new metrics are measured against is instead the primary run's own
        already-computed avg velocity and module_1 yaw rotation
        (baseline_avg_velocity / baseline_yaw_rotation_deg), and the
        reorientation below uses that same run's travel_xy -- all passed
        in by run_headless() rather than re-measured here, since
        run_headless() already paid for that simulated time once.
      - No viewer, so nothing is real-time paced, printed, or camera-
        facing -- "front" is fixed at world +Y (the same convention
        left_bounds/front_bounds already use), and travel_xy is rotated
        onto that instead of onto a camera direction.

    `model` is reused as-is (no reload from disk -- the caller already
    paid the compile cost); a fresh MjData is created so this starts from
    a clean simulation state regardless of whatever run_headless()'s own
    gait left the model in.

    light_target_angles (roblet_simulator.read_light_target_angles_from_xml)
    is this model's own evolved Design Variable 6
    (hinge_angle_on_light_detection) per light-sensitive joint -- forwarded
    into set_angle_to_joint()'s reactive-fold branch below so a triggered
    joint moves to ITS OWN evolved trigger angle, not the flat
    LIGHT_TRIGGER_ANGLE_DEG default.

    Returns {"pheromone_yaw_response_deg": ..., "pheromone_speed_response": ...}
    -- pheromone_yaw_response_deg = yaw_rotation(left stage) -
    baseline_yaw_rotation_deg (degrees); pheromone_speed_response =
    (avg_linear_mps(front stage) - baseline_avg_velocity) /
    baseline_avg_velocity (dimensionless; negative = slower than baseline
    (deceleration), positive = faster (acceleration)). If
    baseline_avg_velocity is 0, pheromone_speed_response is reported as 0.0
    rather than dividing by zero.
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

    # ---- reorient using the primary run's own travel_xy, target = world
    # +Y (the fixed "front" convention left_bounds/front_bounds already
    # use below -- there's no viewer/camera to face here) ----
    travel_dist = float(np.linalg.norm(travel_xy))
    if travel_dist >= LIGHT_TEST_MIN_BASELINE_TRAVEL:
        travel_dir = travel_xy / travel_dist
        target_dir = np.array([0.0, 1.0])
        angle = float(np.arctan2(
            travel_dir[0] * target_dir[1] - travel_dir[1] * target_dir[0],
            travel_dir[0] * target_dir[0] + travel_dir[1] * target_dir[1],
        ))
        c, s = np.cos(angle), np.sin(angle)
        q_rot = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
        new_quat = np.zeros(4)
        for body_id in find_module_labels(model):
            joint_id = model.body_jntadr[body_id]
            qadr = model.jnt_qposadr[joint_id]
            x, y = initial_qpos[qadr], initial_qpos[qadr + 1]
            dx, dy = x - initial_com[0], y - initial_com[1]
            initial_qpos[qadr] = initial_com[0] + dx * c - dy * s
            initial_qpos[qadr + 1] = initial_com[1] + dx * s + dy * c
            mujoco.mju_mulQuat(new_quat, q_rot, initial_qpos[qadr + 3:qadr + 7])
            initial_qpos[qadr + 3:qadr + 7] = new_quat
    else:
        logger.debug(
            "run_headless_light_tests(): primary run's travel distance too small (%.4f mm) to reorient by.",
            travel_dist * 1000,
        )

    reset_to_initial_pose()

    xy_min, xy_max = assembly_xy_bounds(model, data)
    mid_x = (xy_min[0] + xy_max[0]) / 2.0
    mid_y = (xy_min[1] + xy_max[1]) / 2.0

    def spot_ceiling_light(center_xy):
        # Only light_type/pos/dir matter here -- get_light_sensor_values()
        # never reads light_cutoff/diffuse/specular, and there's no
        # renderer for them to matter to either.
        model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_SPOT
        model.light_pos[light_id] = np.array([center_xy[0], center_xy[1], initial_com[2] + light_distance])
        model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])

    def run_stage(duration, light_bounds, track_angular):
        com_start = get_com_position(data)
        yaw_start = module1_yaw_deg() if track_angular else None
        mujoco.set_mjcb_control(magnetic_field_callback)
        while data.time < duration:
            set_angle_to_joint(model, data, target_angle_deg=target_angles, light_bounds=light_bounds,
                                light_target_angles=light_target_angles)
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

    # ---- "left" stage ----
    reset_to_initial_pose()
    left_bounds = (xy_min[0], mid_x, xy_min[1], xy_max[1])
    spot_ceiling_light(((xy_min[0] + mid_x) / 2.0 - LIGHT_TEST_LEFT_EXTRA_OFFSET, mid_y))
    left_result = run_stage(light_test_duration, {light_id: left_bounds}, track_angular=True)

    # ---- "front" stage ----
    reset_to_initial_pose()
    front_bounds = (xy_min[0], xy_max[0], mid_y, xy_max[1])
    spot_ceiling_light((mid_x, (mid_y + xy_max[1]) / 2.0 + LIGHT_TEST_FRONT_EXTRA_OFFSET))
    front_result = run_stage(light_test_duration, {light_id: front_bounds}, track_angular=False)

    mujoco.set_mjcb_control(None)

    pheromone_yaw_response_deg = left_result["yaw_rotation_deg"] - baseline_yaw_rotation_deg
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
    include_light_tests=False,
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
    already-loaded model plus its own avg_velocity and module_1 yaw
    rotation as the "no light" baseline those two new stats are measured
    against -- so this only ever adds run_headless_light_tests()'s own
    settle + "left" + "front" stages on top, never a redundant repeat of
    the primary gait. Left False by default so run_headless_b_sweep()'s
    per-candidate-B trial runs (which get thrown away except for their
    avg_velocity) don't pay for it -- only its final re-run of the winning
    B does. Skipped entirely (both new stats left at 0.0) whenever this
    run's own success is 0 -- see below -- since there's no meaningful "no
    light" baseline to compare a light response against otherwise. Uses
    whatever the module-level B_INTENSITY currently is, same as this run's
    own gait (the global default for a standalone call, or the sweep's
    winning B when called from run_headless_b_sweep()'s final re-run).
    """
    parent_body_magnet_map.clear()
    torque_history.clear()
    b_field_history.clear()
    time_history.clear()

    if model is None:
        model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.01

    if target_angles is None:
        target_angles = read_joint_target_angles_from_xml(model_path)
    if not target_angles:
        target_angles = [45.0] * model.nu
    if light_target_angles is None:
        light_target_angles = read_light_target_angles_from_xml(model_path)

    parent_body_magnet_map.update(find_all_magnets(model))
    module_labels = find_module_labels(model)

    # module_1 is this design's designated sensor/control module (see
    # build_module_element()'s IMU/rangefinder comment in mjcf_generator.py)
    # -- used as the reference body for pheromone_yaw_response_deg's
    # "no light" baseline yaw rotation below.
    module1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_module_1")
    if module1_joint_id < 0:
        module1_qpos_adr = module1_dof_adr = None
    else:
        module1_qpos_adr = model.jnt_qposadr[module1_joint_id]
        module1_dof_adr = model.jnt_dofadr[module1_joint_id]

    def module1_yaw_deg():
        qw, qx, qy, qz = data.qpos[module1_qpos_adr + 3: module1_qpos_adr + 7]
        return float(np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))

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
    steady_state_yaw_deg = None  # captured alongside steady_state_displacement, same windowing
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
        # shape entropy below) - kept on `capture_img` alone, not
        # `capture_img and renderer is not None`, so a renderer/GPU
        # failure only costs the screenshot, never the entropy computation
        # (they need the same settled pose, but are otherwise independent).
        if capture_img:
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

            if physics_ok and renderer is not None:
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
                if module1_qpos_adr is not None:
                    steady_state_yaw_deg = module1_yaw_deg()
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
    pheromone_yaw_response_deg = 0.0
    pheromone_speed_response = 0.0
    if include_light_tests and success:
        baseline_yaw_rotation_deg = 0.0
        if module1_qpos_adr is not None and steady_state_yaw_deg is not None:
            final_yaw_deg = module1_yaw_deg()
            baseline_yaw_rotation_deg = ((final_yaw_deg - steady_state_yaw_deg + 180) % 360) - 180
        light_test_results = run_headless_light_tests(
            model, target_angles, com[:2] - initial_com[:2], module1_qpos_adr, module1_dof_adr,
            baseline_avg_velocity=avg_velocity, baseline_yaw_rotation_deg=baseline_yaw_rotation_deg,
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
    include_light_tests=True,
):
    """Runs one fast (no media) run_headless() rollout per candidate B in
    b_values, picks the one with the highest average velocity among those
    that both succeeded and weren't flagged unstable, then re-runs just
    that winning B for real (with the caller's actual capture_img/
    capture_gif) so media is never spent rendering a discarded candidate.

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
    """
    global B_INTENSITY
    original_b = B_INTENSITY

    # Every candidate B (and the winner re-run below) simulates the exact
    # same XML - only B_INTENSITY (a runtime/callback parameter, never
    # baked into the compiled model) differs between them - so compile
    # once here and hand this same `model`/`target_angles` into every
    # run_headless call instead of re-parsing the MJCF 4 times over.
    model = mujoco.MjModel.from_xml_path(model_path)
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
                include_light_tests=include_light_tests,
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


def run_with_viewer(model_path, stats_output_path, max_sim_time=None):
    """Same closed-loop simulation as run_headless, but with the live passive
    viewer and real-time pacing so you can watch it run.

    max_sim_time is in seconds of simulated time (data.time), same as
    run_headless - once reached, magnet actuation stops (same as reaching a
    wall) but the viewer stays open so you can still inspect the final pose.
    """
    if not os.path.exists(model_path):
        logger.error("Could not find '%s'", model_path)
        return
    logger.info("Loading model: %s...", model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
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

    # Identify module bodies and assign their text labels
    module_labels = find_module_labels(model)

    # Register callback
    mujoco.set_mjcb_control(magnetic_field_callback)

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1  # zoom
        viewer.cam.lookat[:] = [0, 0, 0]
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
                     light_distance=LIGHT_TEST_DEFAULT_DISTANCE):
    """Light-response test: same live-viewer setup as run_with_viewer (load
    model, register the magnetic-field callback, real-time-paced mj_step
    loop, per-second logging), but instead of one continuous free-roam gait
    it runs this fixed sequence:

        load xml -> settle to the folded hinge angle -> record the settled
        pose's COM as `initial_com` -> baseline (no light) -> stage "left"
        -> stage "right" -> stage "front"

    Baseline: LIGHT_TEST_BASELINE_DURATION seconds of magnetic actuation
    with the scene light off entirely (model.light_active = 0), so there's
    a no-light reference avg linear velocity to compare the lit stages
    against.

    Stages 1-3 all use the same fixture: a real MuJoCo spotlight --
    positioned `light_distance` m straight above the assembly (so
    get_light_sensor_values()'s inverse-square falloff over that height is
    actually exercised) and pointed straight down, never from the side --
    with its cutoff cone sized to just cover one symmetric half of the
    assembly's own footprint (assembly_xy_bounds()):

      Stage "left":  cone covers the left half in X, full depth in Y
        (symmetric front-to-back) -- i.e. a panel over just the left half.

      Stage "right": cone covers the right half in X, full depth in Y --
        the mirror image of "left".

      Stage "front": cone covers the front half in Y (+Y is taken as
        "front"), full width in X (symmetric left-to-right) -- the same
        fixture, moved to cover the front half instead.

    Two things make this an *actual* lit/unlit difference, visible in the
    live viewer, rather than the printed lux numbers being the only signal:

      1. model.vis.headlight is disabled for this run. MuJoCo's viewer
         normally adds an automatic camera-attached headlight regardless of
         any <light> in the scene, which would otherwise wash out whatever
         this spotlight does -- with it off, the spotlight is the only
         thing illuminating the model.
      2. get_light_sensor_values()'s `light_bounds` still applies the exact
         rectangular half as a hard cutoff on the *numeric* lux reading (a
         circular spot cone can't perfectly match a straight-edged half, so
         the visual cone is sized to just cover that half's farthest corner
         -- close to, but not pixel-identical to, the measured cutoff).

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
      - "left" and "right" only: avg angular velocity and net yaw rotation
        of module_1's own freejoint (module_1 is this design's designated
        sensor/control module -- see its rangefinder/IMU comment further
        down) -- skipped with a warning if the model has no "free_module_1"
        joint.
    All of it is printed in one summary after every stage completes, not
    written to stats_output_path.

    get_light_sensor_values() is called once per simulated second in each
    stage (prints only -- per-step light-sensor readings are not written to
    stats_output_path).
    """
    if not os.path.exists(model_path):
        logger.error("Could not find '%s'", model_path)
        return
    logger.info("Loading model: %s...", model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.01
    # See docstring: without this, the viewer's automatic camera headlight
    # illuminates everything uniformly and the spotlight below has no
    # visible effect.
    model.vis.headlight.active = 0

    if model.nlight == 0:
        logger.error("Model has no <light> for run_light_tests() to repoint.")
        return
    light_id = 0

    # module_1 is this design's designated sensor/control module (see
    # build_module_element()'s IMU/rangefinder comment in mjcf_generator.py)
    # -- used here as the reference body for angular velocity/yaw tracking.
    module1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_module_1")
    if module1_joint_id < 0:
        logger.warning("No 'free_module_1' joint found - angular velocity/yaw tracking will be skipped.")
        module1_qpos_adr = module1_dof_adr = None
    else:
        module1_qpos_adr = model.jnt_qposadr[module1_joint_id]
        module1_dof_adr = model.jnt_dofadr[module1_joint_id]

    def module1_yaw_deg():
        qw, qx, qy, qz = data.qpos[module1_qpos_adr + 3: module1_qpos_adr + 7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(np.degrees(yaw))

    def module1_angular_speed_degps():
        wx, wy, wz = data.qvel[module1_dof_adr + 3: module1_dof_adr + 6]
        return float(np.degrees(np.linalg.norm([wx, wy, wz])))

    target_angles = read_joint_target_angles_from_xml(model_path)
    if not target_angles:
        target_angles = [45.0] * model.nu
    light_target_angles = read_light_target_angles_from_xml(model_path)

    parent_body_magnet_map.clear()
    parent_body_magnet_map.update(find_all_magnets(model))

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1
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

        def reset_to_initial_pose():
            data.qpos[:] = initial_qpos
            data.qvel[:] = 0
            data.time = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()

        def run_stage(stage_name, duration, light_bounds=None, track_angular=False, track_acceleration=False,
                      v_start_override=None):
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
            angspeed_samples = [] if track_angular else None
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
                                    light_target_angles=light_target_angles)
                mujoco.mj_step(model, data)
                mujoco.mj_subtreeVel(model, data)

                if int(np.sum(data.warning.number)) > 0:
                    logger.error("MuJoCo warning during '%s' stage - stopping it early.", stage_name)
                    break

                if track_angular:
                    angspeed_samples.append(module1_angular_speed_degps())

                current_second = int(data.time)
                if current_second != last_print:
                    last_print = current_second
                    if light_bounds is not None:
                        get_light_sensor_values(model, data, light_bounds=light_bounds)

                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            mujoco.set_mjcb_control(None)
            data.xfrc_applied.fill(0)

            elapsed = data.time if data.time > 0 else duration
            displacement = float(np.linalg.norm(get_com_position(data) - com_start))
            avg_linear_mps = displacement / elapsed if elapsed > 0 else 0.0
            logger.info("Light test stage '%s' done. Avg linear velocity: %.3f mm/s",
                        stage_name, avg_linear_mps * 1000)
            result = {"avg_linear_mps": avg_linear_mps}

            if track_angular:
                yaw_diff = ((module1_yaw_deg() - yaw_start + 180) % 360) - 180
                result["avg_angular_degps"] = float(np.mean(angspeed_samples)) if angspeed_samples else 0.0
                result["yaw_rotation_deg"] = float(yaw_diff)
                logger.info("Light test stage '%s': avg angular velocity %.2f deg/s, net yaw rotation %.2f deg",
                            stage_name, result["avg_angular_degps"], result["yaw_rotation_deg"])

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

        # ---- Baseline: move with no light active at all, before any light stage ----
        # Its avg_linear_mps becomes the front stage's v_start_override
        # below, so that stage's acceleration is measured from the
        # assembly's actual lightless cruising speed, not from an
        # artificial at-rest 0 (reset_to_initial_pose() zeros qvel).
        model.light_active[light_id] = 0
        reset_to_initial_pose()
        results["baseline"] = run_stage("baseline (no light)", LIGHT_TEST_BASELINE_DURATION)
        model.light_active[light_id] = 1

        # ---- Reorient the settled pose (once, from the baseline direction)
        # so the assembly's lightless direction of travel faces the live
        # viewer camera -- every stage from here on (including baseline's
        # own reset target for reference, though baseline itself already
        # ran) resets to this reoriented pose instead of the original one.
        baseline_end_com = get_com_position(data)
        travel_xy = baseline_end_com[:2] - initial_com[:2]
        travel_dist = float(np.linalg.norm(travel_xy))
        if travel_dist < LIGHT_TEST_MIN_BASELINE_TRAVEL:
            logger.warning(
                "Baseline moved only %.4f mm (< %.4f mm) - direction of travel isn't "
                "well-defined, skipping camera-facing reorientation.",
                travel_dist * 1000, LIGHT_TEST_MIN_BASELINE_TRAVEL * 1000,
            )
        else:
            travel_dir = travel_xy / travel_dist

            # "Head" = whichever module ends up furthest along the baseline
            # direction of travel, not a fixed module_1 -- logged for
            # visibility, though the rotation itself only needs travel_dir.
            module_labels = find_module_labels(model)
            head_name, head_proj = None, -np.inf
            for body_id, name in module_labels.items():
                proj = float(np.dot(data.xpos[body_id][:2] - initial_com[:2], travel_dir))
                if proj > head_proj:
                    head_name, head_proj = name, proj

            # Direction from the assembly to the live camera (MuJoCo free-camera
            # convention: campos = lookat - distance*(cos(el)cos(az), cos(el)sin(az), sin(el)),
            # verified directly against mjv_updateScene while building this),
            # projected onto the floor since only the horizontal heading matters.
            az, el = np.radians(viewer.cam.azimuth), np.radians(viewer.cam.elevation)
            camera_dir_xy = np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az)])
            camera_dist = float(np.linalg.norm(camera_dir_xy))
            if camera_dist < 1e-6:
                logger.warning("Camera is looking straight down (no horizontal component) - "
                                "skipping camera-facing reorientation.")
            else:
                camera_dir = camera_dir_xy / camera_dist
                angle = float(np.arctan2(
                    travel_dir[0] * camera_dir[1] - travel_dir[1] * camera_dir[0],
                    travel_dir[0] * camera_dir[0] + travel_dir[1] * camera_dir[1],
                ))
                logger.info(
                    "Baseline travel dist %.3f mm, head module '%s'; rotating settled pose "
                    "%.2f deg about Z so that direction faces the camera.",
                    travel_dist * 1000, head_name, np.degrees(angle),
                )

                c, s = np.cos(angle), np.sin(angle)
                q_rot = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
                new_quat = np.zeros(4)
                for body_id in module_labels:
                    joint_id = model.body_jntadr[body_id]
                    qadr = model.jnt_qposadr[joint_id]
                    x, y = initial_qpos[qadr], initial_qpos[qadr + 1]
                    dx, dy = x - initial_com[0], y - initial_com[1]
                    initial_qpos[qadr] = initial_com[0] + dx * c - dy * s
                    initial_qpos[qadr + 1] = initial_com[1] + dx * s + dy * c
                    # z (qadr+2) is untouched -- pure yaw about the world Z axis.
                    mujoco.mju_mulQuat(new_quat, q_rot, initial_qpos[qadr + 3:qadr + 7])
                    initial_qpos[qadr + 3:qadr + 7] = new_quat

        # Whether or not a rotation was applied above, settle back onto
        # (the possibly-updated) initial_qpos before measuring bounds, so
        # every stage below is sized/positioned against the same reference
        # pose it will actually reset to.
        reset_to_initial_pose()

        # Stages 1-3 are the same ceiling-mounted spotlight (real MuJoCo
        # <light>, d meters overhead, aimed straight down) -- only its
        # position and cutoff cone (sized to just cover the target half's
        # farthest corner) move between stages.
        xy_min, xy_max = assembly_xy_bounds(model, data)
        mid_x = (xy_min[0] + xy_max[0]) / 2.0
        mid_y = (xy_min[1] + xy_max[1]) / 2.0

        def _spot_ceiling_light(center_xy, bounds):
            x_lo, x_hi, y_lo, y_hi = bounds
            corners = ((x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi))
            max_r = max(float(np.hypot(cx - center_xy[0], cy - center_xy[1])) for cx, cy in corners)

            model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_SPOT
            model.light_pos[light_id] = np.array([center_xy[0], center_xy[1], initial_com[2] + light_distance])
            model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])
            model.light_cutoff[light_id] = float(np.degrees(np.arctan2(max_r, light_distance)))
            model.light_diffuse[light_id] = np.array([2, 2, 2])

        # ---- Stage 1: spotlight cone covering only the left half of the body, symmetrically ----
        reset_to_initial_pose()
        left_bounds = (xy_min[0], mid_x, xy_min[1], xy_max[1])  # left half in X, full depth in Y
        left_center = ((xy_min[0] + mid_x) / 2.0 - LIGHT_TEST_LEFT_EXTRA_OFFSET, mid_y)
        _spot_ceiling_light(left_center, left_bounds)
        results["left"] = run_stage("left", light_test_duration, light_bounds={light_id: left_bounds},
                                     track_angular=True)

        # # ---- Stage 2: same spotlight, mirrored to cover only the right half of the body ----
        # reset_to_initial_pose()
        # right_bounds = (mid_x, xy_max[0], xy_min[1], xy_max[1])  # right half in X, full depth in Y
        # _spot_ceiling_light(((mid_x + xy_max[0]) / 2.0, mid_y), right_bounds)
        # results["right"] = run_stage("right", light_test_duration, light_bounds={light_id: right_bounds},
        #                               track_angular=True)

        # ---- Stage 3: same spotlight, moved to cover only the front half (+Y) of the body ----
        reset_to_initial_pose()
        front_bounds = (xy_min[0], xy_max[0], mid_y, xy_max[1])  # full width in X, front half in Y
        front_center = (mid_x, (mid_y + xy_max[1]) / 2.0 + LIGHT_TEST_FRONT_EXTRA_OFFSET)
        _spot_ceiling_light(front_center, front_bounds)
        results["front"] = run_stage("front", light_test_duration, light_bounds={light_id: front_bounds},
                                      track_acceleration=True,
                                      v_start_override=results["baseline"]["avg_linear_mps"])

    mujoco.set_mjcb_control(None)

    print("\n=== run_light_tests summary ===")
    print(f"Baseline (no light, {LIGHT_TEST_BASELINE_DURATION:.1f}s): "
          f"avg linear velocity = {results['baseline']['avg_linear_mps'] * 1000:.3f} mm/s")
    for stage_name in ("left", "right"):
        r = results.get(stage_name)
        if r is None:
            continue
        if "avg_angular_degps" in r:
            print(f"Stage '{stage_name}': avg angular velocity = {r['avg_angular_degps']:.2f} deg/s, "
                  f"net yaw rotation = {r['yaw_rotation_deg']:.2f} deg")
        else:
            print(f"Stage '{stage_name}': angular tracking unavailable (no 'free_module_1' joint)")
    front = results["front"]
    print(f"Stage 'front': avg linear velocity = {front['avg_linear_mps'] * 1000:.3f} mm/s "
          f"| v_start_y = {front['v_start_y_mps'] * 1000:.3f} mm/s, v_end_y = {front['v_end_y_mps'] * 1000:.3f} mm/s, "
          f"avg accel_y = {front['avg_accel_y_mps2'] * 1000:.3f} mm/s^2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--m", type=str, default="../models/assembly.xml",
        # "--m", type=str, default="D:\\microrobotics\\output\\evolution_run\\generation_59\\ind0_assembly.xml",
        help="MJCF model path to run in the live viewer",
    )

    parser.add_argument(
        "--o", type=str, default="../output/simulation_stats.json",
        help="Output path for simulation statistics JSON file",
    )

    parser.add_argument(
        "--max_sim_time", type=float, default=300.0,
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
        "--light_distance", type=float, default=LIGHT_TEST_DEFAULT_DISTANCE,
        help="With --light_tests, standoff distance in meters for the light source "
             f"in each stage (default: {LIGHT_TEST_DEFAULT_DISTANCE}).",
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

    if args.light_tests:
        run_light_tests(
            args.m, args.o, light_test_duration=args.light_test_duration,
            light_distance=args.light_distance,
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
            )
        else:
            run_headless(
                args.m, args.o, max_sim_time=args.max_sim_time,
                capture_img=args.capture_img, capture_gif=args.capture_gif,
                media_dir=os.path.dirname(args.o) or ".",
            )
    else:
        run_with_viewer(args.m, args.o, max_sim_time=args.max_sim_time)