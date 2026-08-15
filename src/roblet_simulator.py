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
B_INTENSITY = 0.01  # 10 mT
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

# Wall geoms are tagged group=1 in the model (see mjcf_generator.py) so this
# raycast can be filtered to see ONLY them.
_WALL_GEOMGROUP = np.zeros(6, dtype=np.uint8)
_WALL_GEOMGROUP[1] = 1


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


def set_angle_to_joint(model, data, target_angle_deg):
    """Set actuator position targets from a scalar, ordered sequence, or name map."""
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
                           b_intensity=None):
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
    which candidate B produced the reported stats."""

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


def run_headless(
    model_path, stats_output_path, max_sim_time=None,
    capture_img=False, capture_gif=False, media_dir="../output", gif_fps=15,
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
    """
    parent_body_magnet_map.clear()
    torque_history.clear()
    b_field_history.clear()
    time_history.clear()

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.01

    target_angles = read_joint_target_angles_from_xml(model_path)
    if not target_angles:
        target_angles = [45.0] * model.nu

    parent_body_magnet_map.update(find_all_magnets(model))
    module_labels = find_module_labels(model)


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
        # Save screenshot if requested
        if capture_img and renderer is not None:
            # Run a short settle phase to let the hinge position actuators reach their
            for _ in range(SETTLE_MAX_STEPS):
                set_angle_to_joint(model, data, target_angle_deg=target_angles)
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

            if physics_ok:
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

    save_simulation_stats(model, avg_velocity, displacement, module_labels,
                           filename=stats_output_path, physics_ok=physics_ok, is_stable=is_stable,
                           b_intensity=B_INTENSITY)

    return {
        "model": model_name,
        "physics_ok": physics_ok,
        "success": success,
        "is_stable": is_stable,
        "B_intensity_T": B_INTENSITY,
        "avg_velocity_mmps": avg_velocity * 1000 if success else 0.0,
        "displacement_mm": displacement * 1000 if success else 0.0,
        "sim_time_s": data.time,
    }


def run_headless_b_sweep(
    model_path, stats_output_path, b_values=B_SWEEP_VALUES, max_sim_time=None,
    capture_img=False, capture_gif=False, media_dir="../output", gif_fps=15,
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
    """
    global B_INTENSITY
    original_b = B_INTENSITY

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
            )
        finally:
            B_INTENSITY = original_b
    else:
        # No successful B found, write a zeroed stats file to the caller's
        # requested path.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--m", type=str, default="../models/assembly.xml",
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

    if args.headless:
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