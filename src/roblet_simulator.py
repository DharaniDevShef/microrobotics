"""
Roblet Simulator - A simulation environment for testing self folding and
magnetic actuation of microrobots using MuJoCo physics engine.
"""

# pylint: disable=no-member

import os
import time
import mujoco
import mujoco.viewer
import numpy as np

# Global dictionaries mapping Parent Body ID -> List of (Magnet Geom ID, Polarity Sign)
parent_body_magnet_map = {}

# Physics constants
# Magnetic field intensity (Tesla)
B_INTENSITY = 0.01  # 10 mT
# 6.5 x 10^-3 Am^2 - 2mm x 2mm neodymium magnet cylinder (N42SH)
M_MOMENT = 6.5e-3
# Maximum torque multiplier
TORQUE_MULTIPLIER = 1.0
# Time taken to reach full torque
TORQUE_RAMP_TIME = 10  # seconds
FREQUENCY = 1  # Hz


def magnetic_field_callback(model, data):
    """
    MuJoCo Control Callback.
    Calculates magnetic torque:
        Tau = M x B
    Applies torque to parent module bodies.
    """
    # Clear previous external forces
    data.xfrc_applied.fill(0)
    # Smooth torque ramp
    ramp_progress = min(data.time / TORQUE_RAMP_TIME, 1.0)
    # Smoothstep: f(x)=x^2*(3−2x)
    # 0 -> 1 with zero slope at start and end
    ramp_factor = ramp_progress * ramp_progress * (3 - 2 * ramp_progress)
    effective_torque_multiplier = TORQUE_MULTIPLIER * ramp_factor

    # Oscillating magnetic field settings
    total_cycle_time = 1.0 / FREQUENCY
    time_in_cycle = data.time % total_cycle_time

    # The magnetic field is controlled to roll forward to -50◦in 0.9 s,
    # and then backward to 50◦in 0.1 s, allowing the robot to
    # slowly tilt down and quickly tilt up to perform the stick-slip motion

    # Change from (-50 + 100*alpha)to (50 - 100*alpha)
    if time_in_cycle < 0.9:
        alpha = time_in_cycle / 0.9
        theta = np.radians(-50.0 + (100.0 * alpha))
    else:
        alpha = (time_in_cycle - 0.9) / 0.1
        theta = np.radians(50.0 - (100.0 * alpha))

    # Magnetic field in XZ plane
    b_vector = np.array([np.cos(theta), 0.0, np.sin(theta)]) * B_INTENSITY

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
        # Apply ramped torque
        data.xfrc_applied[parent_body_id][3:6] = (
            accumulated_torque * effective_torque_multiplier
        )


def find_main_movable_parent(model, body_id):
    """
    Traces up the MuJoCo body tree to find the
    main movable module body.
    """
    current_id = body_id
    while current_id != 0:
        parent_id = model.body_parentid[current_id]
        if parent_id == 0:
            return current_id
        current_id = parent_id
    return body_id


def find_all_magnets(model):
    """
    Finds all magnets in the model and returns a dictionary mapping
    Parent Body ID -> List of (Magnet Geom ID, Polarity Sign)
    """
    magnet_map = {}
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and "magnet_" in geom_name:
            immediate_body_id = model.geom_bodyid[geom_id]
            movable_parent_id = find_main_movable_parent(model, immediate_body_id)
            polarity_sign = 1.0 if "SGA" in geom_name else -1.0
            if movable_parent_id not in magnet_map:
                magnet_map[movable_parent_id] = []
            magnet_map[movable_parent_id].append((geom_id, polarity_sign))
    return magnet_map


def main():
    """Load the MuJoCo model, initialize magnets, and run the simulation."""
    model_path = "../models/assembly_model.xml"
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'")
        return
    print(f"Loading model: {model_path}...")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Simulation timestep
    model.opt.timestep = 0.01  # 10 milliseconds

    # Find all magnets
    parent_body_magnet_map.clear()
    parent_body_magnet_map.update(find_all_magnets(model))

    # Register callback
    mujoco.set_mjcb_control(magnetic_field_callback)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.5  # zoom
        viewer.cam.lookat[:] = [0, 0, 0]
        last_print = -1

        while viewer.is_running():
            step_start = time.time()
            if model.nu > 0:
                # Set control inputs to -10 for all actuators (if any)
                data.ctrl[:] = -10

            mujoco.mj_step(model, data)
            viewer.sync()

            # Print torque ramp progress every second
            current_second = int(data.time)
            if current_second != last_print:
                last_print = current_second
                ramp = min(data.time / TORQUE_RAMP_TIME, 1.0)
                if ramp < 1.0:
                    print(
                        f"Time: {data.time:.2f}s | Torque ramp: {int(ramp*100)} - {int((ramp+0.1)*100)}%"
                    )

            # Maintain real-time speed
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
