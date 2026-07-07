import os
import mujoco
import mujoco.viewer
import time
import numpy as np

# Global lists to hold body IDs
sga_body_ids = []
sgb_body_ids = []

# Baseline torque magnitude (Scale this down if the robot explodes)
# Since the robot is micro-scale, start small!
TORQUE_MAGNITUDE = 0.00003 

def magnetic_field_callback(model, data):
    """
    MuJoCo Control Callback. Applies the oscillating magnetic field torque.
    SGA and SGB experience equal and opposite torques due to opposite polarities.
    """
    # 1. Clear out external forces from the previous physics step
    data.xfrc_applied.fill(0)
    
    # 2. Timing cycle from paper: 0.9s slow phase, 0.1s fast phase
    TOTAL_CYCLE_TIME = 0.001  # Keep at 1.0 for the paper's default 1-second cadence

    time_in_cycle = data.time % TOTAL_CYCLE_TIME
    # 90% of the cycle is the slow phase, 10% is the fast phase
    if time_in_cycle < (0.9 * TOTAL_CYCLE_TIME):
        # Slow tilt phase (90% of the time window)
        base_torque = -TORQUE_MAGNITUDE
    else:
        # Fast snap-back phase (10% of the time window)
        base_torque = TORQUE_MAGNITUDE

        TARGET_AXIS_INDEX = 4  # 3 = X-axis, 4 = Y-axis, 5 = Z-axis
    
    # SGA (NS Polarity)
    for body_id in sga_body_ids:
        data.xfrc_applied[body_id][4] = base_torque
        
    # SGB (SN Polarity - Opposite direction)
    for body_id in sgb_body_ids:
        data.xfrc_applied[body_id][4] = base_torque


def main():
    global sga_body_ids, sgb_body_ids
    
    model_path = "../models/assembly_model.xml"
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}' in the current directory.")
        return

    print(f"Loading model: {model_path}...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Force a micro-scale timestep for numerical stability with welds
    model.opt.timestep = 0.0001 

    # Automatically discover all body IDs matching SGA and SGB naming
    sga_body_ids = []
    sgb_body_ids = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            if "SGA_" in body_name:
                sga_body_ids.append(i)
            elif "SGB_" in body_name:
                sgb_body_ids.append(i)

    print(f"Discovered {len(sga_body_ids)} SGA bodies and {len(sgb_body_ids)} SGB bodies.")
    print("Number of joints:", model.njnt)
    print("Number of DoF:", model.nv)
    print("Actuators:", model.nu)

    # Register the control callback to run automatically inside mj_step
    mujoco.set_mjcb_control(magnetic_field_callback)

    print("Launching viewer. Press 'ESC' in the viewer window to close.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.3 
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.time()

            # Maintain your joint positions if actuators exist

            data.ctrl[:] = -10

            # Advance the physics simulation (triggers callback automatically)
            mujoco.mj_step(model, data)

            # Sync the physics state data with the visual renderer
            viewer.sync()

            # Maintain real-time execution pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    # Clean up the callback when exiting
    mujoco.set_mjcb_control(None)

if __name__ == "__main__":
    main()