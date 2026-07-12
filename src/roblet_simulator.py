
import os
import mujoco
import mujoco.viewer
import time
import numpy as np

# Global dictionaries mapping Parent Body ID -> List of (Magnet Geom ID, Polarity Sign)
parent_body_magnet_map = {}

# --- REAL PHYSICS CONSTANTS ---
B_INTENSITY = 0.0014  # 1.4 mT
M_MOMENT = 6.5e-3     # 6.5 x 10^-3 Am^2

# Test with a moderate multiplier to handle default joint damping
TORQUE_MULTIPLIER = 1

def magnetic_field_callback(model, data):
    """
    MuJoCo Control Callback. Implements suggestions 1, 3, and 5.
    Calculates Tau = M x B using a smooth continuous oscillating field,
    and applies the accumulated torque directly to the main parent module bodies.
    """
    # Clear out external forces from the previous physics step
    data.xfrc_applied.fill(0)
    
    # PROBLEM 5 FIX: Continuous asymmetric wave instead of a digital step flip
    # 1 Hz cycle: 0.9s sweeping forward/down, 0.1s snapping back up
    TOTAL_CYCLE_TIME = 1.0
    time_in_cycle = data.time % TOTAL_CYCLE_TIME
    
    if time_in_cycle < 0.9:
        # Smoothly sweep from +45 down to -45 degrees over 0.9 seconds
        alpha = time_in_cycle / 0.9
        theta = np.radians(45.0 - (90.0 * alpha))
    else:
        # Rapidly snap back from -45 up to +45 degrees over 0.1 seconds
        alpha = (time_in_cycle - 0.9) / 0.1
        theta = np.radians(-45.0 + (90.0 * alpha))
    
    # Global B vector oscillating smoothly in the XZ plane
    B_vector = np.array([np.cos(theta), 0.0, np.sin(theta)]) * B_INTENSITY

    # PROBLEM 1 FIX: Apply torque directly to the main moving parent bodies
    for parent_body_id, magnet_list in parent_body_magnet_map.items():
        accumulated_torque = np.zeros(3)
        
        for geom_id, polarity_sign in magnet_list:
            # Get the real-time world rotation matrix of the magnet geom
            geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
            
            # PROBLEM 3 CHECK: Local magnet dipole along local cylinder height.
            # If your CAD mesh treats the cylinder axis as Z, change this to [0,0,1]
            local_m = np.array([0.0, 0.0, 1.0]) * M_MOMENT * polarity_sign
            
            # Rotate local dipole to global world space
            world_m = geom_mat.dot(local_m)
            
            # Compute cross product vector: Tau = M x B
            torque_vector = np.cross(world_m, B_vector)
            accumulated_torque += torque_vector
            
        # Apply the combined vector to the parent module body (slots 3, 4, 5)
        data.xfrc_applied[parent_body_id][3:6] = accumulated_torque * TORQUE_MULTIPLIER


def find_main_movable_parent(model, body_id):
    """
    Recursively walks up the XML body tree to find the top-level movable parent body
    (the body that is a direct child of the world body, usually 'module_x').
    """
    current_id = body_id
    # Loop up until the parent is the world body (ID 0)
    while current_id != 0:
        parent_id = model.body_parentid[current_id]
        if parent_id == 0:
            return current_id
        current_id = parent_id
    return body_id


def main():
    global parent_body_magnet_map
    
    model_path = "../models/assembly_model.xml"
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'")
        return

    print(f"Loading model: {model_path}...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.01

    # Dynamic Discovery: Trace magnet geoms up to their movable parent modules
    parent_body_magnet_map.clear()
    
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and "magnet_" in geom_name:
            # Immediate body holding the geom (e.g., 'SGA_4')
            immediate_body_id = model.geom_bodyid[geom_id]
            
            # Walk up the tree to find the main moving body (e.g., 'module_2')
            movable_parent_id = find_main_movable_parent(model, immediate_body_id)
            
            # Determine polarity sign based on naming convention
            polarity_sign = 1.0 if "SGA" in geom_name else -1.0
            
            if movable_parent_id not in parent_body_magnet_map:
                parent_body_magnet_map[movable_parent_id] = []
                
            parent_body_magnet_map[movable_parent_id].append((geom_id, polarity_sign))


    # Register our combined vector control loop
    mujoco.set_mjcb_control(magnetic_field_callback)

    print("Launching passive viewer window...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.15
        viewer.cam.lookat[:] = [0, 0, 0]

        while viewer.is_running():
            step_start = time.time()

            if model.nu > 0:
                data.ctrl[:] = -10

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    mujoco.set_mjcb_control(None)

if __name__ == "__main__":
    main()

