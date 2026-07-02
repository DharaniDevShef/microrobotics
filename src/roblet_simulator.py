import os
import mujoco
import mujoco.viewer
import time

def main():
    # Path to your XML model
    model_path = "../models/assembly_model.xml"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}' in the current directory.")
        return

    print(f"Loading model: {model_path}...")
    
    # Load the model and create the simulation data structure
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("Number of joints:", model.njnt)
    print("Number of DoF:", model.nv)
    print("Actuators:", model.nu)

    print("Launching viewer. Press 'ESC' in the viewer window to close.")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.distance = 0.3 # decrease to zoom in
        viewer.cam.lookat[:] = [0,0,0]
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

        while viewer.is_running():
            step_start = time.time()
            time.sleep(0.01)

            data.ctrl[:] = -10

            # Advance the physics simulation
            mujoco.mj_step(model, data)

            # Sync the physics state data with the visual renderer
            viewer.sync()

            # Maintain real-time execution pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
