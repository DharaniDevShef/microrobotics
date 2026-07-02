import os
import time
import mujoco
import mujoco.viewer
import numpy as np


model_path = "../models/simple_roblet.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("Joints:", model.njnt)
print("DoF:", model.nv)
print("Actuators:", model.nu)
# print("Actuator names:", model.actuator_names)

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.distance = 0.05
    viewer.cam.lookat[:] = [0, 0, 0.006]
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -25

    while viewer.is_running():

        # Move hinge to 45 degrees
        data.ctrl[0] = np.deg2rad(45)

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)