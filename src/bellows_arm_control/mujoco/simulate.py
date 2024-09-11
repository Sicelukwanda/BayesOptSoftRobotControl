import os

import mujoco
import mujoco_viewer
import numpy as np

_BELLOWS_MUJOCO_ROOT = os.path.dirname(__file__)
# STL files needed to render
ASSETS = dict()
BASE_DIR = _BELLOWS_MUJOCO_ROOT + "/robotiq_2f85/assets/"
stl_list = os.listdir(BASE_DIR)

for stl in stl_list:
    with open(BASE_DIR + stl, "rb") as f:
        ASSETS[stl] = f.read()

with open(_BELLOWS_MUJOCO_ROOT + "/link0.stl", "rb") as f:
    ASSETS[_BELLOWS_MUJOCO_ROOT + "/link0.stl"] = f.read()

with open(_BELLOWS_MUJOCO_ROOT + "/link1.stl", "rb") as f:
    ASSETS[_BELLOWS_MUJOCO_ROOT + "/link1.stl"] = f.read()

model = mujoco.MjModel.from_xml_path(_BELLOWS_MUJOCO_ROOT + "/bellows_arm.mjcf", ASSETS)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)
viz_step = 1 / 60.0
MAX_PRESSURE = 300e3

start = data.time
# time length of simulation
duration = 10.0  # (seconds)

skip_steps = 1000  # controls sampling resolution (ms) (works out to control rate of 10Hz)

# set integration timestep (i.e., environment physics timestep)
dt = model.opt.timestep

# init commands
P_cmd = np.zeros(12)  # np.random.uniform(0, MAX_PRESSURE, 13)
P_cmd[-1] = 0.0
while data.time < duration:
    mujoco.mj_step(model, data)
    # implement substep loop
    if int(data.time / dt) % skip_steps == 0:
        # P_cmd = np.zeros(13) + 100e3*np.sin(data.time)
        P_cmd = np.random.uniform(0, MAX_PRESSURE, 12)
        # P_cmd[0] = 100.0 #0.5*np.sin(data.time) + 1.0
        # P_cmd[0] = 300e3

        # overwrite pressures for gripper
        P_cmd[-1] = 100e3 * np.sin(data.time)
    data.ctrl = P_cmd

    if viewer.is_alive and data.time - start > viz_step:
        viewer.render()
        start = data.time
