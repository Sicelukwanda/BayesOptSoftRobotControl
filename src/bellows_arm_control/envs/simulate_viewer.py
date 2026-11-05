# type: ignore
"""Run example simulation of the Bellows arm using mujoco"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from bellows_arm_control.envs import BellowsEnv


def main():
    env = BellowsEnv("smash")

    spec = env.action_spec()

    model = mujoco.MjModel.from_xml_string(
        env.task.root_entity.mjcf_model.to_xml_string(),
        env.task.root_entity.mjcf_model.get_assets(),
    )

    data = mujoco.MjData(model)

    MAX_PRESSURE = 300e3

    start = data.time
    # time length of simulation
    duration = 30.0  # (seconds)

    # controls sampling resolution (ms) (works out to control rate of 10Hz)
    skip_steps = 1000

    # set integration timestep (i.e., environment physics timestep)
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)
            if int(data.time / dt) % skip_steps == 0:
                P_cmd = np.random.uniform(0, MAX_PRESSURE, spec.shape)

                # overwrite pressures for gripper
                P_cmd[-1] = 100e3 * np.sin(data.time)
                data.ctrl = P_cmd

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
