"""Environment tests for the BellowsArm inner controller strategies."""

from absl.testing import absltest
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from bellows_arm_control import envs

matplotlib.use("Agg")


class BellowsCtrlrSmoothnessTest(absltest.TestCase):
    """Test that the BellowsArm environment inner controllers behave as expected."""

    def test_n_sub_steps(self):
        """Because we implement our own substep loop for env.step(), any call to
        super().step() should run only once."""
        env = envs.BellowsEnv(task_str="reach", num_substeps=1, order_hold_ctrl=False)
        self.assertEqual(env._n_sub_steps, 1)

        env = envs.BellowsEnv(task_str="reach", num_substeps=1, order_hold_ctrl=True)
        self.assertEqual(env._n_sub_steps, 1)

    def test_ctrlr_smoothness(self):
        actions = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        physics_t_step = 0.001
        duration = 0.25  # seconds

        num_hold_steps = int((duration / physics_t_step) / actions.shape[0])

        env1 = envs.BellowsEnv(
            task_str="reach",
            num_substeps=num_hold_steps,
            order_hold_ctrl=False,
        )  # zero order hold (stepped signal)
        env2 = envs.BellowsEnv(
            task_str="reach",
            num_substeps=num_hold_steps,
            order_hold_ctrl=True,
        )  # first order hold

        env1_obs = [env1.reset()]
        env2_obs = [env2.reset()]

        env1_acts = []
        env2_acts = []

        for action in actions:
            o1_list, a1_list = env1.step(action, debug=True)
            env1_obs.extend(o1_list)
            env1_acts.extend(a1_list)

            o2_list, a2_list = env2.step(action, debug=True)
            env2_obs.extend(o2_list)
            env2_acts.extend(a2_list)

        self.assertEqual(len(env1_obs), len(env2_obs))
        env1_acts_arr = np.array(env1_acts)
        env2_acts_arr = np.array(env2_acts)

        fig = plt.figure(figsize=(12, 6))
        T = np.arange(env1_acts_arr.shape[0]) * physics_t_step
        plt.subplot(2, 2, 1)
        plt.title("Stepped Low-level controller")
        plt.plot(T, env1_acts_arr)

        plt.subplot(2, 2, 2)
        plt.title("First order hold controller")
        plt.plot(T, env2_acts_arr)

        obs1_arr = np.array([t.observation["bellows_arm/joint_states"][-12:-6] for t in env1_obs])
        obs2_arr = np.array([t.observation["bellows_arm/joint_states"][-12:-6] for t in env2_obs])
        self.assertEqual(obs1_arr.shape, obs2_arr.shape)

        T_obs = np.arange(obs1_arr.shape[0]) * physics_t_step
        plt.subplot(2, 2, 3)
        plt.title("Stepped Low-level controller u,v")
        plt.plot(T_obs, obs1_arr)

        plt.subplot(2, 2, 4)
        plt.title("First order hold controller u,v")
        plt.plot(T_obs, obs2_arr)
        plt.close(fig)


if __name__ == "__main__":
    absltest.main()
