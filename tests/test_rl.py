"""Environment tests for the BellowsArm environment."""

import unittest

from absl.testing import absltest
from dm_env import StepType
from dm_env import test_utils
import numpy as np

from bellows_arm_control import envs


class BellowsEnvTest(test_utils.EnvironmentTestMixin):
    """Test that the BellowsArm environment conforms to the dm_env API.
    Tests here should work on all environments."""

    def test_action_spec(self):
        # Assuming that the gripper is attached
        action_spec = self.environment.action_spec()
        self.assertEqual(action_spec.shape, (self.action_spec_dof,))
        self.assertEqual(action_spec.dtype, np.float64)
        self.assertEqual(action_spec.minimum, 0.0)
        self.assertEqual(action_spec.maximum, 1.0)

    @unittest.skip("TODO: add some checks on observation spec")
    def test_observation(self):
        """TODO"""

    def test_timestep_type(self):
        """Test that we have correct Time Step types at start,
        middle and termination of the environment."""

        env = self.environment
        t_obs = env.reset()
        self.assertValidStep(t_obs)
        # Check that timestep is type FIRST at the start
        self.assertEqual(t_obs.step_type, StepType.FIRST)

        hasLastTimeStep = False
        # Iterate through the environment steps
        while env.physics.data.time < env.max_duration:
            action = np.zeros(
                env.action_spec().shape
            )  # Example action, adjust as necessary
            t_obs = env.step(action)
            self.assertValidStep(t_obs)
            # In the middle of the environment, the step type should be MID
            if env.physics.data.time < env.max_duration:
                self.assertEqual(t_obs.step_type, StepType.MID)
            else:
                # On the last step, the step type should be LAST
                self.assertEqual(t_obs.step_type, StepType.LAST)
                hasLastTimeStep = True

        self.assertEqual(hasLastTimeStep, True)

    # def test_determinism(self):
    #     """Test that environment is deterministic."""
    #     env = self.environment
    #     spec = env.action_spec()
    #     derterministic_action = np.random.uniform(
    #         spec.minimum, spec.maximum, spec.shape
    #     )
    #     env.reset()
    #     obs1 = env.step(derterministic_action).observation["bellows_arm/joint_states"]
    #     env.reset()
    #     obs2 = env.step(derterministic_action).observation["bellows_arm/joint_states"]

    #     np.testing.assert_allclose(obs1, obs2, atol=0.00001)


# subclasses for each env
class ReachTest(BellowsEnvTest, absltest.TestCase):
    def make_object_under_test(self):
        self.action_spec_dof = 13
        return envs.BellowsEnv(task_str="reach", order_hold_ctrl=True)

    def test_num_hit_reset(self):
        env = self.environment
        env.reset()
        self.assertEqual(env._task.num_hits, 0)
        env._task.num_hits = 100
        self.assertEqual(env._task.num_hits, 100)
        env.reset()
        self.assertEqual(env._task.num_hits, 0)


class SmashTest(BellowsEnvTest, absltest.TestCase):
    def make_object_under_test(self):
        self.action_spec_dof = 12
        return envs.BellowsEnv(task_str="smash", order_hold_ctrl=True)

    def test_reward_target_distance(self):
        """Test the if the distance of the smash target distance is being measured correctly.
        i.e., place it at known distances"""

        env = self.environment

        random_button_pos = np.zeros(3)
        random_button_pos[:2] = np.random.uniform(low=-5, high=5, size=(2,))
        env.task._button_initial_pose = tuple(random_button_pos)

        env.task._button.set_pose(env.physics, position=env.task._button_initial_pose)
        self.assertEqual(
            np.linalg.norm(env.task.to_button(env.physics)),
            np.linalg.norm(env.task._button_initial_pose),
        )

        action_spec = env.action_spec()

        def nothing_policy(time_step):
            # do nothing
            return np.zeros(action_spec.shape)

        # uncomment to visualize changes
        # viewer.launch(env, policy=nothing_policy)

    def test_reward_button_no_press(self):
        """Test the if the force is zero when button not touched.
        i.e., Place it away from robot.
        """

        env = self.environment

        env.task._button_initial_pose = (2, 2, 0.0)
        env.task._button.set_pose(env.physics, position=env.task._button_initial_pose)

        smash_force_ = env.task._button.observables.touch_force(
            env.physics
        )  # this returns a SynchronizedArrayWrapper object
        smash_force = np.asarray(smash_force_).mean()

        self.assertEqual(smash_force, 0.0)

        action_spec = env.action_spec()

        def nothing_policy(time_step):
            # do nothing
            return np.zeros(action_spec.shape)

        # uncomment to visualize changes
        # viewer.launch(env, policy=nothing_policy)

    def test_reward_button_press(self):
        """Test the if the force on the button is being measured correctly.
        i.e., raise the sensor button so that it gets pressed when the robot drops onto it.
        """

        env = self.environment

        env.task._button_initial_pose = (-0.5, 0.5, 0.4)
        env.task._button.set_pose(env.physics, position=env.task._button_initial_pose)

        env.reset()
        r = 0.0
        while env.physics.data.time < env.max_duration:
            action = np.zeros(
                env.action_spec().shape
            )  # Example action, adjust as necessary
            t_obs = env.step(action)
            r += t_obs.reward

        self.assertGreater(r, 0.0)

        action_spec = env.action_spec()

        def nothing_policy(time_step):
            # do nothing
            smash_force_ = env.task._button.observables.touch_force(
                env.physics
            )  # this returns a SynchronizedArrayWrapper object
            smash_force = np.asarray(smash_force_).mean()
            # self.assertEqual(smash_force, 1.0)
            print("smash force:", smash_force)
            return np.zeros(action_spec.shape)

        # uncomment to visualize changes
        # viewer.launch(env, policy=nothing_policy)


class ThrowTest(BellowsEnvTest, absltest.TestCase):
    def make_object_under_test(self):
        self.action_spec_dof = 13
        return envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)


if __name__ == "__main__":
    absltest.main()
