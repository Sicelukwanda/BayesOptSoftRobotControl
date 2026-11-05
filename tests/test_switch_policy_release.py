from absl.testing import absltest
import numpy as np

from bellows_arm_control import bayesopt
from bellows_arm_control import envs


class SwitchPolicyReleaseTest(absltest.TestCase):
    """Test to see if SwitchPolicyRelease works as expected."""

    def make_object_under_test(self, env, act_discretization, horizon):
        """Creates an instance of SwitchPolicyRelease with given parameters."""
        return bayesopt.SwitchPolicyRelease(env, act_discretization, horizon)

    def test_initialization(self):
        """Test if the SwitchPolicyRelease initializes correctly."""
        env = envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)
        act_discretization = 2
        horizon = 10

        policy = self.make_object_under_test(env, act_discretization, horizon)

        self.assertIsNotNone(policy, msg="Policy object is None!")

    def test_action_discretization(self):
        env = envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)
        act_discretizations = [2, 3, 6, 11]
        horizon = 10

        for ad in act_discretizations:
            policy = self.make_object_under_test(env, ad, horizon)

            self.assertEqual(
                len(policy.act_space),
                ad,
                msg=f"Expected action space length {ad} but got {len(policy.act_space)}",
            )
            self.assertEqual(
                policy.act_space[0],
                env.action_spec().minimum,
            )
            self.assertEqual(
                policy.act_space[-1],
                env.action_spec().maximum,
            )
            self.assertEqual(
                min(policy.act_space),
                env.action_spec().minimum,
            )
            self.assertEqual(
                max(policy.act_space),
                env.action_spec().maximum,
            )

    def test_action_lib(self):
        env = envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)
        act_discretizations = [2, 3, 6, 11]
        horizon = 10

        for ad in act_discretizations:
            policy = self.make_object_under_test(env, ad, horizon)

            num_actions = ad**6
            self.assertEqual(
                policy.dof12_act_lib.shape,
                (num_actions, env.action_spec().shape[0]),
            )

    def test_antagonistic_bellows(self):
        """Test implementation of antagonistic pairings."""
        env = envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)
        act_discretizations = [2, 3, 6, 11]
        horizon = 10

        for ad in act_discretizations:
            policy = self.make_object_under_test(env, ad, horizon)

            # case 1
            a = np.ones((6,)) * policy.act_space[0]
            b = np.zeros(env.action_spec().shape[0])
            b[:12] = np.array([policy.act_space[0], policy.act_space[-1]] * 6)
            np.testing.assert_allclose(policy.dof6_to_dof12(a), b)

            # case 2
            a = np.ones((6,)) * policy.act_space[-1]
            b = np.zeros(env.action_spec().shape[0])
            b[:12] = np.array([policy.act_space[-1], policy.act_space[0]] * 6)
            np.testing.assert_allclose(policy.dof6_to_dof12(a), b)

            # case 3: random vector in [0, 1]^6
            b_rand = np.random.uniform(size=(6,))
            arr = []
            for i in range(6):
                arr += [b_rand[i], 1.0 - b_rand[i]]

            c = np.zeros(env.action_spec().shape[0])
            c[:12] = np.array(arr)
            np.testing.assert_allclose(policy.dof6_to_dof12(b_rand), c)

    def test_policy(self):
        env = envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)
        act_discretizations = [2, 3, 6, 11]
        horizon = 10

        for ad in act_discretizations:
            policy = self.make_object_under_test(env, ad, horizon)
            b = np.random.uniform(size=(horizon + 1,))
            act_seq = policy(b)

            self.assertEqual(
                act_seq.shape,
                (horizon, env.action_spec().shape[0]),
            )
            self.assertIsInstance(act_seq, np.ndarray)

    def test_release(self):
        """Release with zeros, squeeze gripper with ones."""
        env = envs.BellowsEnv(task_str="throw", order_hold_ctrl=True)
        act_discretizations = [2, 3, 6, 11]
        horizon = 10

        for ad in act_discretizations:
            policy = self.make_object_under_test(env, ad, horizon)
            b = np.random.uniform(size=(horizon + 1,))

            # case 1: release at 0
            b[-1] = 0.0
            act_seq = policy(b)
            c = np.zeros(horizon)
            np.testing.assert_allclose(act_seq[:, -1], c)

            # case 2: release at last position (-1)
            b[-1] = 1.0
            act_seq = policy(b)
            c = np.ones(horizon)
            c[-1] = 0.0
            np.testing.assert_allclose(act_seq[:, -1], c)

        horizons = [1, 2, 9]
        for h in horizons:
            policy = self.make_object_under_test(env, 2, h)
            b = np.random.uniform(size=(h + 1,))

            # case 1: release at 0
            b[-1] = 0.0
            act_seq = policy(b)
            c = np.zeros(h)
            np.testing.assert_allclose(act_seq[:, -1], c)

            # case 2: release at last position (-1)
            b[-1] = 1.0
            act_seq = policy(b)
            c = np.ones(h)
            c[-1] = 0.0
            np.testing.assert_allclose(act_seq[:, -1], c)


if __name__ == "__main__":
    absltest.main()
