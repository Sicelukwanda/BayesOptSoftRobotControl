import time
import unittest

from absl.testing import absltest
from dm_env import test_utils
import numpy as np

from bellows_arm_control import envs


@unittest.skip
class BellowsEnvTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    """Test to see if we are faster than real-time"""

    def make_object_under_test(self):
        return envs.BellowsEnv()

    def test_real_time(self):
        """Test to see if we are faster than real-time"""
        env = self.environment
        spec = env.action_spec()

        num_steps = 0  #  step counter
        expected_time_per_step = env.control_timestep()
        start_time = time.time()
        timestep = env.reset()
        while True:
            action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)  # Assuming a continuous action space
            timestep = env.step(action)
            if timestep.last():
                break
            num_steps += 1

        end_time = time.time()

        total_expected_real_time = num_steps * expected_time_per_step
        total_time_taken = end_time - start_time
        real_time_factor = total_expected_real_time / total_time_taken

        self.assertLess(
            total_time_taken,
            total_expected_real_time,
            f"Expected to run in less than {total_expected_real_time} seconds but\
                 took {total_time_taken} seconds. Real-time factor: {real_time_factor:.2f}.",
        )

        # Displaying the real-time factor even if the test passes
        print(f"Real-time factor: {real_time_factor:.2f}")


if __name__ == "__main__":
    absltest.main()
