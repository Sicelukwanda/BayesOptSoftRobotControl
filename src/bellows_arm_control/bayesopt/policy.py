"""Module for policy parameterizations."""

from itertools import product
import os

import numpy as np


class SwitchPolicy:
    """
    Switching policy for Bayesian optimization.
    - defines a dictionary of predetermined controls to switch from
    - horizon -
    """

    def __init__(self, env, act_discretization, horizon):
        self.spec = env.action_spec()
        assert self.spec.minimum == 0.0, "SwitchPolicy only works with action spec [0,1]!"
        assert self.spec.maximum == 1.0, "SwitchPolicy only works with action spec [0,1]!"
        # generate action sequences
        self.env = env
        self.param_dims = 6
        self.horizon = horizon
        self.spec = env.action_spec()
        self.action_dim = self.spec.shape[0]
        self.act_space = np.linspace(self.spec.minimum, self.spec.maximum, act_discretization).tolist()

        combinations = list(product(self.act_space, repeat=6))
        # library of 12 dof actions
        self.dof12_act_lib = np.array(list(map(self.dof6_to_dof12, combinations)))

    def dof6_to_dof12(self, dof6):
        """
        Convert a 6-DOF vector to a 12-DOF vector through pairings of antagonistic bellows.
        """
        dof12 = np.zeros(self.action_dim)  # accounts for inactive grippers

        for j in range(12):
            if j % 2 == 0:
                dof12[j] = dof6[int(j / 2)]
            else:
                dof12[j] = 1.0 - dof6[int(j / 2)]

        return dof12  # may return array of size bigger than 12 if act_spec is larger

    def __call__(self, params, time_step=None, save=False):
        """Converts parameter values to indices of self.dof12_act_lib."""
        params_np = np.asarray(params, dtype=np.float64)
        assert len(params_np.shape) == 1
        # we need param to have the same length as horizon
        assert params_np.shape[0] == self.horizon

        # idx conversion works only for range [0, 1]
        assert np.max(params_np) <= 1.0
        assert np.min(params_np) >= 0.0

        action_indices = []
        for param in params_np:
            idx = int(np.ceil(float(param) * (self.dof12_act_lib.shape[0] - 1)))
            action_indices.append(idx)

        act_seq = self.dof12_act_lib[action_indices, :].copy()

        if save:
            self.save(act_seq)

        return act_seq

    def save(self, act_seq):
        start_idx = 0
        while os.path.exists(f"a_{start_idx}.npy"):
            start_idx += 1
        for j in range(act_seq.shape[0]):
            np.save(f"a_{start_idx + j}.npy", act_seq[j])


class SwitchPolicyRelease(SwitchPolicy):
    """
    Actions here will have shape TxD. For release, the last parameter in range [0,1] will be converted to t range [0,T]
    Then in the released actions, the action from t onwards, will be set to 0 otherwise it will stay as 1
    """

    def __init__(self, env, act_discretization, horizon):
        super().__init__(env, act_discretization, horizon)

    def __call__(self, params, time_step=None, save=False):
        params_np = np.asarray(params, dtype=np.float64)
        act_seq = super().__call__(params_np[:-1], time_step=None, save=False)

        # idx conversion works only for range [0, 1]
        assert params_np[-1] <= 1.0
        assert params_np[-1] >= 0.0

        release_param = float(params_np[-1])
        t = int(np.ceil(release_param * (self.horizon - 1)))

        release_vec = np.ones(act_seq.shape[0])
        release_vec[t:] = 0.0
        act_seq[:, -1] = release_vec

        # update release time for the environment (to calculate reward)
        self.env.task.release_time = ((t + 1) / self.horizon) * self.env._max_duration

        if save:
            self.save(act_seq)
        return act_seq
