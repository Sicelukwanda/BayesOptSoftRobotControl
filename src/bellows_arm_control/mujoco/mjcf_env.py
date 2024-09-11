import os

from dm_control import composer
from dm_control import mjcf
from dm_control import mujoco

# load all assets
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_env import specs
import numpy as np

_ROBOTIQ_2F85_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "robotiq_2f85/assets/")

_BELLOWS_ARM_HAND_MJCF_PATH = os.path.join(os.path.dirname(__file__), "bellows_arm_hand.mjcf")

_LINK0_STL_NAME = "link0.stl"
_LINK0_STL_PATH = os.path.join(os.path.dirname(__file__), _LINK0_STL_NAME)

_LINK1_STL_NAME = "link1.stl"
_LINK1_STL_PATH = os.path.join(os.path.dirname(__file__), _LINK1_STL_NAME)

NUM_SUBSTEPS = 100


class BellowsArm(composer.Entity):
    """A Bellows Arm environment."""

    # STL files needed to render
    ASSETS = dict()

    for stl in os.listdir(_ROBOTIQ_2F85_ASSETS_DIR):
        with open(os.path.join(_ROBOTIQ_2F85_ASSETS_DIR, stl), "rb") as f:
            ASSETS[stl] = f.read()

    with open(_LINK0_STL_PATH, "rb") as f:
        ASSETS[_LINK0_STL_NAME] = f.read()

    with open(_LINK1_STL_PATH, "rb") as f:
        ASSETS[_LINK1_STL_NAME] = f.read()

    def _build(self, *args, **kwargs):
        self._model = mjcf.from_path(
            _BELLOWS_ARM_HAND_MJCF_PATH,
            escape_separators=True,
            assets=self.ASSETS,
        )

        # add velocity sensor to base of eef
        self._eef_site = self._model.find("site", "eef_site")
        self._sensor = self._model.sensor.add(  # type: ignore
            "velocimeter", site=self._eef_site
        )

    def _build_observables(self):
        return BellowsArmObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuators(self):
        return tuple(self._model.find_all("actuator"))

    @property
    def eef_vel_sensor(self):
        return self._sensor


class BellowsArmObservables(composer.Observables):
    """simple observable features for joint angles and velocities"""

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all("joint")  # does not return free joints
        return observable.MJCFFeature("qpos", all_joints)  # we can access this using observables.qpos

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qvel", all_joints)

    @composer.observable
    def eef_velocities(self):
        """average velocity over all simulator substeps"""
        return observable.MJCFFeature("sensordata", self._entity.eef_vel_sensor)
        # buffer_size=NUM_SUBSTEPS,
        # aggregator='mean'
        # )


class SwingFast(composer.Task):
    def __init__(self, bellowsarm):
        """
        swing the arm as fast as possible
        The actions are mapped into range [0, 1], for modelling and RL purposes.
        But we scale them back into the range [0, Pmax] in env.step()

        """
        # TODO: remove floor from MJCF and add it separately
        self._arm_env = bellowsarm
        self._arm_env.mjcf_model.worldbody.add("light", pos=(0, 0, 6))  # 6m high light

        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # enable observables (add corruptors if any, here)
        self._arm_env.observables.joint_positions.enabled = True
        self._arm_env.observables.joint_velocities.enabled = True
        self._arm_env.observables.eef_velocities.enabled = True

        # skipping task observables

        self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

    def action_spec(self, physics):
        """Returns a `BoundedArray` spec matching the `Physics` actuators.

        BoundedArray.name should contain a tab-separated list of actuator names.
        When overloading this method, non-MuJoCo actuators should be added to the
        top of the list when possible, as a matter of convention.

        Args:
        physics: used to query actuator names in the model.
        """
        names = [physics.model.id2name(i, "actuator") or str(i) for i in range(physics.model.nu)]
        action_spec = mujoco.action_spec(physics)
        return specs.BoundedArray(
            shape=action_spec.shape,
            dtype=action_spec.dtype,
            minimum=-1.0,
            maximum=1.0,
            name="\t".join(names),
        )

    @property
    def root_entity(self):
        return self._arm_env

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        # TODO: add code for setting initial joint positions

    def get_reward(self, physics):
        """Measures the speed of the end effector position."""
        # end_effector_speed = np.linalg.norm(physics.velocity(physics.model.geom("2_G9")))
        eef_vel = self._arm_env.observables.eef_velocities(physics)
        # print("EEF Velocity:", eef_vel)
        return np.linalg.norm(eef_vel)


class BellowsEnv(composer.Environment):
    def __init__(self, max_pressures=275.8e3) -> None:
        self.arm = BellowsArm()
        task = SwingFast(self.arm)
        self.MAX_PRESSURES = max_pressures
        self.DOF = 13

        super().__init__(task)

    def step(self, action):
        """Scales actions to the range [0, Pmax] from [-1, 1 ] and calls the parent step method
        This assumes that the parent step method does not check for action bounds.
        """
        scaled_action = (self.MAX_PRESSURES / 2.0) * (action + 1.0)
        time_step = super().step(np.asarray(scaled_action))
        return time_step
