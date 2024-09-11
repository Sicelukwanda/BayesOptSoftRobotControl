from dm_control import composer
from dm_control import mujoco
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_env import StepType
from dm_env import specs
import numpy as np

from bellows_arm_control.envs import bellows_arm
from bellows_arm_control.envs import external_sensors
from bellows_arm_control.envs import grippers
from bellows_arm_control.envs import objects


class SwingFast(composer.Task):
    def __init__(self, bellowsarm, inner_num_substeps):
        """
        swing the arm as fast as possible
        The actions are mapped into range [0, 1], for modelling and RL purposes.
        But we scale them back into the range [0, Pmax] in env.step()

        """
        # world plane & light
        self._arena = floors.Floor()
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))  # 4m high light
        # arm
        self._arm = bellowsarm
        # attach the button to the world (at what position?)

        self._arena.attach(self._arm)

        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # enable observables (add corruptors if any, here)

        # individual bellows disk positions and velocities (54 by default)
        self._arm.observables.joint_positions.enabled = False
        self._arm.observables.joint_velocities.enabled = False

        # velocimeter mesurement of end effector x,y,z velocities
        self._arm.observables.eef_velocities.enabled = True
        # add velocimeter + (joint states (u,v)+ (u_dot, v_dot) for all bellows "joints")
        self._arm.observables.joint_states.enabled = True

        self.inner_num_substeps = inner_num_substeps

        # skipping task observables

        # determine delay in seconds between action updates
        self.control_timestep = self.inner_num_substeps * self.physics_timestep

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
            minimum=0.0,
            maximum=1.0,
            name="\t".join(names),
        )

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        # TODO: add code for setting initial joint positions

    def get_reward(self, physics):
        """Measures the speed of the end effector position."""
        # end_effector_speed = np.linalg.norm(physics.velocity(physics.model.geom("2_G9")))
        eef_vel = self._arm.observables.eef_velocities(physics)
        # print("EEF Velocity:", eef_vel)
        return np.linalg.norm(eef_vel)


class Smash(composer.Task):
    def __init__(self, bellowsarm, inner_num_substeps, max_duration, observe_sensors=True):
        """
        Smash a force sensor with the end effector
        The actions are mapped into range [0, 1], for modelling and RL purposes.
        But we scale them back into the range [0, Pmax] in env.step()

        """
        # TODO: possibly add variator for location of the force sensor

        self._max_duration = max_duration
        self.release_time = max_duration

        # world plane & light
        self._arena = floors.Floor()
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))  # 4m high light
        # arm
        self._arm = bellowsarm
        # attach the button to the world (at what position?)
        self._button = external_sensors.ForceButton(inner_num_substeps, target_force_range=(5, 1000))

        self._arena.attach(self._arm)
        self._arena.attach(self._button)
        self._button_initial_pose = (-0.5, 0.5, 0.0)  # TODO: replace this with variator

        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # enable observables (add corruptors if any, here)

        # individual bellows disk positions and velocities (54 by default)
        self._arm.observables.joint_positions.enabled = False
        self._arm.observables.joint_velocities.enabled = False

        # velocimeter mesurement of end effector x,y,z velocities
        self._arm.observables.eef_velocities.enabled = True
        # add velocimeter + (joint states (u,v)+ (u_dot, v_dot) for all bellows "joints")
        self._arm.observables.joint_states.enabled = False

        # force sensor (disable observations by default)
        self._button.observables.touch_force.enabled = True

        # add task specific observations
        # if observe_sensors:
        self._task_observables = {}
        self._task_observables[self._button.mjcf_model.model + "/positions"] = observable.Generic(self.to_button)
        for obs in self._task_observables.values():
            obs.enabled = True  # enable all observables

        self.inner_num_substeps = inner_num_substeps

        # determine delay in seconds between action updates
        self.control_timestep = self.inner_num_substeps * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def to_button(self, physics):
        """Returns the vector from base of arm to the button in the local frame of the base"""
        button_global_frame_pos, _ = self._button.get_pose(physics)
        # assuming self._arm is type composer.Entity
        assert isinstance(self._arm, composer.Entity)
        return self._arm.global_vector_to_local_frame(physics, button_global_frame_pos)

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
            minimum=0.0,
            maximum=1.0,
            name="\t".join(names),
        )

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        # TODO: add code for setting initial joint positions
        # TODO: add variator for initial button position
        self._button.set_pose(physics, position=self._button_initial_pose)

    def get_reward(self, physics):  # distance to sensor + force applied
        """Combines eef speed, button measured force, and closeness to button."""
        # end_effector_speed = np.linalg.norm(physics.velocity(physics.model.geom("2_G9")))
        eef_vel_z = self._arm.observables.eef_velocities(physics)[-1]
        ball_mass = physics.model.body("bellows_arm/eef_ball/").mass[0]

        vertical_momentum = ball_mass * np.abs(eef_vel_z)
        smash_force_ = self._button.observables.touch_force(physics)  # this returns a SynchronizedArrayWrapper object
        smash_force = np.asarray(smash_force_).mean()

        # distance to sensor (shaped reward)
        btn_eef_dist = np.linalg.norm(self.to_button(physics) - physics.data.body("bellows_arm/eef_ball/").xpos)
        if physics.data.time < self._max_duration * 0.7:
            btn_eef_dist = 0.0
        return (
            vertical_momentum
            - btn_eef_dist  # proxy for linear velocity to goal
            + smash_force
        ) / 1000.0  # normalise reward, force is in the range of 1000s

    def should_terminate_episode(self, physics):  # pylint: disable=unused-argument
        """Determines whether the episode should terminate given the physics state.

        Args:
        physics: A Physics object

        Returns:
        A boolean
        """
        if physics.data.time > self._max_duration:
            return True
        return False


class Throw(composer.Task):
    def __init__(self, bellowsarm, inner_num_substeps, gripper, max_duration):
        """
        swing the arm as fast as possible
        The actions are mapped into range [0, 1], for modelling and RL purposes.
        But we scale them back into the range [0, Pmax] in env.step()

        """
        # world plane & light
        self._arena = floors.Floor()
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))  # 4m high light
        # arm
        self._arm = bellowsarm

        # gripper
        # TODO: define this as attachment, i.e.,
        # self._gripper = self._arm._mjcf_model.worldbody._attachments[0]
        self._gripper = gripper
        self._max_duration = max_duration
        self.release_time = max_duration

        self._arena.attach(self._arm)

        # Add throwing cube as a free entity
        self._cube = objects.Cube()
        self._arena.add_free_entity(self._cube)

        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # enable observables (add corruptors if any, here)

        # individual bellows disk positions and velocities (54 by default)
        self._arm.observables.joint_positions.enabled = False
        self._arm.observables.joint_velocities.enabled = False

        # velocimeter mesurement of end effector x,y,z velocities
        self._arm.observables.eef_velocities.enabled = True
        # add velocimeter + (joint states (u,v)+ (u_dot, v_dot) for all bellows "joints")
        self._arm.observables.joint_states.enabled = False

        self._task_observables = {}
        self._task_observables[self._cube.mjcf_model.model + "/positions"] = observable.Generic(self.to_cube)
        for obs in self._task_observables.values():
            obs.enabled = True  # enable all observables

        self.inner_num_substeps = inner_num_substeps

        # skipping task observables

        # determine delay in seconds between action updates
        self.control_timestep = self.inner_num_substeps * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def to_cube(self, physics):
        """Returns the vector from base of arm to the cube in the local frame of the base"""
        cube_global_frame_pos, _ = self._cube.get_pose(physics)
        # assuming self._arm is type composer.Entity
        assert isinstance(self._arm, composer.Entity)
        return self._arm.global_vector_to_local_frame(physics, cube_global_frame_pos)

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
            minimum=0.0,
            maximum=1.0,
            name="\t".join(names),
        )

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        # TODO: add code for setting initial joint positions
        # make cube spawn at gripper fingers
        # use xmat for access to rotation matrix
        # self._gripper._mjcf_model.worldbody.find("body", "right_pad")
        r_finger_pos = physics.named.data.geom_xpos["bellows_arm/robotiq_2f85/right_pad1"]
        l_finger_pos = physics.named.data.geom_xpos["bellows_arm/robotiq_2f85/left_pad1"]
        l_finger_mat = physics.named.data.geom_xmat["bellows_arm/robotiq_2f85/left_pad1"]

        # get midpoint between fingers
        gripper_pos = (r_finger_pos + l_finger_pos) / 2

        # compute quartenion from rotation matrix
        gripper_ori = np.zeros(4)
        mjlib.mju_mat2Quat(gripper_ori, l_finger_mat)
        self._cube.set_pose(physics, position=gripper_pos, quaternion=gripper_ori)

    def get_reward(self, physics):
        """Measures the speed of the end effector position."""
        # end_effector_speed = np.linalg.norm(physics.velocity(physics.model.geom("2_G9")))
        eef_vel = self._arm.observables.eef_velocities(physics)

        if self.should_terminate_episode(physics):
            return self.get_final_reward(physics)
        elif physics.data.time < self.release_time:
            return np.linalg.norm(eef_vel)
        else:
            return 0.0

    def get_final_reward(self, physics):
        """Computes distance to the cube at the final timestep."""

        return np.linalg.norm(self.to_cube(physics))

    def should_terminate_episode(self, physics):  # pylint: disable=unused-argument
        """Determines whether the episode should terminate given the physics state.

        Args:
        physics: A Physics object

        Returns:
        A boolean
        """
        if physics.data.time > self._max_duration:
            return True
        return False


class Reach(composer.Task):
    def __init__(self, bellowsarm, inner_num_substeps, outer_num_substeps, gripper, max_duration):
        """
        swing the arm as fast as possible
        The actions are mapped into range [0, 1], for modelling and RL purposes.
        But we scale them back into the range [0, Pmax] in env.step()

        """
        # world plane & light
        self._arena = floors.Floor()
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))  # 4m high light
        # arm
        self._arm = bellowsarm

        # gripper
        # TODO: define this as attachment, i.e.,
        # self._gripper = self._arm._mjcf_model.worldbody._attachments[0]
        self._gripper = gripper
        self._max_duration = max_duration
        self.release_time = max_duration

        self._arena.attach(self._arm)

        # Add reach target as position
        self.reach_pos = np.array([0.5, 0.5, 1.0])
        self.reach_std = 0.75  # in meters
        self._ball = objects.TransparentSphere()

        # attach ball to arena
        self._arena.attach(self._ball)
        # self._arena.add_fixed_entity(self._ball)

        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # enable observables (add corruptors if any, here)

        # individual bellows disk positions and velocities (54 by default)
        self._arm.observables.joint_positions.enabled = False
        self._arm.observables.joint_velocities.enabled = False

        # velocimeter mesurement of end effector x,y,z velocities
        self._arm.observables.eef_velocities.enabled = True
        # add velocimeter + (joint states (u,v)+ (u_dot, v_dot) for all bellows "joints")
        self._arm.observables.joint_states.enabled = True

        self._task_observables = {}
        self._task_observables[self._ball.mjcf_model.model + "/positions"] = observable.Generic(self.to_ball)
        for obs in self._task_observables.values():
            obs.enabled = True  # enable all observables

        self.inner_num_substeps = inner_num_substeps
        self.outer_num_substeps = outer_num_substeps

        # skipping task observables

        # determine delay in seconds between action updates
        self.control_timestep = self.inner_num_substeps * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def to_ball(self, physics):
        """Returns the vector from base of arm to the ball in the local frame of the base"""
        ball_global_frame_pos, _ = self._ball.get_pose(physics)
        # assuming self._arm is type composer.Entity
        assert isinstance(self._arm, composer.Entity)
        return self._arm.global_vector_to_local_frame(physics, ball_global_frame_pos)

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
            minimum=0.0,
            maximum=1.0,
            name="\t".join(names),
        )

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        self.num_hits = 0.0

        # TODO: add variator for initial ball position
        self._ball.set_pose(physics, position=self.reach_pos, quaternion=np.array([0.0, 0.0, 0.0, 1.0]))

    def get_reward(self, physics):
        """Measures the speed of the end effector position."""

        ball_pos = self.to_ball(physics)

        r_finger_pos = physics.named.data.geom_xpos["bellows_arm/robotiq_2f85/right_pad1"]
        l_finger_pos = physics.named.data.geom_xpos["bellows_arm/robotiq_2f85/left_pad1"]
        # l_finger_mat = physics.named.data.geom_xmat[
        #     "bellows_arm/robotiq_2f85/left_pad1"
        # ]
        eef_vel = self._arm.observables.eef_velocities(physics)
        # get midpoint between fingers
        gripper_pos = (r_finger_pos + l_finger_pos) / 2.0
        normed_dist = np.linalg.norm(ball_pos - gripper_pos)

        # TODO: Tune reward term coefficients
        if normed_dist < 0.125:  # true if within 7.5 cm of target
            self.num_hits += 1.0 / self.outer_num_substeps
        # else:
        #     self.num_hits = 0.0 # reset every time we miss the target

        gauss_dist = np.exp(-0.5 * normed_dist / self.reach_std**2)

        if self.should_terminate_episode(physics):
            return gauss_dist - np.linalg.norm(eef_vel) / self._max_duration + self.num_hits / self._max_duration
        else:
            return gauss_dist - np.linalg.norm(eef_vel) / self._max_duration + self.num_hits / self._max_duration

    def should_terminate_episode(self, physics):  # pylint: disable=unused-argument
        """Determines whether the episode should terminate given the physics state.

        Args:
        physics: A Physics object

        Returns:
        A boolean
        """
        if physics.data.time > self._max_duration:
            return True
        return False


class BellowsEnv(composer.Environment):
    def __init__(
        self,
        task_str="swing",
        max_pressures=275.8e3,
        num_substeps=500,
        order_hold_ctrl=False,
    ) -> None:
        """
        inner_num_substeps: number of times mujoco executes the same action within every env.step() call.
        """

        tasks = ["swing", "smash", "throw", "reach"]
        assert task_str in tasks, f"Invalid task string, valid options include {tasks}"

        self.arm = bellows_arm.BellowsArm(name="bellows_arm", num_disks=10, floor_and_lights=False)

        self.order_hold_ctrl = order_hold_ctrl

        if self.order_hold_ctrl:
            self.outer_num_substeps = num_substeps
            self.inner_num_substeps = 1

            self.prev_scaled_action = None
        else:
            # Always handle inner steps outside?
            self.outer_num_substeps = num_substeps
            self.inner_num_substeps = 1

        # for env termination
        self._max_duration = 5.0  # seconds

        if task_str == "swing":
            # inner_num_substeps available in the env (parent class) via env._n_sub_steps
            gripper = grippers.Robotiq2F85Gripper()
            self.arm.attach(gripper)
            self.DOF = 13
            task = SwingFast(self.arm, self.inner_num_substeps)

        elif task_str == "smash":
            gripper = grippers.HeavyBallGripper()
            self.arm.attach(gripper)
            self.DOF = 12
            task = Smash(self.arm, self.inner_num_substeps, max_duration=self._max_duration)
        elif task_str == "throw":
            gripper = grippers.Robotiq2F85Gripper()
            self.arm.attach(gripper)
            self.DOF = 13
            task = Throw(self.arm, self.inner_num_substeps, gripper, self._max_duration)
        elif task_str == "reach":
            gripper = grippers.Robotiq2F85Gripper()
            self.arm.attach(gripper)
            self.DOF = 13
            task = Reach(
                self.arm,
                self.inner_num_substeps,
                self.outer_num_substeps,
                gripper,
                self._max_duration,
            )

        self.MAX_PRESSURES = max_pressures

        # base pressures
        self.base_pressures = np.zeros(self.DOF)  # these go up self.MAX_PRESSURES

        self._horizon = int(self._max_duration / task.control_timestep)

        super().__init__(task, strip_singleton_obs_buffer_dim=True, time_limit=self._max_duration)

    @property
    def max_duration(self):
        return self._max_duration

    @property
    def horizon(self):
        return self._horizon

    def step(self, action, debug=False):
        """Scales actions to the range [0, Pmax] from [0, 1 ] and calls the parent step method
        This assumes that the parent step method does not check for action bounds.
        """
        if self._reset_next_step:
            self._reset_next_step = False
            return self.reset()
        substep_obs = []
        substep_acts = []
        scaled_action = np.array(self.MAX_PRESSURES * action)
        # first order hold control
        if not self.order_hold_ctrl:
            for tau in range(self.outer_num_substeps):
                time_step = super().step(scaled_action)

                if debug:
                    substep_obs.append(time_step)
                    substep_acts.append(scaled_action)
        else:
            interpolated_actions = np.linspace(self.prev_scaled_action, scaled_action, self.outer_num_substeps)
            for act in interpolated_actions:
                time_step = super().step(act)

                if debug:
                    substep_obs.append(time_step)
                    substep_acts.append(act)

            self.prev_scaled_action = scaled_action

        if debug:
            return substep_obs, substep_acts

        if time_step.step_type == StepType.LAST:
            self._reset_next_step = True

        return time_step

    def reset(self):
        self._reset_next_step = False
        if self.order_hold_ctrl:
            self.prev_scaled_action = self.base_pressures

        return super().reset()
