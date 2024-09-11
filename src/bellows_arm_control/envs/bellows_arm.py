# type: ignore
import os
from typing import Optional

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import numpy as np

_BELLOWS_ARM_ROOT = os.path.join(os.path.dirname(__file__), "../mujoco/")


class BellowsArm(composer.Entity):
    """A Composer entity of the Bellows Arm.
    By default, it has an action space of 12 (we command 12 pressures, 4 to each of continuum sections)
    Since the robot approximated with disks and tendon actuators, we have access to the individual joint
    positions of these disks through observation["joint_positions"] or data.qpos if using plain mujoco.
    We also have sensors, one velocimeter measuring the x,y,z velocities at the last disk (in the 3rd section),
    this approximates end-effector velocity. Additionally we have approximate joints states u,v for each section
    have values (totaling 6) and associated angular velocities v_dot, u_dot. We can obtain sensor reading through
    either observation["joint_states"] or through data.sensordata. The sensor array will be ordered as follows
    [velocimeter, states_u_v, angular_vels_u_v].
    TODO:
        - Remove floor from MJCF and add it separately in task
    """

    def _build(self, num_disks: int, name: Optional[str] = None, floor_and_lights: bool = True):
        # set default options and visual things.
        if name is None:
            name = "bellows_arm"
        self._mjcf_root = mjcf.RootElement(model=name)

        self.ORANGE = [0.8, 0.2, 0.1, 1]
        # self.ORANGE = [15 / 256.0, 10 / 256.0, 222 / 256.0, 1]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self._set_compiler()
        self._set_options()
        self._set_sim_size()
        self._set_visual()
        self._add_assets()

        # some joint measurements common (hopefully) among all joints
        self.joint_height = 0.2  # length between endplates (m)
        self.num_joints = 3
        self.num_disks = num_disks
        num_spaces = self.num_disks - 1
        self.disk_height = self.joint_height / (self.num_disks + num_spaces)
        self.disk_half_height = self.disk_height / 2

        self._set_defaults()

        # maybe create world plane
        if floor_and_lights:
            self._create_world_plane_and_lights()

        # TODO: add createObject to be manipulated here.
        # pos = (-.5, .5)m, box of .5 m side,

        # Todo: add tactile/touch sensor to foamy covers over links Is

        base = self._create_base(self.mjcf_model.worldbody)
        last_disk = self.create_8_bellows_joint(base, 0, 0.125, 2.653)
        link = self._add_link0(last_disk)
        last_disk = self._create_4_bellows_joint(
            link,
            1,
            0.08,
            1.326,
            [-0.0223, 0.0223, 0.282],
            [1, 1, 0, np.radians(-45)],
        )
        link = self._add_link1(last_disk)
        self.last_disk = self._create_4_bellows_joint(
            link,
            2,
            0.08,
            1.326,
            [-0.103, 0.103, 0.190],
            [1, 1, 0, np.radians(-45)],
        )

        self._create_actuators()
        self._create_sensors()

        self._eef_site = self.last_disk.add(
            "site",
            name="eef_site",
            pos=[0, 0, self.disk_height + self.disk_half_height],
            rgba=[0.0, 0.0, 1.0, 1.0],
        )
        self._sensor = self._mjcf_root.sensor.add(  # type: ignore
            "velocimeter", site=self._eef_site
        )

    @property
    def eef_vel_sensor(self):
        return self._sensor

    @property
    def attachment_site(self):
        return self._eef_site

    def _build_observables(self):
        return BellowsArmObservables(self)

    def _create_world_plane_and_lights(self):
        # create world plane
        self._mjcf_root.worldbody.add(
            "geom",
            condim=1,
            material="matplane",
            name="world",
            size=[0, 0, 1],
            type="plane",
        )
        self._mjcf_root.worldbody.add(
            "light",
            diffuse=[0.6, 0.6, 0.6],
            dir=[0, 0, -1],
            directional="true",
            pos=[0, 0, 4],
            specular=[0.2, 0.2, 0.2],
        )

    def _create_actuators(self):
        # I use fixed tendons to be able to actuate all the little disk joints together.
        time_consts = [0.2, 0.5, 0.8]  # based on size of valve mostly
        gears = [
            0.1,
            0.05,
            0.05,
        ]  # distance from center of joint to center of bellows, moment arms
        diameters = [
            0.05 * np.sqrt(2),
            0.05,
            0.05,
        ]  # sqrt(2) doubles area for big joint

        Psrc = 40  # psi
        maxP = Psrc * 6895  # pascals

        for j in range(self.num_joints):
            # create both x and y tendons
            xtendon = self._mjcf_root.tendon.add("fixed", name=f"x{j}")
            ytendon = self._mjcf_root.tendon.add("fixed", name=f"y{j}")

            # add actuators along tendons
            # ====== X AXIS =========
            # bellows that creates positive rotation
            self._mjcf_root.actuator.add(
                "cylinder",
                name=f"p0_j{j}",
                tendon=f"x{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[gears[j]],
                timeconst=time_consts[j],
            )
            # bellows that creates negative rotation, switched using gear
            self._mjcf_root.actuator.add(
                "cylinder",
                name=f"p1_j{j}",
                tendon=f"x{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[-gears[j]],
                timeconst=time_consts[j],
            )

            # ========= Y AXIS ===========
            self._mjcf_root.actuator.add(
                "cylinder",
                name=f"p2_j{j}",
                tendon=f"y{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[gears[j]],
                timeconst=time_consts[j],
            )
            # bellows that creates negative rotation, switched using gear
            self._mjcf_root.actuator.add(
                "cylinder",
                name=f"p3_j{j}",
                tendon=f"y{j}",
                diameter=diameters[j],  # m, bellows are 50 mm diameter
                ctrllimited=True,
                ctrlrange=[0, maxP],  # pascals
                gear=[-gears[j]],
                timeconst=time_consts[j],
            )

            # tendon length is the sum of joint angles between each disk. (i.e. q_lumped)
            for n in range(1, self.num_disks):
                xtendon.add("joint", joint=f"{j}_Jx_{n}", coef=1)
                ytendon.add("joint", joint=f"{j}_Jy_{n}", coef=1)

    def _create_sensors(self):
        # the values for these sensors are in mjData/sensordata, which is stored
        # as an array of nsensordata x 1
        self._mjcf_root.sensor.add("tendonpos", name="u0", tendon="x0")
        self._mjcf_root.sensor.add("tendonpos", name="v0", tendon="y0")
        self._mjcf_root.sensor.add("tendonpos", name="u1", tendon="x1")
        self._mjcf_root.sensor.add("tendonpos", name="v1", tendon="y1")
        self._mjcf_root.sensor.add("tendonpos", name="u2", tendon="x2")
        self._mjcf_root.sensor.add("tendonpos", name="v2", tendon="y2")
        #
        self._mjcf_root.sensor.add("tendonvel", name="ud0", tendon="x0")
        self._mjcf_root.sensor.add("tendonvel", name="vd0", tendon="y0")
        self._mjcf_root.sensor.add("tendonvel", name="ud1", tendon="x1")
        self._mjcf_root.sensor.add("tendonvel", name="vd1", tendon="y1")
        self._mjcf_root.sensor.add("tendonvel", name="ud2", tendon="x2")
        self._mjcf_root.sensor.add("tendonvel", name="vd2", tendon="y2")

        # self.mjcf_model.sensor.add("actuatorfrc", name="act_force", actuator="p0_j0")

    def _add_link0(self, body):
        link = body.add("body", name="link0", pos=[0, 0, self.disk_half_height])
        link.add("inertial", pos=[0, 0, 0.115], diaginertia=[0.108, 0.108, 0.023], mass=3.881)
        # TODO: fix this inertial frame pos to account for valve block and stuff. Not sure what pos is relative to.
        link.add("geom", name="link0", type="mesh", mesh="link0", rgba=[0.5, 0.5, 0.5, 1])

        return link

    def _add_link1(self, body):
        link = body.add("body", name="link1", pos=[0, 0, self.disk_half_height])
        link.add(
            "inertial",
            pos=[-0.07, 0.07, 0.13],
            diaginertia=[0.05, 0.05, 0.017],
            mass=3.474,
        )
        # TODO: fix this inertial frame pos to account for valve block and stuff. Not sure what pos is relative to.
        link.add("geom", name="link1", type="mesh", mesh="link1", rgba=[0.5, 0.5, 0.5, 1])

        return link

    def create_8_bellows_joint(self, body, joint_num, joint_radius, joint_mass):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        num_spaces = self.num_disks - 1
        disk_height = self.joint_height / (self.num_disks + num_spaces)
        disk_half_height = disk_height / 2
        disk_mass = joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass * (3 * joint_radius**2 + disk_height**2)) / 12
        Iz = (disk_mass * joint_radius**2) / 2

        # create first body, whose frame is offset
        first_disk = body.add(
            "body",
            name=f"{joint_num}_B0",
            childclass="8bellows",
            pos=[0, 0, 0.14 + disk_half_height],
            euler=[0, 0, -45],
        )
        first_disk.add(
            "geom",
            name=f"{joint_num}_G0",
        )
        first_disk.add("inertial", mass=disk_mass, diaginertia=[Ixy, Ixy, Iz], pos=[0, 0, 0])
        first_disk.add("site", name=f"{joint_num}_bottom", pos=[0, 0, 0])

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{joint_num}_B{i}",
                pos=[0, 0, 2 * disk_height],
            )
            body.add(
                "geom",
                name=f"{joint_num}_G{i}",
            )
            body.add("inertial", mass=disk_mass, diaginertia=[Ixy, Ixy, Iz], pos=[0, 0, 0])
            body.add("joint", name=f"{joint_num}_Jx_{i}", axis=self.X)
            body.add("joint", name=f"{joint_num}_Jy_{i}", axis=self.Y)
            prev_body = body

        return body

    def _create_4_bellows_joint(self, body, joint_num, joint_radius, joint_mass, pos, axisangle):
        # break joint specs in to disk specs
        # total joint -> [disk,space,disk,....,space,disk]
        num_spaces = self.num_disks - 1
        disk_height = self.joint_height / (self.num_disks + num_spaces)
        # disk_half_height = disk_height / 2
        disk_mass = joint_mass / self.num_disks
        # get moment of inertia of each disk (thin cylinder technically). (https://shorturl.at/fsuNO)
        Ixy = (disk_mass * (3 * joint_radius**2 + disk_height**2)) / 12
        Iz = (disk_mass * joint_radius**2) / 2

        # create first body, whose frame is offset
        first_disk = body.add(
            "body",
            name=f"{joint_num}_B0",
            childclass="4bellows",
            pos=pos,  # from pneubotics
            axisangle=axisangle,
        )
        first_disk.add(
            "geom",
            name=f"{joint_num}_G0",
        )
        first_disk.add("inertial", mass=disk_mass, diaginertia=[Ixy, Ixy, Iz], pos=[0, 0, 0])
        first_disk.add("site", name=f"{joint_num}_bottom", pos=[0, 0, 0])

        # for self.num_disks (+1 bc I already made first disk above): create body, add inertial, add geom
        prev_body = first_disk
        for i in range(1, self.num_disks):
            body = prev_body.add(
                "body",
                name=f"{joint_num}_B{i}",
                pos=[0, 0, 2 * disk_height],
            )
            body.add(
                "geom",
                name=f"{joint_num}_G{i}",
            )
            body.add("inertial", mass=disk_mass, diaginertia=[Ixy, Ixy, Iz], pos=[0, 0, 0])
            body.add("joint", name=f"{joint_num}_Jx_{i}", axis=self.X)
            body.add("joint", name=f"{joint_num}_Jy_{i}", axis=self.Y)
            prev_body = body

        return body

    def _create_base(self, body):
        # create box that serves as robot base.
        base = body.add("body", name="base", pos=[0, 0, 0.14], euler=[0, 0, 45])
        # add geom
        base.add(
            "geom",
            name="base",
            pos="0 0 0",
            type="box",
            size=[0.19685, 0.19685, 0.14],
            rgba=[0, 0, 0, 1],
        )

        # add inertial properties
        base.add(
            "inertial",
            mass=0.0136,
            diaginertia=[8.497e-4, 8.497e-4, 1.6992e-3],
            pos=[0, 0, 0],
        )

        return base

    def _set_compiler(self):
        self._mjcf_root.compiler.angle = "radian"

    def _set_options(self):
        self._mjcf_root.option.set_attributes(
            timestep=0.001,
            iterations=50,
            solver="Newton",
            jacobian="sparse",
            cone="elliptic",
            tolerance=1e-10,
        )

        self._mjcf_root.option.flag.set_attributes(gravity="enable")

    def _set_sim_size(self):
        self._mjcf_root.size.set_attributes(njmax=5000, nconmax=5000, nstack=5000000)

    def _set_visual(self):
        # visual already has all possible children elements created, so just change them here.
        self._mjcf_root.visual.quality.shadowsize = 2048
        self._mjcf_root.visual.map.set_attributes(stiffness=7015, fogstart=10, fogend=15, zfar=40, shadowscale=0.5)
        self._mjcf_root.visual.rgba.haze = [0.15, 0.25, 0.35, 1]

    def _add_assets(self):
        # add children elements
        self._mjcf_root.asset.add(
            "texture",
            type="2d",
            name="texplane",
            builtin="checker",
            mark="cross",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.15, 0.2],
            markrgb=[0.8, 0.8, 0.8],
            width=512,
            height=512,
        )

        self._mjcf_root.asset.add(
            "material",
            name="matplane",
            texture="texplane",
            texuniform="true",
            reflectance=0.3,
        )

        self._mjcf_root.asset.add(
            "mesh",
            name="link0",
            file=os.path.join(_BELLOWS_ARM_ROOT, "link0.stl"),
        )
        self._mjcf_root.asset.add(
            "mesh",
            name="link1",
            file=os.path.join(_BELLOWS_ARM_ROOT, "link1.stl"),
        )

    def _set_defaults(self):
        # bellows joint limits
        eight_limit = np.radians(50) / self.num_disks
        four_limit = np.radians(90) / self.num_disks

        joint_radius8 = 0.125
        joint_radius4 = 0.08

        eight_lumped_stiffness = 136  # Nm/rad
        four_lumped_stiffness = 27  # Nm/rad
        eight_lumped_damping = 7.5  #
        four_lumped_damping = 2

        # lumped stiffness/damping uniformly distributed over each disk
        # These are spings/dampers in series, so k_total = k_disk/num_disks
        eight_stiffness = eight_lumped_stiffness * self.num_disks
        eight_damping = eight_lumped_damping * self.num_disks
        four_stiffness = four_lumped_stiffness * self.num_disks
        four_damping = four_lumped_damping * self.num_disks

        # create default class for 8 bellows disks. Then I use this as childclass
        # so that all elements in a given body default to these settings,
        # unless overwritten.
        disk_class = self._mjcf_root.default.add("default", dclass="8bellows")
        disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.ORANGE,
            size=[joint_radius8, self.disk_half_height],
        )
        disk_class.joint.set_attributes(
            type="hinge",
            group=3,
            stiffness=eight_stiffness,
            damping=eight_damping,
            pos=[0, 0, self.disk_height],
            limited="true",
            range=[-eight_limit, eight_limit],
        )

        # create default class for 4 bellows disks. Then I use this as childclass
        # so that all elements in a given body default to these settings,
        # unless overwritten.
        disk_class = self.mjcf_model.default.add("default", dclass="4bellows")
        disk_class.geom.set_attributes(
            type="cylinder",
            rgba=self.ORANGE,
            size=[joint_radius4, self.disk_half_height],
        )
        disk_class.joint.set_attributes(
            type="hinge",
            group=3,
            stiffness=four_stiffness,
            damping=four_damping,
            pos=[0, 0, self.disk_height],
            limited="true",
            range=[-four_limit, four_limit],
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root


class BellowsArmObservables(composer.Observables):
    """simple observable features for joint angles and velocities"""

    @composer.observable
    def joint_positions(self):
        # does not return free joints
        all_joints = self._entity.mjcf_model.find_all("joint")
        # we can access this using observables.qpos
        return observable.MJCFFeature("qpos", all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qvel", all_joints)

    @composer.observable
    def eef_velocities(self):
        """average velocity over all simulator substeps"""
        return observable.MJCFFeature("sensordata", self._entity.eef_vel_sensor)

    @composer.observable
    def joint_states(self):
        """approximate joint positions over all simulator substeps"""
        return observable.MJCFFeature(
            "sensordata",
            list(self._entity.mjcf_model.sensor.tendonpos) + list(self._entity.mjcf_model.sensor.tendonvel),
        )


# arm = BellowsArm(
#             name="bellows_arm", num_disks=10, floor_and_lights=False
#         )

# print(arm._mjcf_root.sensor.all_children())
# breakpoint()
# print(arm.eef_vel_sensor)
# self._mjcf_root.sensor
