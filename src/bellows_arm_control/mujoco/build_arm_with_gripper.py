# ruff: noqa
from dm_control import mjcf
import numpy as np
import os

_BELLOWS_MUJOCO_ROOT = os.path.dirname(__file__)


class BellowsArm:
    def __init__(self, name, num_disks) -> None:
        # set default options and visual things.
        self.mjcf_model = mjcf.RootElement(model=name)

        self.ORANGE = [0.8, 0.2, 0.1, 1]
        # self.ORANGE = [15 / 256.0, 10 / 256.0, 222 / 256.0, 1]
        self.X = [1, 0, 0]
        self.Y = [0, 1, 0]

        self.setCompiler()
        self.setOptions()
        self.setSimSize()
        self.setVisual()
        self.addAssets()

        # some joint measurements common (hopefully) among all joints
        self.joint_height = 0.2  # length between endplates (m)
        self.num_joints = 3
        self.num_disks = num_disks
        num_spaces = self.num_disks - 1
        self.disk_height = self.joint_height / (self.num_disks + num_spaces)
        self.disk_half_height = self.disk_height / 2

        self.setDefaults()
        # create world plane
        self.mjcf_model.worldbody.add(
            "geom",
            condim=1,
            material="matplane",
            name="world",
            size=[0, 0, 1],
            type="plane",
        )
        self.mjcf_model.worldbody.add(
            "light",
            diffuse=[0.6, 0.6, 0.6],
            dir=[0, 0, -1],
            directional="true",
            pos=[0, 0, 4],
            specular=[0.2, 0.2, 0.2],
        )

        # TODO: add createObject to be manipulated here.
        # pos = (-.5, .5)m, box of .5 m side,

        # Todo: add tactile/touch sensor to foamy covers over links Is

        base = self.createBase(self.mjcf_model.worldbody)
        last_disk = self.create8bellowsjoint(base, 0, 0.125, 2.653)
        link = self.addLink0(last_disk)
        last_disk = self.create4bellowsjoint(
            link,
            1,
            0.08,
            1.326,
            [-0.0223, 0.0223, 0.282],
            [1, 1, 0, np.radians(-45)],
        )
        link = self.addLink1(last_disk)
        self.last_disk = self.create4bellowsjoint(
            link,
            2,
            0.08,
            1.326,
            [-0.103, 0.103, 0.190],
            [1, 1, 0, np.radians(-45)],
        )

        self.createActuators()
        self.createSensors()

    def createActuators(self):
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
            xtendon = self.mjcf_model.tendon.add("fixed", name=f"x{j}")
            ytendon = self.mjcf_model.tendon.add("fixed", name=f"y{j}")

            # add actuators along tendons
            # ====== X AXIS =========
            # bellows that creates positive rotation
            self.mjcf_model.actuator.add(
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
            self.mjcf_model.actuator.add(
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
            self.mjcf_model.actuator.add(
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
            self.mjcf_model.actuator.add(
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

    def createSensors(self):
        # the values for these sensors are in mjData/sensordata, which is stored as an array of nsensordata x 1
        self.mjcf_model.sensor.add("tendonpos", name="u0", tendon="x0")
        self.mjcf_model.sensor.add("tendonpos", name="v0", tendon="y0")
        self.mjcf_model.sensor.add("tendonpos", name="u1", tendon="x1")
        self.mjcf_model.sensor.add("tendonpos", name="v1", tendon="y1")
        self.mjcf_model.sensor.add("tendonpos", name="u2", tendon="x2")
        self.mjcf_model.sensor.add("tendonpos", name="v2", tendon="y2")
        #
        self.mjcf_model.sensor.add("tendonvel", name="ud0", tendon="x0")
        self.mjcf_model.sensor.add("tendonvel", name="vd0", tendon="y0")
        self.mjcf_model.sensor.add("tendonvel", name="ud1", tendon="x1")
        self.mjcf_model.sensor.add("tendonvel", name="vd1", tendon="y1")
        self.mjcf_model.sensor.add("tendonvel", name="ud2", tendon="x2")
        self.mjcf_model.sensor.add("tendonvel", name="vd2", tendon="y2")

        # self.mjcf_model.sensor.add("actuatorfrc", name="act_force", actuator="p0_j0")

    def addLink0(self, body):
        link = body.add("body", name="link0", pos=[0, 0, self.disk_half_height])
        link.add("inertial", pos=[0, 0, 0.115], diaginertia=[0.108, 0.108, 0.023], mass=3.881)
        # TODO: fix this inertial frame pos to account for valve block and stuff. Not sure what pos is relative to.
        link.add("geom", name="link0", type="mesh", mesh="link0", rgba=[0.5, 0.5, 0.5, 1])

        return link

    def addLink1(self, body):
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

    def create8bellowsjoint(self, body, joint_num, joint_radius, joint_mass):
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

    def create4bellowsjoint(self, body, joint_num, joint_radius, joint_mass, pos, axisangle):
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

    def createBase(self, body):
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

    def setCompiler(self):
        self.mjcf_model.compiler.angle = "radian"

    def setOptions(self):
        self.mjcf_model.option.set_attributes(
            timestep=0.001,
            iterations=50,
            solver="Newton",
            jacobian="sparse",
            cone="elliptic",
            tolerance=1e-10,
        )

        self.mjcf_model.option.flag.set_attributes(gravity="enable")

    def setSimSize(self):
        self.mjcf_model.size.set_attributes(njmax=5000, nconmax=5000, nstack=5000000)

    def setVisual(self):
        # visual already has all possible children elements created, so just change them here.
        self.mjcf_model.visual.quality.shadowsize = 2048
        self.mjcf_model.visual.map.set_attributes(stiffness=7015, fogstart=10, fogend=15, zfar=40, shadowscale=0.5)
        self.mjcf_model.visual.rgba.haze = [0.15, 0.25, 0.35, 1]

    def addAssets(self):
        # add children elements
        self.mjcf_model.asset.add(
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

        self.mjcf_model.asset.add(
            "material",
            name="matplane",
            texture="texplane",
            texuniform="true",
            reflectance=0.3,
        )

        self.mjcf_model.asset.add(
            "mesh",
            file=_BELLOWS_MUJOCO_ROOT + "/link0.stl",
        )
        self.mjcf_model.asset.add(
            "mesh",
            file=_BELLOWS_MUJOCO_ROOT + "/link1.stl",
        )

    def setDefaults(self):
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

        # create default class for 8 bellows disks. Then I use this as childclass so that all elements in a given body default to these settings, unless overwritten.
        disk_class = self.mjcf_model.default.add("default", dclass="8bellows")
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

        # create default class for 4 bellows disks. Then I use this as childclass so that all elements in a given body default to these settings, unless overwritten.
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


if __name__ == "__main__":
    arm = BellowsArm(
        "bellows_arm",
        10,
    )

    last_disk = arm.last_disk
    eef_site = last_disk.add(
        "site",
        name="eef_site",
        pos=[0, 0, arm.disk_height + arm.disk_half_height],
        rgba=[0.0, 0.0, 1.0, 1.0],
    )
    gripper_mjcf = mjcf.from_path(_BELLOWS_MUJOCO_ROOT + "/robotiq_2f85/2f85.xml")

    eef_site.attach(gripper_mjcf)

    # Add gripper
    arm_mjcf = arm.mjcf_model
    xml = arm_mjcf.to_xml_string()

    # bandaid for weird bug
    xml = xml.replace("link0-3a58470d5f47619f3061a26dc14b3b43155780fe.stl", "link0.stl")
    xml = xml.replace("link1-136477aad1ed3510896f193f57b72a429b1e3222.stl", "link1.stl")
    xml = xml.replace(
        "base_mount-22e57178defe77afb6bdd333bfae16607f1eb3dd.stl", "base_mount.stl"
    )
    xml = xml.replace(
        "driver-97efa43184c575b31ff1f3980641896f51492762.stl", "driver.stl"
    )
    xml = xml.replace("base-e5dacbcc3971bfdb549ff8c7d38ea873ca7f2933.stl", "base.stl")
    xml = xml.replace(
        "coupler-0a4240dc94992944cca6ec9e270ff1658fa86c55.stl", "coupler.stl"
    )
    xml = xml.replace(
        "follower-39e4b8048f1395ee38cb45b37b5fec0e6f2aaec9.stl", "follower.stl"
    )
    xml = xml.replace("pad-e6a633b2c81740b1f783ec4c6e695c8cc570f09d.stl", "pad.stl")
    xml = xml.replace(
        "silicone_pad-c284384f3ca6dcdc24d6188a5d1a2d4c42c412ac.stl", "silicone_pad.stl"
    )
    xml = xml.replace(
        "spring_link-8f50234325193b84e9f86b7a63a24560a389c9bf.stl", "spring_link.stl"
    )
    # to actually write xml file. There's a weird bug in the stl that you need to fix.
    f = open("bellows_arm_hand2.mjcf", "w")
    f.write(xml)
    f.close()
