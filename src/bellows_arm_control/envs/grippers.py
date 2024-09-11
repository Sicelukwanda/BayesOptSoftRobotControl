import os

from dm_control import composer
from dm_control import mjcf

_ROBOTIQ_2F85_XML_PATH = os.path.join(os.path.dirname(__file__), "../mujoco/robotiq_2f85/2f85.xml")


class Robotiq2F85Gripper(composer.ModelWrapperEntity):
    """A Robotiq 2F85 gripper from MuJoCo Menagerie."""

    def _build(self, name=None):
        mjcf_model = mjcf.from_path(_ROBOTIQ_2F85_XML_PATH)
        if name:
            mjcf_model.model = name
        return super()._build(mjcf_model)


class HeavyBallGripper(composer.ModelWrapperEntity):
    """A Ball gripper."""

    def _build(self, name=None):
        if name is None:
            name = "eef_ball"
        mjcf_model = mjcf.RootElement(model=name)
        mjcf_model.compiler.angle = "radian"

        radius = 0.08
        _ = mjcf_model.worldbody.add(
            "geom",
            type="sphere",
            size=[radius],
            rgba=[0.2, 0.7, 0, 1],
            pos=[0, 0, radius],
            mass=2.0,
        )

        return super()._build(mjcf_model)
