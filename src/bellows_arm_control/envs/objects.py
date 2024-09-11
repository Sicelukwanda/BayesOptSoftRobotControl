from dm_control import composer
from dm_control import mjcf


class Cube(composer.ModelWrapperEntity):
    """A Robotiq 2F85 gripper from MuJoCo Menagerie."""

    def _build(self, name=None):
        if name is None:
            name = "cube"
        mjcf_model = mjcf.RootElement(model=name)
        mjcf_model.compiler.angle = "radian"

        length = 0.025
        _ = mjcf_model.worldbody.add(
            "geom",
            type="box",
            size=[length, length, length],
            rgba=[0.2, 0.7, 0, 1],
            pos=[0, 0, length / 2.0],  # origin offset
            mass=0.25,
        )

        return super()._build(mjcf_model)


class TransparentSphere(composer.ModelWrapperEntity):
    """A transparent sphere that does not collide with other objects."""

    def _build(self, name=None):
        if name is None:
            name = "reach_target"
        mjcf_model = mjcf.RootElement(model=name)
        mjcf_model.compiler.angle = "radian"

        radius = 0.08
        _ = mjcf_model.worldbody.add(
            "geom",
            type="sphere",
            size=[radius],
            rgba=[0.2, 0.7, 0, 0.5],
            pos=[0.0, 0.0, 0.0],  # origin offset
            mass=0.0,  # mass of zero means gravity does not affect it
            contype=0,  # no contact type
            conaffinity=0,  # no contact affinity
        )

        return super()._build(mjcf_model)
