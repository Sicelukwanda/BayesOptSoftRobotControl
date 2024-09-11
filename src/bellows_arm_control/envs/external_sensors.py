from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable


class ForceButton(composer.Entity):
    """A Button which changes color when pressed with a certain force"""

    def _build(self, num_substeps, target_force_range=(5, 100)):
        # TODO: verify the units of the force
        # TODO: verify that min force is not too high
        self._min_force, self._max_force = target_force_range
        self._mjcf_model = mjcf.RootElement(model="force_sensor")
        self._geom = self._mjcf_model.worldbody.add(
            "geom",
            type="cylinder",
            size=[0.25, 0.02],
            rgba=[1, 0, 0, 1],
        )
        # TODO: what are sites? so far it looks like they are similar to collison meshes?
        self._site = self._mjcf_model.worldbody.add(
            "site",
            type="cylinder",
            size=self._geom.size * 1.01,
            rgba=[1, 0, 0, 0.1],
        )
        # TODO tourch sensor vs force sensor?
        self._sensor = self._mjcf_model.sensor.add(
            "touch",
            site=self._site,
        )
        self._num_activated_steps = 0
        self._num_substeps = num_substeps

    def _build_observables(self):
        return ButtonObservables(self)

    @property
    def num_substeps(self):
        return self._num_substeps

    @property
    def mjcf_model(self):
        return self._mjcf_model

    def _update_activation(self, physics):
        """Update the activation and colour if the desired force is applied"""
        # get element at 0 because we get a SynchronizingArrayWrapper here
        current_force = physics.bind(self._sensor).sensordata[0]

        is_activated = (current_force >= self._min_force) and (current_force <= self._max_force)

        red = [1, 0, 0, 1]
        green = [0, 1, 0, 1]

        # why do we need to bind the physics to the geom?
        physics.bind(self._geom).rgba = green if is_activated else red
        self._num_activated_steps += 1 if is_activated else 0
        self._num_activated_steps += int(is_activated)

    def initialize_episode(self, physics, random_state):
        self._reward = 0.0
        self._num_activated_steps = 0
        self._update_activation(physics)

    def after_substep(self, physics, random_state):
        self._update_activation(physics)

    @property
    def touch_sensor(self):
        return self._sensor

    @property
    def num_activated_steps(self):
        return self._num_activated_steps


class ButtonObservables(composer.Observables):
    """A touch sensor which averages contact force over physics substeps.
    This button will count the number of substeps the button is pressed for with the desired force.
    These actual force value will be averaged over the actual.
    """

    @composer.observable
    def touch_force(self):
        return observable.MJCFFeature(
            "sensordata",
            self._entity.touch_sensor,
            buffer_size=self._entity.num_substeps,  # save every substep
            aggregator="mean",
        )
