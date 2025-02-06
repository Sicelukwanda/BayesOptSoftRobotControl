from absl import app
from dm_control import viewer  # Import the viewer module

# from dm_control.viewer import application
import numpy as np

from bellows_arm_control import envs

class RandomPolicy:
    def __init__(self, env):
        self.i = 0 
        self.env = env
        self.action_spec = env.action_spec()
    def __call__(self,  time_step):
        print("i:", self.i, ", Time elapsed:", self.env.physics.data.time)  
        self.i += 1
        
        return np.random.uniform(
            low=self.action_spec.minimum, high=self.action_spec.maximum, size=self.action_spec.shape
        )

def main(argv):
    """
    Launch the viewer application. Quick reference:
        ['Help', 'F1'],
        ['Info', 'F2'],
        ['Stereo', 'F5'],
        ['Frame', 'F6'],
        ['Label', 'F7'],
        ['--------------', ''],
        ['Pause', 'Space'],
        ['Reset', 'BackSpace'],
        ['Autoscale', 'Ctrl A'],
        ['Geoms', '0 - 4'],
        ['Sites', 'Shift 0 - 4'],
        ['Speed Up', '='],
        ['Slow Down', '-'],
        ['Switch Cam', '[ ]'],
        ['--------------', ''],
        ['Translate', 'R drag'],
        ['Rotate', 'L drag'],
        ['Zoom', 'Scroll'],
        ['Select', 'L dblclick'],
        ['Center', 'R dblclick'],
        ['Track', 'Ctrl R dblclick / Esc'],
        ['Perturb', 'Ctrl [Shift] L/R drag']
    """

    del argv

    env = envs.BellowsEnv(task_str="throw", num_substeps=1, order_hold_ctrl=True) # envs: smash, throw, reach 

    policy = RandomPolicy(env) 

    viewer.launch(env, policy=policy)   
    # run viewer.launch(env) for passsive dynamics (no control)


if __name__ == "__main__":
    app.run(main)