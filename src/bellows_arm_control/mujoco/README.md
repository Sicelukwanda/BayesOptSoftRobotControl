# Curtis' Mujoco Sandbox

## Important things to Know:

* bellows_arm.py uses DeepMind's dm_control package to generate mjcf files programatically. See the code for details on what it does. In summary, this is the place to change your model, since the xml file is really hard to work with directly. [dm_control](https://github.com/deepmind/dm_control) is a separate packaege specifically written for RL. May be a good place to start with RL. The API exposes a way to create your own custom environments. 
* The STL files in this directory are loaded at runtime by mujoco. They come from the cad files from the bellows arm. 
* You can install mujoco a few different ways. You can simply download precompiled binaries (in C) and some examples from the [github release](https://github.com/deepmind/mujoco/releases). If you intend mostly on using mujoco from Python, the easiest way is to install mujoco via [pip](https://mujoco.readthedocs.io/en/stable/python.html). This comes with the source code wrapped via pybind11. Obviously, the downside here is some speed loss due to python being...slow.
* simulate_bellows_arm.py loads the mjcf file generated in bellows_arm.py and then simulates in programatically. This is the main entry point for running a simulation. Like I mentioned above, for RL applications, might be worth wrapping the model in a custom env using the dm_control composer.
* If you want to just run the simulation (without any control), you can run the following and then examine the code (cd into project root folder):
```shell 
python3 simulate.py
```


## Dependencies and Installation
I have specific versions mentioned here, but DeepMind says they are committed to backwards compatibilty and version matching for both libraries, so hopefully installing newest versions later will be fine. 

```shell
pip3 install dm_control==1.0.11
pip3 install mujoco==2.3.3
```




