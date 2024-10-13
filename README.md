# Bellows Arm Control

Simulated environment code for the paper  
### Learning Dynamic Tasks on a Large-scale Soft Robot in a Handful of Trials - [Project Page](https://sites.google.com/view/bayesoptsoftrobotcontrol)
**Abstract**
Soft robots offer more flexibility, compliance, and adaptability than traditional rigid robots. They are also typically lighter and cheaper to manufacture. However, their use in real-world applications is limited due to modeling challenges and difficulties in integrating effective proprioceptive sensors. Large-scale soft robots ($\approx$ two meters in length) have greater modeling complexity due to increased inertia and related effects of gravity. Common efforts to ease these modeling difficulties such as assuming simple kinematic and dynamics models also limit the general capabilities of soft robots and are not applicable in tasks requiring fast, dynamic motion like throwing and hammering. To overcome these challenges, we propose a data-efficient Bayesian optimization-based approach for learning control policies for dynamic tasks on a large-scale soft robot. Our approach optimizes the task objective function directly from commanded pressures, without requiring approximate kinematics or dynamics as an intermediate step. We demonstrate the effectiveness of our approach through both simulated and real-world experiments.

## Environments
**Throwing Task:**

<div align="center" style="display: flex; justify-content: center; align-items: flex-start;">

<img src="images/throw.png" alt="Image 1 description" style="max-width: 100%; height: auto;">
</div>

**Hammering Task:**

<div align="center" style="display: flex; justify-content: center; align-items: flex-start;">

<img src="images/hammer.png" alt="Image 1 description" style="max-width: 100%; height: auto;">
</div>

## Installation
*Instructions tested on ubuntu 20.04 and Python3.8*

Clone the repo and run pip install. 
```bash
pip install -e .
```

## ToDo's

- [X] Add Mujoco environments
- [ ] Add BayesOpt Code
- [ ] Add Genetic algorithm baselines

## Bibtex
If you find this useful, please cite the paper!
```bash
@inproceedings{Zwane2024,
 author = {Sicelukwanda Zwane and Daniel G. Cheney and Curtis C Johnson and Yicheng Luo and Yasemin Bekiroglu and Marc Killpack and Marc P. Deisenroth},
 booktitle = {Proceedings of the International Conference on Intelligent Robots and Systems (IROS)},
 date = {2024-10-14},
 title = {Learning Dynamic Tasks on a Large-scale Soft Robot in a Handful of Trials},
 year = {2024}
}
```
