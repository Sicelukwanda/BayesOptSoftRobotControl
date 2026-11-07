"""Run Global Optimization baselines (Cross-Entropy Method and Random Search) on Bellows Arm tasks.

Examples:
    # Run Cross-Entropy Method (CEM) on the throwing task for 500 robot evaluations (10 steps of pop_size=50):
    python experiments/run_go.py --env=throw --algo=CEM --act_discretization=2 --num_trials=500

    # Run Random Search (RS) on the hammering (smash) task for 10,000 robot evaluations:
    python experiments/run_go.py --env=smash --algo=RS --act_discretization=5 --num_trials=10000
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from absl import app
from absl import flags
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bellows_arm_control import bayesopt
from bellows_arm_control import envs
from bellows_arm_control.bayesopt.optim import CrossEntropyMethod
from bellows_arm_control.bayesopt.optim import RandomSearch
from experiments.logger import ExperimentLogger

FLAGS = flags.FLAGS


def _define_flags():
    flags.DEFINE_integer("seed", 0, "Random seed.")
    flags.DEFINE_enum("env", "throw", ["throw", "reach", "smash"], "Bellows arm task environment.")
    flags.DEFINE_enum("algo", "CEM", ["CEM", "RS"], "Global Optimization algorithm (CEM or RS).")
    flags.DEFINE_integer(
        "act_discretization",
        2,
        "Discrete action space cardinality P between action spec min and max.",
    )
    flags.DEFINE_integer(
        "num_trials",
        500,
        "Total number of robot trial evaluations to run (each step evaluates population_size trials).",
    )
    flags.DEFINE_integer("num_init_points", 12, "Number of initial random trials to seed the distribution.")
    flags.DEFINE_integer("horizon", 10, "Task horizon H (number of high-level actions in policy).")
    flags.DEFINE_integer("population_size", 50, "Number of candidates evaluated per optimization step.")
    flags.DEFINE_float("elite_ratio", 0.2, "Top proportion of candidates used to update CEM distribution.")
    flags.DEFINE_float("init_variance", None, "Optional initial variance for CEM distribution.")

    # Logging flags
    flags.DEFINE_string("logdir", "runs", "Directory for TensorBoard logs.")
    flags.DEFINE_string("save_dir", "data", "Directory for saving JSON/NPZ trial data.")
    flags.DEFINE_bool("use_tensorboard", True, "Log metrics to TensorBoard.")
    flags.DEFINE_bool("use_wandb", False, "Log metrics to Weights & Biases.")
    flags.DEFINE_string("wandb_project", "bellows-arm-GO", "Weights & Biases project name.")
    flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity/username.")
    flags.DEFINE_string("wandb_mode", "online", "Weights & Biases mode (online, offline, disabled).")
    flags.DEFINE_bool(
        "render_last_trial",
        False,
        "Render and save an MP4 video of the best policy performance after the last trial.",
    )
    flags.DEFINE_bool(
        "render_all_trials",
        False,
        "Render and save an MP4 video of the policy performance for every trial step.",
    )
    flags.DEFINE_integer("video_fps", 25, "Frame rate (FPS) for rendered MP4 rollout videos.")


def run_go_trial(
    env_name: str = "throw",
    algo: str = "CEM",
    act_discretization: int = 2,
    horizon: int = 10,
    num_init_points: int = 12,
    population_size: int = 50,
    elite_ratio: float = 0.2,
    init_variance: Optional[float] = None,
    num_trials: int = 500,
    seed: int = 0,
    logdir: Optional[str] = "runs",
    save_dir: Optional[str] = "data",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "bellows-arm-GO",
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
    render_last_trial: bool = False,
    render_all_trials: bool = False,
    video_fps: int = 25,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Execute a single Global Optimization (CEM or Random Search) experiment run."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = envs.BellowsEnv(task_str=env_name, order_hold_ctrl=True)
    spec = env.action_spec()

    if env_name == "throw":
        policy = bayesopt.SwitchPolicyRelease(env, act_discretization, horizon)
        D = horizon + 1
    else:
        policy = bayesopt.SwitchPolicy(env, act_discretization, horizon)
        D = horizon

    def rl_env_objective(X, render=False):
        """Evaluate one or more parameter vectors on the BellowsEnv task."""
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        N, _ = X.shape
        actions_list = list(map(policy, X.detach().cpu().numpy()))
        y = torch.zeros(N, dtype=torch.float64)
        for i in range(N):
            env.reset()
            total_reward = 0.0
            for action in actions_list[i]:
                time_step = env.step(action)
                total_reward += float(np.asarray(time_step.reward).squeeze())
                if render:
                    print(f"Reward: {time_step.reward}")
            y[i] = total_reward
        return y[:, None]

    arr = np.random.uniform(spec.minimum, spec.maximum, (num_init_points, D))
    X_init = torch.tensor(arr, dtype=torch.float64)
    y_init = rl_env_objective(X_init, render=False)

    bounds = torch.tensor(
        [[float(spec.minimum)] * D, [float(spec.maximum)] * D],
        dtype=torch.float64,
    )

    if algo == "CEM":
        GO = CrossEntropyMethod(
            X_init,
            y_init,
            obj_func=rl_env_objective,
            bounds=bounds,
            pop_size=population_size,
            elite_ratio=elite_ratio,
            init_variance=init_variance,
        )
    elif algo == "RS":
        GO = RandomSearch(
            X_init,
            y_init,
            obj_func=rl_env_objective,
            bounds=bounds,
            pop_size=population_size,
        )
    else:
        raise ValueError(f"Unsupported global optimization algorithm: {algo}")

    num_generations = max(1, int(np.ceil(num_trials / population_size)))
    run_name = f"go_{algo}_{env_name}_P{act_discretization}_seed{seed}"
    config = {
        "algo": algo,
        "env": env_name,
        "act_discretization": act_discretization,
        "horizon": horizon,
        "num_init_points": num_init_points,
        "population_size": population_size,
        "elite_ratio": elite_ratio,
        "init_variance": init_variance,
        "num_trials": num_trials,
        "num_generations": num_generations,
        "seed": seed,
    }

    exp_logger = ExperimentLogger(
        run_name=run_name,
        config=config,
        logdir=logdir,
        save_dir=save_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=f"{wandb_project}-{algo}-{env_name}",
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
    )

    best_at_milestones: Dict[int, float] = {}
    iterator = tqdm(range(num_generations), desc=run_name, disable=not show_progress)
    for i in iterator:
        GO.step()
        trial_count = (i + 1) * population_size
        latest_y = float(GO.Y[-1].item())
        best_y = float(GO.Y[GO.best_idx].item())

        exp_logger.log_step(trial_count, {"y": latest_y, "best_y": best_y})

        if trial_count in (100, 500, 10000, num_generations * population_size):
            best_at_milestones[trial_count] = best_y

        is_last_step = i == num_generations - 1
        if render_all_trials or (render_last_trial and is_last_step):
            best_actions = policy(GO.best_x.detach().cpu().numpy())
            frames, _ = env.render_rollout(best_actions, fps=video_fps)
            exp_logger.log_video(
                step=trial_count,
                frames=frames,
                fps=video_fps,
                tag="policy/rollout",
                filename_suffix=f"trial_{trial_count}",
            )

    final_best_y = float(GO.Y[GO.best_idx].item())
    exp_logger.save_results(
        X=GO.X,
        Y=GO.Y,
        best_x=GO.best_x,
        best_y=final_best_y,
        extra_summary={"best_at_trials": best_at_milestones},
    )
    exp_logger.close()

    return {
        "run_name": run_name,
        "config": config,
        "best_y": final_best_y,
        "best_x": GO.best_x.detach().cpu().numpy(),
        "best_at_trials": best_at_milestones,
    }


def run_experiment(_):
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)-4s %(levelname)s:%(name)s:%(message)s",
    )

    result = run_go_trial(
        env_name=FLAGS.env,
        algo=FLAGS.algo,
        act_discretization=FLAGS.act_discretization,
        horizon=FLAGS.horizon,
        num_init_points=FLAGS.num_init_points,
        population_size=FLAGS.population_size,
        elite_ratio=FLAGS.elite_ratio,
        init_variance=FLAGS.init_variance,
        num_trials=FLAGS.num_trials,
        seed=FLAGS.seed,
        logdir=FLAGS.logdir,
        save_dir=FLAGS.save_dir,
        use_tensorboard=FLAGS.use_tensorboard,
        use_wandb=FLAGS.use_wandb,
        wandb_project=FLAGS.wandb_project,
        wandb_entity=FLAGS.wandb_entity,
        wandb_mode=FLAGS.wandb_mode,
        render_last_trial=FLAGS.render_last_trial,
        render_all_trials=FLAGS.render_all_trials,
        video_fps=FLAGS.video_fps,
    )
    logging.info(f"Best candidate x:\n{result['best_x']}\nBest objective y: {result['best_y']:.4f}")


if __name__ == "__main__":
    _define_flags()
    app.run(run_experiment)
