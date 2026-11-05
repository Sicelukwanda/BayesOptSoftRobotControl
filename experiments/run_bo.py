"""Run Bayesian Optimization experiments on the simulated Bellows Arm environments.

Examples:
    # Run BayesOpt with Log Expected Improvement (LEI) on the throwing task:
    python experiments/run_bo.py --env=throw --acq_func=LEI --act_discretization=2 --num_trials=100

    # Run BayesOpt with UCB (beta=0.9) on the hammering (smash) task:
    python experiments/run_bo.py --env=smash --acq_func=UCB --beta=0.9 --act_discretization=5 --num_trials=100

    # Run BayesOpt with CEM or CMA-ES acquisition refinement:
    python experiments/run_bo.py --env=throw --acq_func=LEI --acq_optim=cem --num_trials=100
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from absl import app
from absl import flags
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bellows_arm_control import bayesopt
from bellows_arm_control import envs
from bellows_arm_control.bayesopt import BayesOpt
from bellows_arm_control.bayesopt import BayesOptCEM
from bellows_arm_control.bayesopt import BayesOptCMAES
from experiments.logger import ExperimentLogger

FLAGS = flags.FLAGS


def _define_flags():
    flags.DEFINE_integer("seed", 0, "Random seed.")
    flags.DEFINE_enum("env", "throw", ["throw", "reach", "smash"], "Bellows arm task environment.")
    flags.DEFINE_integer(
        "act_discretization",
        2,
        "Discrete action space cardinality P between action spec min and max.",
    )
    flags.DEFINE_integer("num_trials", 500, "Number of Bayesian Optimization trials.")
    flags.DEFINE_integer("num_init_points", 12, "Number of random initial points (default 12, i.e., D+1 for throw).")
    flags.DEFINE_float("beta", 0.9, "Exploration parameter kappa/beta for UCB acquisition function.")
    flags.DEFINE_enum(
        "acq_func",
        "LEI",
        ["qEI", "EI", "LEI", "UCB"],
        "Acquisition function to use.",
    )
    flags.DEFINE_enum(
        "acq_optim",
        "botorch",
        ["botorch", "cem", "cmaes"],
        "Optimizer used for maximizing the acquisition function (standard BoTorch L-BFGS-B, +CEM, or +CMA-ES).",
    )
    flags.DEFINE_integer("horizon", 10, "Task horizon H (number of high-level actions in policy).")
    flags.DEFINE_integer("num_restarts", 10, "Number of multi-start restarts for acquisition optimization.")
    flags.DEFINE_integer("raw_samples", 100, "Number of raw Sobol samples for acquisition initialization.")

    # Logging flags
    flags.DEFINE_string("logdir", "runs", "Directory for TensorBoard logs.")
    flags.DEFINE_string("save_dir", "data", "Directory for saving JSON/NPZ trial data.")
    flags.DEFINE_bool("use_tensorboard", True, "Log metrics to TensorBoard.")
    flags.DEFINE_bool("use_wandb", False, "Log metrics to Weights & Biases.")
    flags.DEFINE_string("wandb_project", "bellows-bo", "Weights & Biases project name.")
    flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity/username.")
    flags.DEFINE_string("wandb_mode", "online", "Weights & Biases mode (online, offline, disabled).")


def make_acq_factory(acq_func_name: str, beta: float, seed: int = 0):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=seed)
    if acq_func_name == "UCB":
        return lambda gp: UpperConfidenceBound(gp, beta=beta)
    if acq_func_name == "LEI":
        return lambda gp: LogExpectedImprovement(gp, best_f=gp.train_targets.max())
    if acq_func_name == "EI":
        return lambda gp: ExpectedImprovement(gp, best_f=gp.train_targets.max())
    if acq_func_name == "qEI":
        return lambda gp: qExpectedImprovement(gp, best_f=gp.train_targets.max(), sampler=sampler)
    raise ValueError(f"Unsupported acquisition function: {acq_func_name}")


def run_bo_trial(
    env_name: str = "throw",
    acq_func: str = "LEI",
    acq_optim: str = "botorch",
    beta: float = 0.9,
    act_discretization: int = 2,
    horizon: int = 10,
    num_init_points: int = 12,
    num_trials: int = 500,
    num_restarts: int = 10,
    raw_samples: int = 100,
    seed: int = 0,
    logdir: Optional[str] = "runs",
    save_dir: Optional[str] = "data",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "bellows-bo",
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Execute a single Bayesian Optimization experiment run and return summary metrics."""
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

    acq_factory = make_acq_factory(acq_func, beta=beta, seed=seed)

    if acq_optim == "botorch":
        BO = BayesOpt(X_init, y_init, rl_env_objective, bounds=bounds, acq_func=acq_factory)
    elif acq_optim == "cem":
        BO = BayesOptCEM(
            X_init,
            y_init,
            rl_env_objective,
            bounds=bounds,
            acq_func=acq_factory,
            pop_size=1000,
            elite_size=0.2,
            cem_steps=20,
        )
    elif acq_optim == "cmaes":
        BO = BayesOptCMAES(
            X_init,
            y_init,
            rl_env_objective,
            bounds=bounds,
            acq_func=acq_factory,
            pop_size=100,
            elite_size=0.2,
            maxiter=20,
        )
    else:
        raise ValueError(f"Unsupported acquisition optimizer: {acq_optim}")

    method_tag = f"bo_{acq_optim}_{acq_func}" + (f"_beta{beta}" if acq_func == "UCB" else "")
    run_name = f"{method_tag}_{env_name}_P{act_discretization}_seed{seed}"
    config = {
        "algo": "BayesOpt",
        "env": env_name,
        "acq_func": acq_func,
        "acq_optim": acq_optim,
        "beta": beta,
        "act_discretization": act_discretization,
        "horizon": horizon,
        "num_init_points": num_init_points,
        "num_trials": num_trials,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "seed": seed,
    }

    exp_logger = ExperimentLogger(
        run_name=run_name,
        config=config,
        logdir=logdir,
        save_dir=save_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=f"{wandb_project}-{env_name}",
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
    )

    best_at_milestones: Dict[int, float] = {}
    iterator = tqdm(range(num_trials), desc=run_name, disable=not show_progress)
    for i in iterator:
        BO.step(n_restarts=num_restarts, raw_samples=raw_samples, verbose=False)
        latest_y = float(BO.Y[-1].item())
        best_y = float(BO.Y[BO.best_idx].item())
        trial_num = i + 1
        exp_logger.log_step(trial_num, {"y": latest_y, "best_y": best_y})

        if trial_num in (100, 500, num_trials):
            best_at_milestones[trial_num] = best_y

    final_best_y = float(BO.Y[BO.best_idx].item())
    exp_logger.save_results(
        X=BO.X,
        Y=BO.Y,
        best_x=BO.best_x,
        best_y=final_best_y,
        extra_summary={"best_at_trials": best_at_milestones},
    )
    exp_logger.close()

    return {
        "run_name": run_name,
        "config": config,
        "best_y": final_best_y,
        "best_x": BO.best_x.detach().cpu().numpy(),
        "best_at_trials": best_at_milestones,
    }


def run_experiment(_):
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)-4s %(levelname)s:%(name)s:%(message)s",
    )
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    result = run_bo_trial(
        env_name=FLAGS.env,
        acq_func=FLAGS.acq_func,
        acq_optim=FLAGS.acq_optim,
        beta=FLAGS.beta,
        act_discretization=FLAGS.act_discretization,
        horizon=FLAGS.horizon,
        num_init_points=FLAGS.num_init_points,
        num_trials=FLAGS.num_trials,
        num_restarts=FLAGS.num_restarts,
        raw_samples=FLAGS.raw_samples,
        seed=FLAGS.seed,
        logdir=FLAGS.logdir,
        save_dir=FLAGS.save_dir,
        use_tensorboard=FLAGS.use_tensorboard,
        use_wandb=FLAGS.use_wandb,
        wandb_project=FLAGS.wandb_project,
        wandb_entity=FLAGS.wandb_entity,
        wandb_mode=FLAGS.wandb_mode,
    )
    logging.info(f"Best candidate x:\n{result['best_x']}\nBest objective y: {result['best_y']:.4f}")


if __name__ == "__main__":
    _define_flags()
    app.run(run_experiment)
