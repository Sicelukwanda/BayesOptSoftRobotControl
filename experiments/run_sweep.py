"""Reproduce the benchmark sweeps from the paper (Tables I & II).

Standalone Python sweep runner that executes locally (sequentially or in parallel) and
prints Markdown summary tables of Mean +/- Std across random seeds.

Examples:
    # Preview all sweep configurations without running simulations:
    python experiments/run_sweep.py --dry_run

    # Run a quick sweep for P=2 on throw across 2 seeds:
    python experiments/run_sweep.py --envs=throw --discretizations=2 --seeds=1234,42 --bo_trials=100 --go_trials=100

    # Run the full paper benchmark sweep across 4 parallel workers:
    python experiments/run_sweep.py --num_workers=4
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import itertools
import json
import logging
import os
import sys
from typing import Any, Dict, List

from absl import app
from absl import flags
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_bo import run_bo_trial
from experiments.run_go import run_go_trial

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "seeds",
    ["1234", "42", "1337", "69", "420"],
    "Random seeds to average across (defaults to the 5 seeds used in the paper).",
)
flags.DEFINE_list(
    "discretizations",
    ["2", "5", "7", "10"],
    "Pressure discretization settings P to evaluate.",
)
flags.DEFINE_list(
    "envs",
    ["throw", "smash"],
    "Simulated dynamic tasks to evaluate (throw = Throwing, smash = Hammering).",
)
flags.DEFINE_list(
    "methods",
    ["BayesOpt-UCB", "BayesOpt-LEI", "CEM", "RS"],
    "Optimization methods to include. Also supports BayesOptCEM-LEI, BayesOptCEM-UCB, BayesOptCMAES-LEI.",
)
flags.DEFINE_float("beta", 0.9, "UCB kappa/beta value (0.9 in Tables I & II).")
flags.DEFINE_integer("bo_trials", 500, "Number of trials for BayesOpt methods.")
flags.DEFINE_integer("go_trials", 500, "Number of trials for CEM and Random Search (set to 10000 for 10K column).")
flags.DEFINE_integer("num_init_points", 12, "Initial random trials D_0.")
flags.DEFINE_integer("horizon", 10, "Task horizon H.")
flags.DEFINE_integer("num_workers", 1, "Number of parallel worker processes.")
flags.DEFINE_string("sweep_index", None, "Optional comma-separated list of job indices to run.")
flags.DEFINE_bool("dry_run", False, "Print the generated job configurations and exit.")
flags.DEFINE_string("logdir", "runs", "Directory for TensorBoard logs.")
flags.DEFINE_string("save_dir", "data", "Directory for saving JSON/NPZ results.")
flags.DEFINE_bool("use_tensorboard", True, "Log metrics to TensorBoard.")
flags.DEFINE_bool("use_wandb", False, "Log metrics to Weights & Biases.")
flags.DEFINE_string("wandb_project", "bellows-sweep", "Weights & Biases project name.")
flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity.")
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


def build_sweep_jobs() -> List[Dict[str, Any]]:
    seeds = [int(s) for s in FLAGS.seeds]
    discretizations = [int(p) for p in FLAGS.discretizations]
    envs = [str(e) for e in FLAGS.envs]
    methods = [str(m) for m in FLAGS.methods]

    jobs: List[Dict[str, Any]] = []
    for env_name, method, p, seed in itertools.product(envs, methods, discretizations, seeds):
        jobs.append(
            {
                "env": env_name,
                "method": method,
                "act_discretization": p,
                "seed": seed,
                "beta": FLAGS.beta,
                "bo_trials": FLAGS.bo_trials,
                "go_trials": FLAGS.go_trials,
                "num_init_points": FLAGS.num_init_points,
                "horizon": FLAGS.horizon,
                "logdir": FLAGS.logdir,
                "save_dir": FLAGS.save_dir,
                "use_tensorboard": FLAGS.use_tensorboard,
                "use_wandb": FLAGS.use_wandb,
                "wandb_project": FLAGS.wandb_project,
                "wandb_entity": FLAGS.wandb_entity,
                "render_last_trial": FLAGS.render_last_trial,
                "render_all_trials": FLAGS.render_all_trials,
                "video_fps": FLAGS.video_fps,
            }
        )
    return jobs


def execute_job(job: Dict[str, Any]) -> Dict[str, Any]:
    method = job["method"]
    if method.startswith("BayesOpt"):
        acq_optim = "botorch"
        if method.startswith("BayesOptCEM"):
            acq_optim = "cem"
        elif method.startswith("BayesOptCMAES"):
            acq_optim = "cmaes"

        acq_func = method.split("-")[-1]
        res = run_bo_trial(
            env_name=job["env"],
            acq_func=acq_func,
            acq_optim=acq_optim,
            beta=job["beta"],
            act_discretization=job["act_discretization"],
            horizon=job["horizon"],
            num_init_points=job["num_init_points"],
            num_trials=job["bo_trials"],
            seed=job["seed"],
            logdir=job["logdir"],
            save_dir=job["save_dir"],
            use_tensorboard=job["use_tensorboard"],
            use_wandb=job["use_wandb"],
            wandb_project=job["wandb_project"],
            wandb_entity=job["wandb_entity"],
            render_last_trial=job["render_last_trial"],
            render_all_trials=job["render_all_trials"],
            video_fps=job["video_fps"],
            show_progress=False,
        )
    elif method in ("CEM", "RS"):
        res = run_go_trial(
            env_name=job["env"],
            algo=method,
            act_discretization=job["act_discretization"],
            horizon=job["horizon"],
            num_init_points=job["num_init_points"],
            population_size=50,
            elite_ratio=0.2,
            num_trials=job["go_trials"],
            seed=job["seed"],
            logdir=job["logdir"],
            save_dir=job["save_dir"],
            use_tensorboard=job["use_tensorboard"],
            use_wandb=job["use_wandb"],
            wandb_project=job["wandb_project"],
            wandb_entity=job["wandb_entity"],
            render_last_trial=job["render_last_trial"],
            render_all_trials=job["render_all_trials"],
            video_fps=job["video_fps"],
            show_progress=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "env": job["env"],
        "method": method,
        "act_discretization": job["act_discretization"],
        "seed": job["seed"],
        "best_y": res["best_y"],
        "best_at_trials": res["best_at_trials"],
    }


def format_summary_tables(results: List[Dict[str, Any]], envs: List[str], methods: List[str], ps: List[int]) -> str:
    lines = []
    for env_name in envs:
        lines.append(f"\n### Simulated {env_name.capitalize()} Performance (Best solution so far, Mean +/- Std)")
        header = "| Method | # Trials | " + " | ".join(f"P = {p}" for p in ps) + " |"
        sep = "| :--- | :---: | " + " | ".join(":---:" for _ in ps) + " |"
        lines.append(header)
        lines.append(sep)

        for method in methods:
            milestones = sorted(
                {
                    int(k)
                    for r in results
                    if r["env"] == env_name and r["method"] == method
                    for k in r["best_at_trials"].keys()
                }
            )
            for m_budget in milestones:
                row_cells = []
                for p in ps:
                    vals = [
                        r["best_at_trials"][m_budget]
                        for r in results
                        if r["env"] == env_name
                        and r["method"] == method
                        and r["act_discretization"] == p
                        and m_budget in r["best_at_trials"]
                    ]
                    if vals:
                        row_cells.append(f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}")
                    else:
                        row_cells.append("-")
                lines.append(f"| {method} | {m_budget} | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def main(_):
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)-4s %(levelname)s:%(name)s:%(message)s",
    )

    jobs = build_sweep_jobs()
    if FLAGS.sweep_index is not None:
        indices = [int(idx.strip()) for idx in FLAGS.sweep_index.split(",") if idx.strip()]
        jobs = [jobs[i] for i in indices]

    logging.info(f"Total jobs in sweep: {len(jobs)}")
    if FLAGS.dry_run:
        for idx, job in enumerate(jobs):
            print(
                f"[{idx}] env={job['env']} method={job['method']} "
                f"P={job['act_discretization']} seed={job['seed']}"
            )
        return

    results: List[Dict[str, Any]] = []
    if FLAGS.num_workers <= 1:
        for idx, job in enumerate(jobs):
            logging.info(
                f"Running job {idx + 1}/{len(jobs)}: {job['method']} on {job['env']} "
                f"(P={job['act_discretization']}, seed={job['seed']})"
            )
            results.append(execute_job(job))
    else:
        with ProcessPoolExecutor(max_workers=FLAGS.num_workers) as executor:
            future_to_job = {executor.submit(execute_job, job): job for job in jobs}
            for idx, future in enumerate(as_completed(future_to_job)):
                job = future_to_job[future]
                res = future.result()
                results.append(res)
                logging.info(
                    f"Completed {idx + 1}/{len(jobs)}: {job['method']} on {job['env']} "
                    f"(P={job['act_discretization']}, seed={job['seed']}) -> best_y={res['best_y']:.3f}"
                )

    table_str = format_summary_tables(
        results,
        envs=[str(e) for e in FLAGS.envs],
        methods=[str(m) for m in FLAGS.methods],
        ps=[int(p) for p in FLAGS.discretizations],
    )
    print(table_str)

    if FLAGS.save_dir:
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        summary_path = os.path.join(FLAGS.save_dir, "sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "markdown_table": table_str}, f, indent=2)
        logging.info(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    app.run(main)
