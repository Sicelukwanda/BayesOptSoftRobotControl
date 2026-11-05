"""Unified experiment logger supporting TensorBoard (default), optional Weights & Biases, and local JSON/NPZ output."""

import json
import logging
import os
from typing import Any, Dict, Optional

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Logs optimization trials to TensorBoard by default, with optional W&B and local file persistence."""

    def __init__(
        self,
        run_name: str,
        config: Dict[str, Any],
        logdir: Optional[str] = "runs",
        save_dir: Optional[str] = "data",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "bellows-bo",
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online",
    ):
        self.run_name = run_name
        self.config = config
        self.logdir = logdir
        self.save_dir = save_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.history = []

        self.tb_writer: Optional[SummaryWriter] = None
        if self.use_tensorboard and self.logdir:
            tb_path = os.path.join(self.logdir, self.run_name)
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_path)
            self.tb_writer.add_text("config", json.dumps(self.config, indent=2, default=str), 0)

        self.wandb_run = None
        if self.use_wandb:
            import wandb

            os.environ["WANDB_MODE"] = wandb_mode
            init_kwargs: Dict[str, Any] = {
                "project": wandb_project,
                "name": run_name,
                "config": config,
                "reinit": True,
            }
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity
            self.wandb_run = wandb.init(**init_kwargs)

    def log_step(self, step: int, metrics: Dict[str, float]):
        record = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self.history.append(record)

        if self.tb_writer is not None:
            for k, v in metrics.items():
                tag = k.replace(" ", "_")
                self.tb_writer.add_scalar(f"optimization/{tag}", float(v), int(step))

        if self.wandb_run is not None:
            import wandb

            wandb.log({**metrics, "trial": int(step)})

    def log_video(
        self,
        step: int,
        frames: np.ndarray,
        fps: int = 25,
        tag: str = "policy/rollout",
        filename_suffix: Optional[str] = None,
    ) -> Optional[str]:
        """Save rollout frames (T, H, W, C) uint8 as an MP4 file and log to TensorBoard and/or W&B."""
        mp4_path: Optional[str] = None
        base_video_dir: Optional[str] = None
        if self.save_dir:
            base_video_dir = os.path.join(self.save_dir, "videos")
        elif self.logdir:
            base_video_dir = os.path.join(self.logdir, self.run_name, "videos")

        if base_video_dir is not None:
            os.makedirs(base_video_dir, exist_ok=True)
            suffix = filename_suffix if filename_suffix else f"trial_{step}"
            mp4_path = os.path.join(base_video_dir, f"{self.run_name}_{suffix}.mp4")
            imageio.mimsave(mp4_path, frames, fps=fps)
            logging.info(f"Saved rollout video to {mp4_path}")

        if self.tb_writer is not None:
            vid_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
            self.tb_writer.add_video(tag, vid_tensor, global_step=int(step), fps=fps)
            self.tb_writer.flush()

        if self.wandb_run is not None:
            import wandb

            video_input: Any = mp4_path if mp4_path is not None else frames.transpose(0, 3, 1, 2)
            wandb.log({tag: wandb.Video(video_input, fps=fps, format="mp4"), "trial": int(step)})

        return mp4_path

    def save_results(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        best_x: torch.Tensor,
        best_y: float,
        extra_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.save_dir:
            return None

        os.makedirs(self.save_dir, exist_ok=True)
        npz_path = os.path.join(self.save_dir, f"{self.run_name}.npz")
        json_path = os.path.join(self.save_dir, f"{self.run_name}.json")

        np.savez(
            npz_path,
            X=X.detach().cpu().numpy(),
            Y=Y.detach().cpu().numpy(),
            best_x=best_x.detach().cpu().numpy(),
            best_y=np.array(float(best_y)),
        )

        payload = {
            "run_name": self.run_name,
            "config": self.config,
            "best_y": float(best_y),
            "best_x": best_x.detach().cpu().numpy().tolist(),
            "summary": extra_summary or {},
            "history": self.history,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logging.info(f"Saved run results to {json_path} and {npz_path}")
        return json_path

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb

            wandb.finish()
