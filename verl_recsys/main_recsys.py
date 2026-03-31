"""Unified recsys entrypoint on top of VeRL PPO runtime."""

from __future__ import annotations

import os
import sys

import hydra
import ray

from omegaconf import OmegaConf

from verl_recsys.acceleration.runtime import apply_acceleration_config

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VERL_DIST_PATH = os.path.join(_REPO_ROOT, "verl_distillation")
if _VERL_DIST_PATH not in sys.path:
    sys.path.insert(0, _VERL_DIST_PATH)


def _inject_recsys_runtime_env(config):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    if ray.is_initialized():
        return

    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env_vars = runtime_env_kwargs.get("env_vars", {})

    recsys_env = {
        "VERL_RECSYS_MODEL_ENGINE": config.recsys.model_engine.get("mode", "new"),
        "VERL_RECSYS_TRANSFER_QUEUE": "1" if config.transfer_queue.enable else "0",
        "VERL_RECSYS_ASYNC_TRAINER": "1" if config.recsys.training.get("enable_async", False) else "0",
        "VERL_RECSYS_ROUTER_REPLAY": "1" if config.recsys.rollout.get("enable_router_replay", False) else "0",
    }
    runtime_env_vars.update(recsys_env)
    runtime_env_kwargs["env_vars"] = runtime_env_vars

    default_runtime_env = get_ppo_ray_runtime_env()
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    config.ray_kwargs.ray_init = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})


def _apply_teacher_paths(config) -> None:
    """Map recsys teacher list into current VeRL reference model path."""
    teacher_paths = config.recsys.distillation.get("teacher_model_paths", [])
    if not teacher_paths:
        return

    # Current VeRL runtime expects one ref model path. Keep the first path and
    # leave multi-teacher orchestration for future trainer-level extension.
    if len(teacher_paths) > 1:
        print("Using first teacher in teacher_model_paths for current runtime compatibility.")
    teacher = teacher_paths[0]
    if teacher and config.actor_rollout_ref.ref.model.get("path", None) in (None, ""):
        config.actor_rollout_ref.ref.model.path = teacher


def run_recsys(config) -> None:
    from verl.trainer.main_ppo import run_ppo

    from verl_recsys.trainer.recsys_task_runner import RecsysTaskRunner

    apply_acceleration_config(config)
    _inject_recsys_runtime_env(config)
    _apply_teacher_paths(config)
    task_runner_class = ray.remote(num_cpus=1)(RecsysTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


@hydra.main(config_path="config", config_name="recsys_trainer", version_base=None)
def main(config):
    run_recsys(config)


if __name__ == "__main__":
    main()
