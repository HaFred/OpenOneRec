# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable

import torch
import torch.distributed

from verl import DataProto
from verl.utils.device import get_nccl_backend, get_torch_device

from ..base import BaseEngine, EngineRegistry
from .transformer_impl import initialize_megatron_model_parallel, summarize_parallelism_state, validate_parallelism


@EngineRegistry.register("megatron")
class MegatronEngine(BaseEngine):
    def __init__(self, config):
        self.config = config
        self.mode = None
        self.module = None
        self.optimizer = None
        self.lr_scheduler = None

    def init_model(self):
        if not torch.distributed.is_initialized():
            rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
            torch.distributed.init_process_group(backend=get_nccl_backend())
            get_torch_device().set_device(rank)

        megatron_cfg = getattr(self.config, "megatron", None)
        if megatron_cfg is None:
            raise ValueError("MegatronEngine requires `config.megatron`.")
        validate_parallelism(megatron_cfg, world_size=torch.distributed.get_world_size(), role_name="engine_megatron")
        initialize_megatron_model_parallel(megatron_cfg)
        state = summarize_parallelism_state()
        print(
            f"[megatron_init][engine] TP={state['tp_size']} PP={state['pp_size']} CP={state['cp_size']} "
            f"DP={state['dp_size']} ranks(tp/pp/cp/dp)=({state['tp_rank']}/{state['pp_rank']}/"
            f"{state['cp_rank']}/{state['dp_rank']})"
        )

    def train_mode(self):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        class _TrainCtx:
            def __init__(self, engine):
                self.engine = engine

            def __enter__(self):
                self.engine.mode = "train"

            def __exit__(self, exc_type, exc_value, traceback):
                self.engine.mode = None

        return _TrainCtx(self)

    def eval_mode(self):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        class _EvalCtx:
            def __init__(self, engine):
                self.engine = engine

            def __enter__(self):
                self.engine.mode = "eval"

            def __exit__(self, exc_type, exc_value, traceback):
                self.engine.mode = None

        return _EvalCtx(self)

    def infer_batch(
        self,
        data: DataProto,
        post_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        Perform inference on a mini batch of data.

        Args:
            data: The input data for inference, typically containing tensors and metadata.
            post_fn: A post-processing function that takes a micro-batch and predictions as input,
                     and returns a tuple containing processed predictions and a dictionary of outputs.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the predictions for the entire batch.
        """
        raise NotImplementedError(
            "MegatronEngine.infer_batch is not wired in this fork. "
            "Use legacy `verl.workers.megatron_workers` for actor/critic/ref rollout execution."
        )

    def train_batch(
        self,
        data: DataProto,
        loss_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        Perform a training step on a mini-batch of data.

        Args:
            data (DataProto): The input data for training, typically containing tensors and metadata.
            loss_fn (Callable): A function that computes the loss and metrics given a micro-batch and predictions.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the mini-batch.
        """
        raise NotImplementedError(
            "MegatronEngine.train_batch is not wired in this fork. "
            "Use legacy `verl.workers.megatron_workers` for actor/critic/ref rollout execution."
        )

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        if self.optimizer is None:
            return torch.tensor(0.0)
        self.optimizer.step()
        return torch.tensor(0.0)

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        if self.lr_scheduler is None:
            return [0.0]
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()

    def shard_data(self, data):
        """
        Shard or partition data for distributed training or parallel execution.

        Args:
            data: Data structure to be sharded across devices/workers.

        Returns:
            Sharded data in the same format as input.
        """
        return data

    def unshard_data(self, data):
        """
        Reconstruct or gather sharded data back to a unified format.

        Args:
            data: Sharded data structure to reconstruct.

        Returns:
            Unsharded, combined data.
        """
        return data

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        if device not in ("cuda", "cpu"):
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        raise NotImplementedError("Checkpointing for MegatronEngine shim is unsupported in this fork.")

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        raise NotImplementedError("Checkpointing for MegatronEngine shim is unsupported in this fork.")
