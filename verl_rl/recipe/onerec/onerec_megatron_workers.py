from __future__ import annotations

import logging

from verl.models.mcore import get_mcore_weight_converter
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.megatron_workers import ActorRolloutRefWorker
from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

from recipe.onerec.onerec_vllm_rollout import OneRecvLLMRollout

logger = logging.getLogger(__name__)


class OneRecMegatronActorRolloutRefWorker(ActorRolloutRefWorker):
    """Megatron worker that plugs in OneRec two-stage vLLM rollout."""

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "two_stage":
            return super()._build_rollout(trust_remote_code)

        if self.config.rollout.mode != "sync":
            raise NotImplementedError("OneRec two_stage rollout currently supports only sync mode in Megatron")

        from torch.distributed.device_mesh import init_device_mesh

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )

        rollout_device_mesh = init_device_mesh(
            get_device_name(),
            mesh_shape=(dp, infer_tp),
            mesh_dim_names=["dp", "infer_tp"],
        )

        log_gpu_memory_usage("Before building OneRec two_stage rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        rollout = OneRecvLLMRollout(
            model_path=local_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            device_mesh=rollout_device_mesh,
            trust_remote_code=trust_remote_code,
        )
        log_gpu_memory_usage("After building OneRec two_stage rollout", logger=logger)

        weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        sharding_manager = MegatronVLLMShardingManager(
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            rollout_config=self.config.rollout,
            layer_name_mapping=layer_name_mapping,
            actor_module=self.actor.actor_module,
            weight_converter=weight_converter,
            device_mesh=rollout_device_mesh,
            offload_param=self._is_offload_param,
            bridge=self.bridge,
        )
        log_gpu_memory_usage("After building OneRec sharding manager", logger=logger)

        print(f"OneRec two_stage rollout and sharding manager init done: {sharding_manager}")
        return rollout, sharding_manager
