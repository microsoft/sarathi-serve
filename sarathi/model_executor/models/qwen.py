# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens.
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.layers.activation import SiluAndMul
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from sarathi.transformers_utils.configs.qwen import QWenConfig


class QWenMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self._mlp_activation_timer = CudaTimer(OperationMetrics.MLP_ACTIVATION)

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads

        # pylint: disable=invalid-name
        self.c_attn = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
        )
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cache_idx: int,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = attention_backend_wrapper.forward(
            q,
            k,
            v,
            layer_cache_idx,
            self.scaling,
        )

        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):

    def __init__(self, config: QWenConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cache_idx: int,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            layer_cache_idx=layer_cache_idx,
            attention_backend_wrapper=attention_backend_wrapper,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QWenModel(nn.Module):

    def __init__(self, config: QWenConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = None

        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.wte = VocabParallelEmbedding(
                vocab_size, config.hidden_size, perform_initialization=False
            )
        self.h = nn.ModuleList(
            [
                QWenBlock(config)
                for _ in range(
                    config.num_hidden_layers // get_pipeline_model_parallel_world_size()
                )
            ]
        )
        self.ln_f = None
        if is_pipeline_last_stage():
            self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if self.wte:
            hidden_states = self.wte(hidden_states)

        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                positions, hidden_states, i, attention_backend_wrapper
            )
        if self.ln_f:
            hidden_states = self.ln_f(hidden_states)

        return hidden_states


class QWenLMHeadModel(nn.Module):

    def __init__(self, config: QWenConfig):
        super().__init__()
        self.config = config
        self.transformer = QWenModel(config)

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            vocab_size = ((config.vocab_size + 63) // 64) * 64

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)
        hidden_states = self.transformer(
            hidden_states, positions, attention_backend_wrapper
        )

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    _column_parallel_weights = []
    _row_parallel_weights = ["c_proj.weight"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        tp_world_size = get_tensor_model_parallel_world_size()
        pp_world_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()
        state_dict = self.state_dict()

        assert self.config.num_hidden_layers % pp_world_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_world_size

        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            if pp_rank != 0 and "wte" in name:
                continue

            if pp_rank != pp_world_size - 1 and ("lm_head" in name or "ln_f" in name):
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            if "model.h." in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(str(layer_id), str(new_layer_id))

            if "c_attn" in name:
                total_num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // total_num_heads
                num_heads = total_num_heads // tp_world_size
                head_start = tp_rank * num_heads
                head_end = (tp_rank + 1) * num_heads

                if "weight" in name:
                    loaded_weight = loaded_weight.view(
                        3, total_num_heads, head_size, hidden_size
                    )
                    loaded_weight = loaded_weight[:, head_start:head_end, :, :]
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                elif "bias" in name:
                    loaded_weight = loaded_weight.view(3, total_num_heads, head_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :]
                    loaded_weight = loaded_weight.reshape(-1)

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["w2", "w1"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]

            if "wte" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                self._column_parallel_weights,
                self._row_parallel_weights,
                tp_rank,
            )
