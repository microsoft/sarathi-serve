# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The Sarathi team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens.
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import LlamaConfig

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
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_id
        )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_id = layer_id

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
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
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = attention_backend_wrapper.forward(
            q,
            k,
            v,
            layer_cache_idx,
            self.scaling,
            self.layer_id,
        )
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            layer_id=layer_id,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
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
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            layer_cache_idx=layer_cache_idx,
            attention_backend_wrapper=attention_backend_wrapper,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                config.hidden_size,
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )

        num_layers = (
            config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_id=layer_id + layer_offset)
                for layer_id in range(num_layers)
            ]
        )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions, hidden_states, i, attention_backend_wrapper
            )

        if self.norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
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

        hidden_states = self.model(hidden_states, positions, attention_backend_wrapper)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        weight_suffixes = ["weight"]

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        pp_model_parallel_rank = get_pipeline_model_parallel_rank()

        assert self.config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_size

        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1

        q_proj_shard_size = self.config.hidden_size // tp_size
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * self.config.num_key_value_heads
            // tp_size
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                _load_lm_head_weights_if_llama3_2(loaded_weight, model_name_or_path, pp_model_parallel_rank, pp_size,
                                                  state_dict, tensor_model_parallel_rank)
                continue

            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(str(layer_id), str(new_layer_id))

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size
                    * tensor_model_parallel_rank : shard_size
                    * (tensor_model_parallel_rank + 1)
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

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                _load_lm_head_weights_if_llama3_2(loaded_weight, model_name_or_path, pp_model_parallel_rank, pp_size,
                                                  state_dict, tensor_model_parallel_rank)

                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )


def _load_lm_head_weights_if_llama3_2(loaded_weight, model_name_or_path, pp_model_parallel_rank, pp_size,
                                      state_dict, tensor_model_parallel_rank):
    if pp_model_parallel_rank == pp_size - 1 and "3.2" in model_name_or_path:
        # pp case weight tying embed tokens to lm_head
        param_lm_head = state_dict["lm_head.weight"]
        load_padded_tensor_parallel_vocab(
            param_lm_head, loaded_weight, tensor_model_parallel_rank
        )
