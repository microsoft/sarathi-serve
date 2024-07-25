from typing import Optional

import torch

from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.parallel_utils.tensor_parallel.mappings import (
    get_tensor_model_parallel_world_size,
    reduce_from_tensor_model_parallel_region,
)


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        linear_metric_name: Optional[str] = None,
        communication_metric_name: Optional[str] = None,
        world_size: Optional[int] = None,
        layer_id: Optional[int] = None,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Keep input parameters
        self.params_dtype = params_dtype
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.world_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )

        self.create_weights()

        self._linear_timer = CudaTimer(linear_metric_name, layer_id=layer_id)
        self._communication_timer = CudaTimer(
            communication_metric_name, layer_id=layer_id
        )

    def create_weights(self):
        # Fused gate_up_proj (column parallel)
        self.w13_weight = torch.nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_size_per_partition,
                self.hidden_size,
                dtype=self.params_dtype,
            ),
            requires_grad=False,
        )

        # down_proj (row parallel)
        self.w2_weight = torch.nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size_per_partition,
                dtype=self.params_dtype,
            ),
            requires_grad=False,
        )

    def apply_weights(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        from sarathi.model_executor.layers.fused_moe.fused_moe import fused_moe

        with self._linear_timer:
            return fused_moe(
                x,
                self.w13_weight,
                self.w2_weight,
                router_logits,
                self.top_k,
                renormalize=self.renormalize,
                inplace=True,
                use_grouped_topk=self.use_grouped_topk,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
            )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # Matrix multiply.
        final_hidden_states = self.apply_weights(
            x=hidden_states, router_logits=router_logits
        )

        if self.reduce_results and self.world_size > 1:
            with self._communication_timer:
                final_hidden_states = reduce_from_tensor_model_parallel_region(
                    final_hidden_states
                )

        return final_hidden_states
