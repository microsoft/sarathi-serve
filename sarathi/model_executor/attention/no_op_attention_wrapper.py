from typing import List, Optional, Tuple

import torch

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper


class NoOpAttentionWrapper(BaseAttentionWrapper):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        self.device = device

    def init_gpu_cache(self, num_gpu_blocks: int) -> None:
        pass

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        pass

    def get_cache_block_size(self) -> int:
        pass

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        pass

    def end_forward(self):
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache_idx: int,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        return torch.empty_like(query, device=self.device)
