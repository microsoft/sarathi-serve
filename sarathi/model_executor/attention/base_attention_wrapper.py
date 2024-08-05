from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer


class BaseAttentionWrapper(ABC):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        device: torch.device,
    ):
        self.device = device
        self.num_q_heads = model_config.get_num_q_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_dim = model_config.get_head_size()
        self.dtype = model_config.dtype
        self.block_size = cache_config.block_size
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_gpu_blocks: int = cache_config.num_gpu_blocks
        self.gpu_cache: Optional[List[torch.Tensor]] = None
        self.timers: Optional[Dict[Tuple[OperationMetrics, int], CudaTimer]] = {}

    """
    For a given model, all layers same the same AttentionWrapper instance.
    However, we cannot have a single timer for all layers because the same timer cannot be turned on/off dynamically.
    So, we have timers for each layer separately.
    """

    def get_timer(self, operation: OperationMetrics, layer_id: Optional[int] = None):
        if self.timers.get((operation, layer_id)) is None:
            self.timers[(operation, layer_id)] = CudaTimer(operation, layer_id)
        return self.timers.get((operation, layer_id))

    def get_cache_block_size(self) -> int:
        key_cache_block = self.block_size * self.num_kv_heads * self.head_dim
        value_cache_block = key_cache_block
        total = self.num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(self.dtype)
        return dtype_size * total

    @abstractmethod
    def init_gpu_cache(self, num_gpu_blocks: int) -> None:
        pass

    @abstractmethod
    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        pass

    @abstractmethod
    def end_forward(self):
        pass

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache_idx: int,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        pass


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
