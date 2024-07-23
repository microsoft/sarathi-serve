"""CacheEngine class for managing the KV cache."""

from typing import List, Tuple, Union

import torch

from sarathi.config import ModelConfig, ParallelConfig, SystemConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper

logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """

    def __init__(
        self,
        config: SystemConfig,
    ) -> None:
        self.head_size = config.model_config.get_head_size()
        self.num_layers = config.model_config.get_num_layers(config.parallel_config)
        self.num_heads = config.model_config.get_num_kv_heads(config.parallel_config)
        self.dtype = config.model_config.dtype

        self.block_size = config.cache_config.block_size
        self.num_gpu_blocks = config.cache_config.num_gpu_blocks

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()

    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        gpu_cache: List[torch.Tensor] = []

        for _ in range(self.num_layers):
            gpu_blocks = get_attention_wrapper().get_cache_block(
                self.num_gpu_blocks, dtype=self.dtype, device="cuda"
            )
            gpu_cache.append(gpu_blocks)
        return gpu_cache

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
