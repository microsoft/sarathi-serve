from abc import abstractmethod, ABC
from typing import List, Optional, Union, Tuple

import torch

from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer


class BaseAttentionWrapper(ABC):
    _inst = None

    def init(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        device: torch.device,
    ):
        self.device = device
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self._timers = {}

    """
    For a given model, all layers same the same AttentionWrapper instance.
    However, we cannot have a single timer for all layers because the same timer cannot be turned on/off dynamically.
    So, we have timers for each layer separately.
    """

    def get_timer(self,
                  operation: OperationMetrics,
                  layer_id: Optional[int] = None):
        if self._timers.get((operation, layer_id)) is None:
            self._timers[(operation,
                          layer_id)] = CudaTimer(operation, layer_id)
        return self._timers.get((operation, layer_id))

    @abstractmethod
    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        pass

    @classmethod
    def get_instance(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    @abstractmethod
    def end_forward(self):
        pass

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        pass
