from typing import Dict, Tuple

import torch


class KVBuffer:
    """
    A class which is the key-value buffer for the model.
    A loose analogy is that this buffer is like an L1 cache and the conventional
    KV-cache is like an L2 cache
    """

    def __init__(
        self,
        max_seq_len: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.device = device
        self.dtype = dtype
        self.v_buffer = torch.zeros(
            (2 * max_seq_len, self.num_kv_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.k_buffer = torch.zeros(
            (2 * max_seq_len, self.num_kv_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.buffer_indices: Dict[int, int] = {}
        self.buffer_active_lens: Dict[int, int] = {}
        self.buffer_offset: int = 0

    def add_request(self, seq_id: int) -> None:
        assert seq_id not in self.buffer_indices
        assert seq_id not in self.buffer_active_lens
        # we only support two requests at a time -- no more is required
        assert len(self.buffer_indices) < 2
        if len(self.buffer_indices) == 0:
            self.buffer_indices[seq_id] = 0
        else:
            self.buffer_indices[seq_id] = self.max_seq_len
        self.buffer_active_lens[seq_id] = 0

    def free_request(self, seq_id: int) -> None:
        assert seq_id in self.buffer_indices
        assert seq_id in self.buffer_active_lens
        del self.buffer_indices[seq_id]
        del self.buffer_active_lens[seq_id]

    def get_kv_tensors(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert seq_id in self.buffer_indices
        assert seq_id in self.buffer_active_lens
        start_offset = self.buffer_indices[seq_id]
        end_offset = start_offset + self.buffer_active_lens[seq_id]
        return (
            self.k_buffer[start_offset:end_offset],
            self.v_buffer[start_offset:end_offset],
        )

    def append(self, seq_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        assert key.shape == value.shape
        active_length = self.buffer_active_lens[seq_id]
        assert active_length + key.shape[0] <= self.max_seq_len
        start_offset = self.buffer_indices[seq_id] + active_length
        end_offset = start_offset + key.shape[0]

        self.k_buffer[start_offset:end_offset].copy_(key)
        self.v_buffer[start_offset:end_offset].copy_(value)
        self.buffer_active_lens[seq_id] += key.shape[0]

    def reset(self) -> None:
        self.buffer_indices = {}
        self.buffer_active_lens = {}

    def has_seq_id(self, seq_id: int) -> bool:
        return seq_id in self.buffer_indices
