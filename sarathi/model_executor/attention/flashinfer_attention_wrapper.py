from typing import List, Optional

import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.utils import round_up_to_multiple


class FlashinferAttentionWrapper(BaseAttentionWrapper):
    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        super().init(model_config, parallel_config, block_size, device)

        workspace_buffer = torch.empty(
            16 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.qo_indptr_tensor = None
        self.kv_page_indices_tensor = None
        self.kv_page_indptr_tensor = None
        self.kv_last_page_len_tensor = None

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        # The indptr tensor captures the location query tokens in the input tensor.
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # Flashinfer calls this layout as a raggedtensor. The indptr tensor captures the start of each
        # sequence in the ragged tensor. The length of the indptr tensor is the number of sequences + 1.
        # We perform both prefill and decode attention in a single call to batched prefill kernel.
        # qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        qo_indptr: List[int] = [0]
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        kv_last_page_len: List[int] = []
        # Since the kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        kv_page_indptr: List[int] = [0]

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            current_total_len = processed_prompt_len + prompt_chunk_len

            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            # indptr for the prompt tokens in q/o tensor
            qo_indptr.append(qo_indptr[-1] + prompt_chunk_len)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1] + num_blocks_in_use)
            kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            context_len = seq_metadata.seq.get_len()
            # indptr for the prompt tokens in q/o tensor
            qo_indptr.append(qo_indptr[-1] + 1)
            # Compute the kv page indices for the prompt tokens.
            kv_page_indices.extend(seq_metadata.block_table)
            kv_page_indptr.append(kv_page_indptr[-1] + len(seq_metadata.block_table))
            kv_last_page_len.append(context_len % self.block_size or self.block_size)

        # Convert to tensors.
        self.qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
        self.kv_page_indices = torch.tensor(
            kv_page_indices, dtype=torch.int32, device=self.device
        )
        self.kv_page_indptr = torch.tensor(
            kv_page_indptr, dtype=torch.int32, device=self.device
        )
        self.kv_last_page_len = torch.tensor(
            kv_last_page_len, dtype=torch.int32, device=self.device
        )

        self._wrapper.begin_forward(
            self.qo_indptr,
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
        )

    def end_forward(self):
        self._wrapper.end_forward()
        self.is_metadata_initialized = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            append_paged_kv_cache(
                key,
                value,
                self.qo_indptr,
                kv_cache,
                self.kv_page_indices,
                self.kv_page_indptr,
                self.kv_last_page_len,
                kv_layout="NHD",
            )

        with self.get_timer(OperationMetrics.ATTN, layer_id):
            output = self._wrapper.forward(
                query,
                kv_cache,
                pos_encoding_mode="NONE",
                sm_scale=softmax_scale,
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output
