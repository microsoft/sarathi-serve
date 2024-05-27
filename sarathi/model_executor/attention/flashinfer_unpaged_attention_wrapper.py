from typing import List, Optional

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    append_paged_kv_cache,
    single_prefill_with_kv_cache,
)

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.attention.kv_buffer import KVBuffer


class FlashinferUnpagedAttentionWrapper(BaseAttentionWrapper):
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
        self._wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        self.kv_buffers: List[KVBuffer] = []
        num_layers = model_config.get_num_layers(parallel_config)
        for _ in range(num_layers):
            self.kv_buffers.append(
                KVBuffer(
                    model_config.get_max_model_len(),
                    self.num_kv_heads,
                    self.head_dim,
                    device,
                    self.dtype,
                )
            )

        self.is_metadata_initialized: bool = False
        self.is_profiling_iteration: bool = False
        self.qo_indptr_tensor: torch.Tensor = None
        self.kv_page_indices_tensor: torch.Tensor = None
        self.kv_page_indptr_tensor: torch.Tensor = None
        self.kv_last_page_len_tensor: torch.Tensor = None
        self.layer_index: int = 0
        self.decode_batch_size: int = 0
        self.prompt_seq_ids: List[int] = []
        self.prompt_chunk_lens: List[int] = []
        self.processed_prompt_lens: List[int] = []
        self.total_prompt_lens: List[int] = []

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.empty(
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
        # We perform only decode using the paged attention api with te following layout:
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        kv_page_indices: List[int] = []
        decode_kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        kv_last_page_len: List[int] = []
        decode_kv_last_page_len: List[int] = []
        # Since the kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        kv_page_indptr: List[int] = [0]
        decode_kv_page_indptr: List[int] = [0]
        # we also create a qo_indptr tensor to capture the start of each sequence in the
        # ragged tensor which is used for the kv cache append api.
        # qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        qo_indptr: List[int] = [0]

        prompt_seq_ids: List[int] = []
        prompt_chunk_lens: List[int] = []
        processed_prompt_lens: List[int] = []
        total_prompt_lens: List[int] = []

        decode_batch_size: int = 0

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        for seq_metadata in seq_metadata_list:
            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len

            prompt_seq_ids.append(seq_metadata.seq.seq_id)
            prompt_chunk_lens.append(prompt_chunk_len)
            processed_prompt_lens.append(processed_prompt_len)
            total_prompt_lens.append(seq_metadata.seq.get_prompt_len())
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
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            if seq_metadata.is_prompt:
                continue

            decode_batch_size += 1

            context_len = seq_metadata.seq.get_len()
            # indptr for the prompt tokens in q/o tensor
            qo_indptr.append(qo_indptr[-1] + 1)
            # Compute the kv page indices for the prompt tokens.
            kv_page_indices.extend(seq_metadata.block_table)
            decode_kv_page_indices.extend(seq_metadata.block_table)
            kv_page_indptr.append(kv_page_indptr[-1] + len(seq_metadata.block_table))
            decode_kv_page_indptr.append(
                decode_kv_page_indptr[-1] + len(seq_metadata.block_table)
            )
            kv_last_page_len.append(context_len % self.block_size or self.block_size)
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

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
        decode_kv_page_indices = torch.tensor(
            decode_kv_page_indices, dtype=torch.int32, device=self.device
        )
        decode_kv_page_indptr = torch.tensor(
            decode_kv_page_indptr, dtype=torch.int32, device=self.device
        )
        decode_kv_last_page_len = torch.tensor(
            decode_kv_last_page_len, dtype=torch.int32, device=self.device
        )

        self.prompt_seq_ids = prompt_seq_ids
        self.prompt_chunk_lens = prompt_chunk_lens
        self.processed_prompt_lens = processed_prompt_lens
        self.total_prompt_lens = total_prompt_lens
        self.layer_index = 0
        self.decode_batch_size = decode_batch_size

        self._wrapper.begin_forward(
            decode_kv_page_indptr,
            decode_kv_page_indices,
            decode_kv_last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
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

        output = torch.empty_like(query).view(-1, self.num_q_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        qo_offset: int = 0
        for i, seq_id in enumerate(self.prompt_seq_ids):
            kv_buffer = self.kv_buffers[self.layer_index]

            prompt_chunk_len = self.prompt_chunk_lens[i]
            processed_prompt_len = self.processed_prompt_lens[i]
            total_prompt_len = self.total_prompt_lens[i]

            q = query[qo_offset : qo_offset + prompt_chunk_len]
            k = key[qo_offset : qo_offset + prompt_chunk_len]
            v = value[qo_offset : qo_offset + prompt_chunk_len]

            if prompt_chunk_len == total_prompt_len:
                # if all the tokens are processed at once, we can skip the kv buffer management
                with self.get_timer(OperationMetrics.ATTN, layer_id):
                    output[qo_offset : qo_offset + prompt_chunk_len] = (
                        single_prefill_with_kv_cache(
                            q,
                            k,
                            v,
                            causal=True,
                            pos_encoding_mode="NONE",
                            sm_scale=softmax_scale,
                        )
                    )
            else:
                if seq_id not in kv_buffer.buffer_indices:
                    kv_buffer.add_request(seq_id)

                kv_buffer.append(seq_id, k, v)
                k_, v_ = kv_buffer.get_kv_tensors(seq_id)
                with self.get_timer(OperationMetrics.ATTN, layer_id):
                    output[qo_offset : qo_offset + prompt_chunk_len] = (
                        single_prefill_with_kv_cache(
                            q,
                            k_,
                            v_,
                            causal=True,
                            pos_encoding_mode="NONE",
                            sm_scale=softmax_scale,
                        )
                    )

                if total_prompt_len == processed_prompt_len + prompt_chunk_len:
                    kv_buffer.free_request(seq_id)

            qo_offset += prompt_chunk_len

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

        if self.decode_batch_size > 0:
            with self.get_timer(OperationMetrics.ATTN, layer_id):
                output[qo_offset : qo_offset + self.decode_batch_size] = (
                    self._wrapper.forward(
                        query[qo_offset : qo_offset + self.decode_batch_size],
                        kv_cache,
                        pos_encoding_mode="NONE",
                        sm_scale=softmax_scale,
                    )
                )
                qo_offset += self.decode_batch_size

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        self.layer_index += 1
        assert self.layer_index <= len(self.kv_buffers)

        return output
