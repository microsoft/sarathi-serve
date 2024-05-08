import torch

from flash_attn import flash_attn_with_kvcache
from typing import List, Optional, Tuple

from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper


class FlashAttentionWrapper(BaseAttentionWrapper):
    _inst = None

    def init(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        device: torch.device,
    ):
        super().init(num_q_heads, num_kv_heads, head_dim, block_size, device)

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.prefill_query_lens: List[int] = None
        self.prefill_cache_lens: List[torch.Tensor] = None
        self.decode_cache_len: torch.Tensor = None
        self.prefill_block_tables: List[torch.Tensor] = None
        self.decode_block_table: torch.Tensor = None

    def get_cache_block(self, num_blocks: int,
                        **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        k_cache = torch.randn(
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )
        v_cache = torch.randn(
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

        return k_cache, v_cache

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        prefill_query_lens: List[int] = []
        prefill_cache_lens: List[List[int]] = []
        decode_cache_len: List[int] = []
        prefill_block_tables: List[List[int]] = []
        decode_block_table: List[List[int]] = []

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue
            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len)
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed(
            )

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            prefill_query_lens.append(current_prompt_chunk_len)
            prefill_cache_lens.append([processed_prompt_len])

            num_blocks_in_use = (current_total_len + self.block_size -
                                 1) // self.block_size
            prefill_block_tables.append(
                seq_metadata.block_table[:num_blocks_in_use])

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            context_len = seq_metadata.seq.get_len()
            decode_cache_len.append(context_len - 1)
            # Compute the kv page indices for the prompt tokens.
            decode_block_table.append(seq_metadata.block_table)

        self.prefill_query_lens = prefill_query_lens
        self.prefill_cache_lens = [
            torch.tensor(cache_lens, dtype=torch.int32, device=self.device)
            for cache_lens in prefill_cache_lens
        ]
        self.prefill_block_tables = [
            torch.tensor(block_table, dtype=torch.int32,
                         device=self.device).reshape(1, -1)
            for block_table in prefill_block_tables
        ]

        if decode_cache_len == []:
            # no decode block table
            return

        self.decode_cache_len = torch.tensor(decode_cache_len,
                                             dtype=torch.int32,
                                             device=self.device)

        max_decode_blocks = max(
            len(seq_block) for seq_block in decode_block_table)
        decode_block_table_padded = [
            seq_block + [0] * (max_decode_blocks - len(seq_block))
            for seq_block in decode_block_table
        ]
        self.decode_block_table = torch.tensor(decode_block_table_padded,
                                               dtype=torch.int32,
                                               device=self.device)

    def end_forward(self):
        self.is_metadata_initialized = False

        self.prefill_query_lens = None
        self.prefill_cache_lens = None
        self.prefill_block_tables = None
        self.decode_cache_len = None
        self.decode_block_table = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        token_offset = 0

        output = torch.empty_like(query, device=self.device)

        # first process the prefill attention
        for prefill_cache_len, prefill_block_table, query_len in zip(
                self.prefill_cache_lens, self.prefill_block_tables,
                self.prefill_query_lens):
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset:token_offset +
                                  query_len].reshape(1, -1, self.num_q_heads,
                                                     self.head_dim)
                seq_key = key[token_offset:token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim)
                seq_value = value[token_offset:token_offset +
                                  query_len].reshape(1, -1, self.num_kv_heads,
                                                     self.head_dim)

            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = flash_attn_with_kvcache(
                    seq_query,
                    kv_cache[0],  # k_cache,
                    kv_cache[1],  # v_cache,
                    seq_key,
                    seq_value,
                    cache_seqlens=prefill_cache_len,
                    block_table=prefill_block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE,
                                layer_id):
                output[token_offset:token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim))

            token_offset += query_len

        if self.decode_cache_len is None:
            return output

        decode_batch_size = self.decode_cache_len.size(0)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            decode_query = query[token_offset:token_offset +
                                 decode_batch_size].reshape(
                                     -1, 1, self.num_q_heads, self.head_dim)
            decode_key = key[token_offset:token_offset +
                             decode_batch_size].reshape(
                                 -1, 1, self.num_kv_heads, self.head_dim)
            decode_value = value[token_offset:token_offset +
                                 decode_batch_size].reshape(
                                     -1, 1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            decode_output = flash_attn_with_kvcache(
                decode_query,
                kv_cache[0],  # k_cache,
                kv_cache[1],  # v_cache,
                decode_key,
                decode_value,
                cache_seqlens=self.decode_cache_len,
                block_table=self.decode_block_table,
                softmax_scale=softmax_scale,
                causal=True,
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            # flatten the seq_output and copy it to the output tensor
            output[token_offset:token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim))

        return output
