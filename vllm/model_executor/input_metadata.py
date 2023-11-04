from typing import Dict, List, Tuple

import torch
from xformers.ops import AttentionBias

from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        processed_prompt_lens: Lengths of processed parts of prompts.
        current_prompt_chunk_lens: Lengths of current chunks (to be processed in this iteration) of prompts.
        prefix_plus_current_prompt_tokens_slot_mapping: The address of KV cache of each token in the processed part and the current chunk of prompt is stored.
        current_tokens_slot_mapping: The address to write the new KV to of each token in the current chunk of prompts/decode.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        processed_prompt_lens: List[int],
        current_prompt_chunk_lens: List[int],
        prefix_plus_current_prompt_tokens_slot_mapping: torch.Tensor,
        current_tokens_slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        block_tables: torch.Tensor,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.processed_prompt_lens = processed_prompt_lens
        self.current_prompt_chunk_lens = current_prompt_chunk_lens
        self.prefix_plus_current_prompt_tokens_slot_mapping = prefix_plus_current_prompt_tokens_slot_mapping
        self.current_tokens_slot_mapping = current_tokens_slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables

        assert ([x > 0 for x in current_prompt_chunk_lens].count(False) == 0)
        self.num_prompts = len([x for x in current_prompt_chunk_lens if x > 0])
        self.num_processed_prompt_tokens = sum(processed_prompt_lens)
        self.num_current_prompt_tokens = sum(current_prompt_chunk_lens)
        self.num_generation_tokens = context_lens.shape[0]
        # number of total tokens in the input sequence (prompt + generated so far)
        self.num_valid_tokens = current_tokens_slot_mapping.shape[0]
        if block_tables.numel() > 0:
            self.max_num_blocks_per_seq = block_tables.shape[1]
        else:
            self.max_num_blocks_per_seq = 0
        # assert block_tables.shape[0] == self.num_generation_tokens
        # assert context_lens.shape[0] == self.num_generation_tokens

        # Set during the execution of the first attention op.
        self.attn_bias: List[AttentionBias] = []

    def __repr__(self) -> str:
        # Print only useful metadata.
        return (
            f"InputMetadata("
            f"num_valid_tokens={self.num_valid_tokens}, "
            f"num_prompts={self.num_prompts}, "
            f"processed_prompt_lens={self.processed_prompt_lens}, "
            f"current_prompt_chunk_lens={self.current_prompt_chunk_lens}, "
            f"num_current_prompt_tokens={self.num_current_prompt_tokens}, "
            f"num_generation_tokens={self.num_generation_tokens}, "
            f"context_lens={self.context_lens}, "
            f"max_context_len={self.max_context_len}), "
            f"max_num_blocks_per_seq={self.max_num_blocks_per_seq}, "
            f"block_tables={self.block_tables}), "
            f"prefix_plus_current_prompt_tokens_slot_mapping={self.prefix_plus_current_prompt_tokens_slot_mapping}), "
            f"current_tokens_slot_mapping={self.current_tokens_slot_mapping}")
