from typing import List, Optional, Tuple

import torch
import torch.distributed

from sarathi.config import SchedulerType, SystemConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.model_executor import get_model, set_random_seed
from sarathi.model_executor.attention.attention_backend_registry import (
    AttentionBackendRegistry,
)
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.layers.sampler import Sampler
from sarathi.model_executor.utils import pad_to_alignment
from sarathi.utils import get_gpu_memory

logger = init_logger(__name__)


class ModelRunner:

    def __init__(
        self,
        config: SystemConfig,
        device: torch.device,
        rank: int,
    ):
        self.config = config
        self.device = device
        self.rank = rank

        self.attention_backend_wrapper: BaseAttentionWrapper = (
            AttentionBackendRegistry.get(
                config.worker_config.attention_backend,
                config.model_config,
                config.parallel_config,
                config.cache_config,
                device,
            )
        )

        self.model = get_model(self.config.model_config)

        self.sampler: Optional[Sampler] = None
        if self.model.lm_head:
            self.sampler = Sampler(
                self.model.lm_head.weight, self.model.config.vocab_size
            )

        self._prepare_inputs_e2e_timer = CpuTimer(
            CpuOperationMetrics.PREPARE_INPUTS_E2E, rank=self.rank
        )
        self._sampler_e2e_timer = CpuTimer(
            CpuOperationMetrics.SAMPLER_E2E, rank=self.rank
        )
        self._model_execution_e2e_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E, rank=self.rank
        )

    def init_kv_cache(self, num_gpu_blocks: int):
        self.attention_backend_wrapper.init_gpu_cache(num_gpu_blocks)

    def _prepare_inputs(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        # need to know prompt chunk sizes for each prompt sequence for sampler
        current_prompt_chunk_lens: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_tokens = (
                seq_metadata.seq.get_next_prompt_chunk_token_ids(prompt_chunk_len)
            )
            current_prompt_chunk_len = len(current_prompt_chunk_tokens)
            current_prompt_chunk_lens.append(current_prompt_chunk_len)
            processed_prompt_len = (
                seq_metadata.seq.get_num_prompt_tokens_stage_processed()
            )

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            input_tokens.extend(current_prompt_chunk_tokens)
            input_positions.extend(range(processed_prompt_len, current_total_len))

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            generation_token = seq_metadata.seq.get_last_token_id()
            input_tokens.append(generation_token)

            context_len = seq_metadata.seq.get_len()
            position = context_len - 1
            input_positions.append(position)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )

        return tokens_tensor, positions_tensor

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        torch.cuda.set_device(self.device)

        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = (
            self.config.scheduler_config.get_max_num_batched_tokens(
                self.config.model_config.max_model_len
            )
        )
        max_num_seqs = self.config.scheduler_config.max_num_seqs

        seq_metadata_list: List[SequenceMetadata] = []

        if (
            self.config.scheduler_config.get_type() == SchedulerType.SARATHI
            or self.config.scheduler_config.get_type() == SchedulerType.SIMPLE_CHUNKING
        ):
            # Profile memory usage with a single `chunk_size` chunk
            # which is the last chunk in the longest supported sequence.
            chunk_size = self.config.scheduler_config.chunk_size
            seq_len = self.config.model_config.max_model_len
            chunk_size = min(chunk_size, seq_len)
            seq = Sequence(
                seq_id=0,
                prompt=None,
                prompt_token_ids=[0] * seq_len,
                block_size=block_size,
                eos_token_id=1,
                arrival_time=None,
                sampling_params=sampling_params,
            )
            seq_metadata = SequenceMetadata(
                seq=seq,
                block_table=None,
                prompt_chunk_len=chunk_size,
            )
            seq_metadata_list.append(seq_metadata)
        else:
            # Profile memory usage with max_num_sequences sequences and the total
            # number of tokens equal to max_num_batched_tokens.
            for seq_id in range(max_num_seqs):
                seq_len = max_num_batched_tokens // max_num_seqs + (
                    seq_id < max_num_batched_tokens % max_num_seqs
                )

                seq = Sequence(
                    seq_id=str(seq_id),
                    prompt=None,
                    prompt_token_ids=[0] * seq_len,
                    block_size=block_size,
                    eos_token_id=1,
                    arrival_time=None,
                    sampling_params=sampling_params,
                )
                seq_metadata = SequenceMetadata(
                    seq=seq,
                    block_table=None,
                    prompt_chunk_len=seq_len,
                )
                seq_metadata_list.append(seq_metadata)

        input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)
        self.attention_backend_wrapper.begin_forward(seq_metadata_list)

        # Execute the model.
        num_layers = self.config.model_config.get_num_layers(
            self.config.parallel_config
        )
        self.model(
            hidden_states=input_tokens,
            positions=input_positions,
            attention_backend_wrapper=self.attention_backend_wrapper,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = self.attention_backend_wrapper.get_cache_block_size()
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory)
            // cache_block_size
        )
        num_gpu_blocks = max(num_gpu_blocks, 0)
        torch.cuda.empty_cache()

        self.attention_backend_wrapper.end_forward()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.config.model_config.seed)
        return num_gpu_blocks

    def run(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> torch.Tensor:
        # Prepare input tensors.
        with self._prepare_inputs_e2e_timer:
            input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)

        self.attention_backend_wrapper.begin_forward(seq_metadata_list)

        with self._model_execution_e2e_timer:
            # Execute the model.
            try:

                output = self.model(
                    hidden_states=input_tokens,
                    positions=input_positions,
                    attention_backend_wrapper=self.attention_backend_wrapper,
                )
            except RuntimeError as e:
                logger.error(
                    f"RuntimeError: {e} for seq_metadata_list: {seq_metadata_list}"
                )
                raise e

        with self._sampler_e2e_timer:
            if self.sampler is not None:
                output = self.sampler(output, seq_metadata_list)

        self.attention_backend_wrapper.end_forward()

        return output
