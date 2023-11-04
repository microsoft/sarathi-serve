"""A GPU worker class."""
import os
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed
import ray

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         MetricsConfig, BaseSchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory
from vllm.metrics.metrics_store import MetricsStore
from vllm.metrics.cpu_timer import CpuTimer
from vllm.metrics.constants import CpuOperationMetrics
from vllm.model_executor.layers.sampler import Sampler
from vllm.all_reduce_ops import init_nccl


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        metrics_config: MetricsConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.metrics_config = metrics_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

        self.metrics_store = MetricsStore(metrics_config)

        self._graph_cache = {}
        self._input_tensors_cache = {}
        self._output_tensors_cache = {}
        self._disable_cuda_graph = True

        self._prepare_inputs_e2e_timer = CpuTimer(
            CpuOperationMetrics.PREPARE_INPUTS_E2E)
        self._prepare_inputs_launch_timer = CpuTimer(
            CpuOperationMetrics.PREPARE_INPUTS_LAUNCH)
        self._post__prepare_inputs_barrier_timer = CpuTimer(
            CpuOperationMetrics.POST_PREPARE_INPUTS_BARRIER)
        self._sampler_launch_timer = CpuTimer(
            CpuOperationMetrics.SAMPLER_LAUNCH)
        self._sampler_e2e_timer = CpuTimer(CpuOperationMetrics.SAMPLER_E2E)
        self._model_execution_launch_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_LAUNCH)
        self._model_execution_e2e_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E)

    def init_model(self, rendezvous_id: int):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Required for properly capturing nccl ops
        os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        # gpu_ids = ray.get_gpu_ids()
        # print("visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
        # assert len(gpu_ids) == 1, gpu_ids
        # local_rank = gpu_ids[0]
        # print("Local rank: ", local_rank)
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      rendezvous_id,
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config)
        self.sampler = Sampler(self.model.config.vocab_size)

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                prompt_chunk_size=seq_len,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        print(
            f"Finished profiling num_gpu_blocks={num_gpu_blocks}, num_cpu_blocks={num_cpu_blocks}"
        )
        return num_gpu_blocks, num_cpu_blocks

    def init_graph(self, batch_size):
        seqs = []
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99,
                                         top_k=vocab_size - 1,
                                         ignore_eos=True,
                                         max_tokens=1)
        for group_id in range(batch_size):
            seq_data = SequenceData([0])
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=False,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables={group_id: [100]},
                prompt_chunk_size=0,
            )
            seqs.append(seq)

        tokens_tensor, positions_tensor, input_metadata = self._prepare_inputs(
            seqs)

        input_metadata.max_context_len = self.model_config.max_model_len

        self._input_tensors_cache[batch_size] = {
            "tokens_tensor": tokens_tensor,
            "positions_tensor": positions_tensor,
            "current_tokens_slot_mapping_tensor":
            input_metadata.current_tokens_slot_mapping,
            "context_lens_tensor": input_metadata.context_lens,
            "block_tables_tensor": input_metadata.block_tables,
        }

        torch.cuda.synchronize()
        print(f"Recording graph for batch size: {batch_size}")
        for _ in range(5):
            hidden_states = self.model(
                input_ids=self._input_tensors_cache[batch_size]
                ["tokens_tensor"],
                positions=self._input_tensors_cache[batch_size]
                ["positions_tensor"],
                kv_caches=self.gpu_cache,
                input_metadata=input_metadata,
                cache_events=None,
            )

        self._output_tensors_cache[batch_size] = hidden_states

        graph = torch.cuda.CUDAGraph()

        torch.cuda.synchronize()

        mempool = torch.cuda.graph_pool_handle()

        self.start_profiling("cpu")

        with torch.cuda.graph(graph, pool=mempool):
            hidden_states = self.model(
                input_ids=self._input_tensors_cache[batch_size]
                ["tokens_tensor"],
                positions=self._input_tensors_cache[batch_size]
                ["positions_tensor"],
                kv_caches=self.gpu_cache,
                input_metadata=input_metadata,
                cache_events=None,
            )
            self._output_tensors_cache[batch_size].copy_(hidden_states)
        torch.cuda.synchronize()
        self.stop_profiling("cpu")

        self._graph_cache[batch_size] = graph

        self.start_profiling("cuda")
        self._graph_cache[batch_size].replay()
        self.stop_profiling("cuda")

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

        if self._disable_cuda_graph:
            return

        max_num_seqs_rounded = self.scheduler_config.max_num_seqs + \
            (-self.scheduler_config.max_num_seqs % 8)

        for batch_size in range(8, max_num_seqs_rounded + 1, 8):
            self.init_graph(batch_size)

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        prefix_plus_current_prompt_tokens_slot_mapping: List[int] = []
        current_tokens_slot_mapping: List[int] = []

        # Add prompt tokens.
        processed_prompt_lens: List[int] = []
        current_prompt_chunk_lens: List[int] = []
        contains_prompt = False

        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            contains_prompt = True

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_chunk_size = seq_group_metadata.prompt_chunk_size
            current_prompt_chunk_tokens = seq_data.get_next_prompt_chunk_token_ids(
                prompt_chunk_size)
            current_prompt_chunk_len = len(current_prompt_chunk_tokens)
            current_prompt_chunk_lens.append(current_prompt_chunk_len)
            processed_prompt_len = seq_data.get_num_prompt_tokens_processed()
            processed_prompt_lens.append(processed_prompt_len)

            input_tokens.extend(current_prompt_chunk_tokens)
            input_positions.extend(
                range(processed_prompt_len,
                      processed_prompt_len + current_prompt_chunk_len))

            # ONLY used for profiling
            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                prefix_plus_current_prompt_tokens_slot_mapping.extend(
                    [0] * (processed_prompt_len + current_prompt_chunk_len))
                current_tokens_slot_mapping.extend([0] *
                                                   current_prompt_chunk_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(processed_prompt_len + current_prompt_chunk_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                prefix_plus_current_prompt_tokens_slot_mapping.append(slot)
                if i >= processed_prompt_len:
                    current_tokens_slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                # max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                #                              len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position //
                                           self.block_size]  # type: ignore
                block_offset = position % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                current_tokens_slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)
        current_tokens_slot_mapping = _pad_to_alignment(
            current_tokens_slot_mapping, multiple_of=8)
        context_lens = _pad_to_alignment(context_lens, multiple_of=8)
        # TODO(amey): make this configurable
        max_num_blocks_per_seq = self.model_config.max_model_len // 16
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        # extent this to closest multiple of 8
        num_pad_tokens = ((-len(padded_block_tables)) % 8)
        padded_block_tables += [[0] * max_num_blocks_per_seq] * num_pad_tokens

        # this tensor is only used in prefill, so doesn't matter for graph
        prefix_plus_current_prompt_tokens_slot_mapping_tensor = torch.tensor(
            prefix_plus_current_prompt_tokens_slot_mapping,
            dtype=torch.int,
            device="cuda")

        num_tokens = len(input_tokens)

        # Convert to tensors.
        if contains_prompt or not self.cache_config or self._disable_cuda_graph or num_tokens not in self._input_tensors_cache:
            tokens_tensor = torch.tensor(input_tokens,
                                         dtype=torch.long,
                                         device="cuda")
            positions_tensor = torch.tensor(input_positions,
                                            dtype=torch.long,
                                            device="cuda")
            current_tokens_slot_mapping_tensor = torch.tensor(
                current_tokens_slot_mapping, dtype=torch.int, device="cuda")
            context_lens_tensor = torch.tensor(context_lens,
                                               dtype=torch.int,
                                               device="cuda")
            block_tables_tensor = torch.tensor(padded_block_tables,
                                               dtype=torch.int,
                                               device="cuda")
        else:
            assert len(input_tokens) == len(input_positions)
            assert len(input_tokens) == len(current_tokens_slot_mapping)
            assert len(input_tokens) == len(
                context_lens), f"{len(input_tokens)} != {len(context_lens)}"
            assert len(input_tokens) == len(padded_block_tables)

            # just copy over data to existing tensors
            self._input_tensors_cache[num_tokens]["tokens_tensor"].copy_(
                torch.tensor(input_tokens, dtype=torch.long))
            self._input_tensors_cache[num_tokens]["positions_tensor"].copy_(
                torch.tensor(input_positions, dtype=torch.long))
            self._input_tensors_cache[num_tokens][
                "current_tokens_slot_mapping_tensor"].copy_(
                    torch.tensor(current_tokens_slot_mapping, dtype=torch.int))
            self._input_tensors_cache[num_tokens]["context_lens_tensor"].copy_(
                torch.tensor(context_lens, dtype=torch.int))
            self._input_tensors_cache[num_tokens]["block_tables_tensor"].copy_(
                torch.tensor(padded_block_tables, dtype=torch.int))

            tokens_tensor = self._input_tensors_cache[num_tokens][
                "tokens_tensor"]
            positions_tensor = self._input_tensors_cache[num_tokens][
                "positions_tensor"]
            current_tokens_slot_mapping_tensor = self._input_tensors_cache[
                num_tokens]["current_tokens_slot_mapping_tensor"]
            context_lens_tensor = self._input_tensors_cache[num_tokens][
                "context_lens_tensor"]
            block_tables_tensor = self._input_tensors_cache[num_tokens][
                "block_tables_tensor"]

        seq_data_dict: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data_dict.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data_dict,
            processed_prompt_lens=processed_prompt_lens,
            current_prompt_chunk_lens=current_prompt_chunk_lens,
            prefix_plus_current_prompt_tokens_slot_mapping=
            prefix_plus_current_prompt_tokens_slot_mapping_tensor,
            current_tokens_slot_mapping=current_tokens_slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        with self._prepare_inputs_e2e_timer:
            with self._prepare_inputs_launch_timer:
                input_tokens, input_positions, input_metadata = self._prepare_inputs(
                    seq_group_metadata_list)
            torch.cuda.synchronize()

        with self._post__prepare_inputs_barrier_timer:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        start_time = time.perf_counter()

        with self._model_execution_e2e_timer:
            with self._model_execution_launch_timer:
                if input_metadata.num_current_prompt_tokens == 0 and self.cache_config and not self._disable_cuda_graph:
                    num_tokens = input_metadata.num_generation_tokens
                    self._graph_cache[num_tokens].replay()
                    hidden_states = self._output_tensors_cache[num_tokens]
                else:
                    hidden_states = self.model(
                        input_ids=input_tokens,
                        positions=input_positions,
                        kv_caches=self.gpu_cache,
                        input_metadata=input_metadata,
                        cache_events=cache_events,
                    )
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        with self._sampler_e2e_timer:
            with self._sampler_launch_timer:
                output = self.sampler(self.model.lm_head.weight, hidden_states,
                                      input_metadata)
            torch.cuda.synchronize()

        return output, end_time - start_time

    def get_metrics_store(self) -> MetricsStore:
        return self.metrics_store

    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()

    def reset_metrics(self) -> None:
        self.metrics_store.reset()

    def start_profiling(self, activity) -> None:
        if activity == "cpu":
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
        elif activity == "cuda":
            activities = [
                torch.profiler.ProfilerActivity.CUDA,
            ]
        self.profiler = torch.profiler.profile(activities=activities, )
        self.profiler.__enter__()

    def stop_profiling(self, activity) -> None:
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(
            f"{self.metrics_config.output_dir}/profiler_trace_rank_{self.rank}_{activity}.json"
        )

    def get_gpu_id(self) -> int:
        gpu_ids = ray.get_gpu_ids()
        assert len(gpu_ids) == 1
        return gpu_ids[0]


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    rendezvous_id: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    print(
        f"Initializing nccl comm for rank {rank}, world size {parallel_config.world_size} with rendezvous id {rendezvous_id}"
    )
    init_nccl(parallel_config.world_size, rank, f"/tmp/comm_{rendezvous_id}")

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))
