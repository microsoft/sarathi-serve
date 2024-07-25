import copy
import math
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import zmq

from sarathi.config import ModelConfig, SystemConfig
from sarathi.core.datatypes.comm_info import CommInfo
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from sarathi.core.datatypes.step_inputs import StepInputs
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from sarathi.engine.ray_utils import RayWorker, initialize_cluster, ray
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.transformers_utils.tokenizer import get_tokenizer
from sarathi.utils import Counter, get_ip, unset_cuda_visible_devices
from sarathi.utils.threading_utils import synchronized

logger = init_logger(__name__)

_MAX_WORKER_CONCURRENCY = 1

ModelParallelRank = Tuple[int, int]


class BaseLLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the Sarathi engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    Args:
        config; System Config: The system configuration for the engine.
    """

    def __init__(
        self,
        config: SystemConfig,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={config.model_config.model!r}, "
            f"dtype={config.model_config.dtype}, "
            f"tensor_parallel_size={config.parallel_config.tensor_parallel_size}, "
            f"pipeline_parallel_size={config.parallel_config.pipeline_parallel_size}, "
            f"seed={config.model_config.seed})"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.config = config
        self._verify_args()

        self.tokenizer = get_tokenizer(
            config.model_config.model,
            trust_remote_code=config.model_config.trust_remote_code,
            revision=config.model_config.revision,
        )

        self.seq_manager = EngineSequenceManager(self.tokenizer, config)
        self.seq_counter = Counter()

        self.metrics_store = MetricsStore.get_or_create_instance(
            config.replica_config,
            config.model_config,
            config.metrics_config,
        )

        self.worker_map: Dict[ModelParallelRank, int] = {}

        # Initialize the cluster.
        initialize_cluster()

        # Create the parallel GPU workers.
        self._init_workers_ray()

        # Setup ZMQ communication channels
        self._init_zmq_sockets()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Initialize the worker map.
        self._init_worker_map()

        self.mark_initial_memory_profiling_done()

        # Create the scheduler.
        self.scheduler = SchedulerRegistry.get(
            config.scheduler_config.get_type(),
            config.model_config,
            config.scheduler_config,
            config.cache_config,
            config.parallel_config,
        )

        self._scheduler_timer = CpuTimer(CpuOperationMetrics.SCHEDULE)
        self._process_model_outputs_timer = CpuTimer(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS
        )

        self.new_seqs: List[Sequence] = []

        self._run_workers("wait_till_ready")

    def _init_zmq_sockets(self):
        self.zmq_context = zmq.Context()
        self.enqueue_socket = self.zmq_context.socket(zmq.PUB)
        self.enqueue_socket.bind(f"tcp://*:{self.comm_info.enqueue_socket_port}")
        self.output_socket = self.zmq_context.socket(zmq.PULL)
        self.output_socket.bind(f"tcp://*:{self.comm_info.output_socket_port}")

    def _validate_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size == 1

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.base_worker import (
            BaseWorker,  # pylint: disable=import-outside-toplevel
        )

        return BaseWorker

    def _init_workers_ray(self, **ray_remote_kwargs):
        resource_mapping = self.config.replica_config.get_resource_mapping(
            self.config.parallel_config.world_size
        )
        logger.info(f"Starting workers with resource mapping: {resource_mapping}")

        self.workers: List[RayWorker] = []

        unset_cuda_visible_devices()

        driver_ip = None
        for rank, (node_ip, _) in enumerate(resource_mapping):
            worker_class = ray.remote(
                num_cpus=1,
                # num_gpus=1, # we don't use ray for managing GPUs
                **ray_remote_kwargs,
            )(RayWorker)

            if node_ip:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                    resources={
                        node_ip: 0.01,
                    },
                )
            else:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                )

            if rank == 0:
                if node_ip:
                    # remove node: prefix
                    driver_ip = node_ip.split(":")[1]
                else:
                    driver_ip = get_ip()

            worker = worker_class.remote(self.config.model_config.trust_remote_code)

            self.workers.append(worker)

        self.comm_info = CommInfo(driver_ip)

        # Initialize torch distributed process group for the workers.
        config = copy.deepcopy(self.config)
        config.metrics_config = self.metrics_store.get_config_for_worker()

        worker_impl = self._get_worker_impl()

        for rank, worker in enumerate(self.workers):
            local_rank = resource_mapping[rank][1]
            promise = worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: worker_impl(
                    config,
                    local_rank,
                    rank,
                    self.comm_info,
                )
            )
            ray.get(promise)

        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _verify_args(self) -> None:
        self._validate_parallel_config()
        self.config.model_config.verify_with_parallel_config(
            self.config.parallel_config
        )

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU.
        num_gpu_blocks_across_workers = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.config.cache_config.block_size,
            gpu_memory_utilization=self.config.worker_config.gpu_memory_utilization,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(num_gpu_blocks_across_workers)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when "
                "initializing the engine."
            )
        max_blocks_per_request = math.ceil(
            self.config.model_config.max_model_len / self.config.cache_config.block_size
        )
        if num_gpu_blocks < max_blocks_per_request:
            raise ValueError(
                f"Not enough available memory to schedule a request will maximum allowed length {self.config.model_config.max_model_len}. "
                f"Need {max_blocks_per_request}, available {num_gpu_blocks} gpu blocks. "
                f"Try decreasing `max_batch_size`, `max_model_len`."
            )
        self.config.cache_config.num_gpu_blocks = num_gpu_blocks

        # Initialize the cache.
        self._run_workers(
            "init_cache_engine",
            cache_config=self.config.cache_config,
            get_all_outputs=True,
        )

    def _init_worker_map(self) -> None:
        model_parallel_ranks = self._run_workers(
            "get_model_parallel_ranks",
            get_all_outputs=True,
        )

        self.worker_map = {mp_rank: i for i, mp_rank in enumerate(model_parallel_ranks)}

    def _on_step_completed(
        self,
        scheduler_outputs: SchedulerOutputs,
        ignored_seqs: List[SequenceMetadata],
        seq_metadata_list: List[SequenceMetadata],
        sampler_outputs: Optional[SamplerOutputs],
        start_time: float,
    ) -> List[RequestOutput]:
        with self._process_model_outputs_timer:
            self.seq_manager.on_step_completed(
                scheduler_outputs,
                sampler_outputs,
            )
            self.scheduler.on_step_completed()

        end_time = time.perf_counter()

        self.metrics_store.on_batch_end(
            seq_metadata_list=seq_metadata_list,
            scheduler_outputs=scheduler_outputs,
            batch_start_time=start_time,
            batch_end_time=end_time,
        )
        all_request_outputs = self.seq_manager.generate_request_outputs(
            ignored_seqs, seq_metadata_list
        )
        return all_request_outputs

    def get_model_config(self) -> ModelConfig:
        return self.config.model_config

    def add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            seq_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()

        if not seq_id:
            seq_id = str(next(self.seq_counter))

        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.config.cache_config.block_size
        eos_token_id = self.tokenizer.eos_token_id

        seq = Sequence(
            seq_id,
            prompt,
            prompt_token_ids,
            block_size,
            eos_token_id,
            arrival_time,
            sampling_params,
        )
        # Add the sequence to the scheduler.
        self.seq_manager.add_seq(seq)
        # we create a copy of the seq so that the workers
        # receive an unmodified version of the seq
        # which is unaffected by the engine's actions
        self._append_new_seq(copy.deepcopy(seq))
        self.scheduler.add_seq(seq)
        self.metrics_store.on_request_arrival(seq)

    @synchronized
    def _append_new_seq(self, seq: Sequence) -> None:
        self.new_seqs.append(seq)

    @synchronized
    def _get_new_seqs(
        self,
    ) -> List[Sequence]:
        new_seqs = self.new_seqs
        self.new_seqs = []
        return new_seqs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seqs()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration.
        Then, it executes the model and updates the scheduler with the model outputs.
        Finally, it decodes the sequences and returns the newly generated results.
        """
        start_time = time.perf_counter()

        with self._scheduler_timer:
            scheduler_outputs = self.scheduler.schedule()

        if scheduler_outputs.is_empty():
            return []

        ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
            scheduler_outputs
        )

        self.enqueue_socket.send_pyobj(
            StepInputs(
                scheduler_outputs,
                new_seqs=self._get_new_seqs(),
            )
        )
        sampler_outputs = self.output_socket.recv_pyobj()

        return self._on_step_completed(
            scheduler_outputs,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
        )

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = partial(worker.execute_method.remote, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if ignore_output:
            return

        while True:
            try:
                all_outputs = ray.get(all_outputs, timeout=0)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0)
                continue

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def _run_worker(
        self,
        model_parallel_rank: ModelParallelRank,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        worker = self.workers[self.worker_map[model_parallel_rank]]
        executor = partial(worker.execute_method.remote, method)

        output = executor(*args, **kwargs)

        while True:
            try:
                output = ray.get(output, timeout=0)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0)
                continue

        return output

    def plot_metrics(self) -> None:
        self.metrics_store.plot()

    def pull_worker_metrics(self) -> None:
        worker_metrics = self._run_workers(
            "get_metrics_store",
            get_all_outputs=True,
        )
        for worker_metric in worker_metrics:
            self.metrics_store.merge(worker_metric)

    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()
        self._run_workers("mark_initial_memory_profiling_done", get_all_outputs=True)

    def reset_metrics(self) -> None:
        self.scheduler.reset_state()
        self.metrics_store.reset()
        self._run_workers("reset_metrics", get_all_outputs=True)

    def start_profiling(self) -> None:
        self._run_workers("start_profiling")

    def stop_profiling(self) -> None:
        self._run_workers("stop_profiling")

    def get_metric_store(self) -> MetricsStore:
        return self.metrics_store
