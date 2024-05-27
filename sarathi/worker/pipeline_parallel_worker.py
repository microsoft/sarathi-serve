"""A GPU worker class."""

from queue import Queue
from threading import Thread
from typing import Optional, Tuple

import torch
import torch.distributed

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized
from sarathi.worker.base_worker import BaseWorker

logger = init_logger(__name__)


class PipelineParallelWorker(BaseWorker):
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
        cache_config: CacheConfig,
        metrics_config: MetricsConfig,
        local_rank: int,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_config,
            parallel_config,
            scheduler_config,
            cache_config,
            metrics_config,
            local_rank,
            rank,
            distributed_init_method,
        )
        self.execution_queue = Queue()
        self.output_queue = Queue()
        self.execution_thread = Thread(target=self._execution_loop, daemon=True)

    def _verify_parallel_config(self) -> None:
        assert self.parallel_config.pipeline_parallel_size > 1

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        super().init_cache_engine(cache_config)
        self.execution_thread.start()

    def enqueue(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> None:
        self.execution_queue.put(scheduler_outputs)

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        # in pipeline parallel case, each stage won't have sampler output
        # so we need to do the book keeping update later
        pass

    @synchronized
    def on_sampling_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)

    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)

        while True:
            scheduler_outputs = self.execution_queue.get()
            output = self.execute_model(scheduler_outputs)

            if not self.is_tensor_parallel_rank_zero:
                continue

            if self.is_first_pipeline_stage or self.is_last_pipeline_stage:
                self.output_queue.put(output)

    def get_output(self) -> Optional[SamplerOutputs]:
        return self.output_queue.get()

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank
