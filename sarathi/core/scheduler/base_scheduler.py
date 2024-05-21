import time
from abc import ABC, abstractmethod
from typing import List, Tuple

from sarathi.config import CacheConfig, BaseSchedulerConfig
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.core.datatypes.sequence import Sequence
from sarathi.core.datatypes.sequence import SequenceStatus
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.core.block_space_manager.block_space_manager_registry import BlockSpaceManagerRegistry

logger = init_logger(__name__)


class BaseScheduler(ABC):

    def __init__(
        self,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.metrics_store = MetricsStore()
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        # we maintain this just for logging purposes
        self._iteration_id = -1

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.type,
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            scheduler_config.max_model_len,
        )

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[Sequence] = []
        # Sequence groups in the RUNNING state.
        self.running: List[Sequence] = []

    def reset_state(self) -> None:
        self._iteration_id = -1

    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running

    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running)

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        pass

    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        self._iteration_id += 1

        if self.num_running_batches >= self.scheduler_config.num_pipeline_stages:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        return scheduler_outputs

    def remove_finished_seqs(self) -> None:
        self.running = [seq for seq in self.running if not seq.is_finished()]

    def free_finished_seqs(self) -> None:
        for seq in self.running:
            if seq.is_finished():
                self._free_seq(seq)

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.remove_finished_seqs()
        self.num_running_batches -= 1

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self.block_manager.append_slot(seq)

    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self._free_seq(seq)
        self.waiting.insert(0, seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.scheduler_config.max_model_len:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {seq.sampling_params.max_tokens}")
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.pop(0)
            return False

        return True
