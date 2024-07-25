import time
from typing import List

from sarathi.config import CacheConfig, ModelConfig, OrcaSchedulerConfig, ParallelConfig
from sarathi.core.block_space_manager.orca_block_space_manager import (
    OrcaBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class OrcaScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: OrcaSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

    def get_block_space_manager_class(self):
        return OrcaBlockSpaceManager

    def _schedule(self) -> SchedulerOutputs:
        ignored_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        now = time.monotonic()

        self.running = self.policy.sort_by_priority(now, self.running)

        for seq in self.running:
            if not seq.is_paused():
                continue

            assert seq.prompt_stage_processing_finished

            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )

        # Optimization: We do not sort the waiting queue since the preempted
        # sequences are added to the front and the new sequences
        # are added to the back.
        while self.waiting:
            seq = self.waiting[0]

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                break

            if len(self.running) + 1 > self.scheduler_config.max_num_seqs:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            self.running.append(seq)
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=[],
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
