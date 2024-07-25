import time
from enum import Enum, auto
from typing import List

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SimpleChunkingSchedulerConfig,
)
from sarathi.core.block_space_manager.vllm_block_space_manager import (
    VLLMBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class Turn(Enum):
    PREFILL = auto()
    DECODE = auto()


class SimpleChunkingScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SimpleChunkingSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        self.whose_turn = Turn.PREFILL

    def get_block_space_manager_class(self):
        return VLLMBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            self.chunk_size - num_batched_tokens,
        )

        return next_num_tokens

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_batched_tokens = 0
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.

        self.running = self.policy.sort_by_priority(now, self.running)

        while self.running and self.whose_turn == Turn.PREFILL:
            seq = self.running.pop(0)

            if not seq.is_paused():
                # The sequence group is already in the RUNNING state.
                running.append(seq)
                continue

            if seq.prompt_stage_processing_finished:
                running.append(seq)
                continue

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                # not enough token space to allocate the sequence
                running.append(seq)
                continue

            num_batched_tokens += next_num_prefill_tokens
            running.append(seq)
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )

        if running:
            assert not self.running
            self.running = running
            running = []

        if scheduled_seq_metadata_list:
            self.whose_turn = Turn.DECODE
            return SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=preempted_seq_ids,
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )

        while self.waiting and self.whose_turn == Turn.PREFILL:
            seq = self.waiting[0]
            # This is required to handle benchmarking where
            # we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                break

            if len(self.running) + 1 > self.scheduler_config.max_num_seqs:
                break

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                # not enough space to allocate the sequence
                break

            self.waiting.pop(0)
            self._allocate(seq)
            self.running.append(seq)
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )

        if scheduled_seq_metadata_list or ignored_seq_ids:
            self.whose_turn = Turn.DECODE
            return SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=preempted_seq_ids,
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )

        # if we reach here it means that there were no prefills
        # to execute, and we should switch to decode turn to avoid idle cycle
        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                # The sequence group is already in the RUNNING state.
                running.append(seq)
                continue

            if not seq.prompt_stage_processing_finished:
                running.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )

        self.running = running
        self.whose_turn = Turn.PREFILL
        scheduler_outputs = SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
        return scheduler_outputs
