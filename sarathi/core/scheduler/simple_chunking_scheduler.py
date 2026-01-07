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
        self.P = 2625
        self.B = 562
        self.M = self.chunk_size
        self.prompt_limit = self.P
        self.count = 0
        self.begin_decodes = False
        self.done = False

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

        while (
            self.begin_decodes and
            self.running
        ):

            # Stop once we reach a prefill
            if not self.running[0].prompt_processing_finished:
                break

            seq = self.running.pop(0)

            assert seq.is_paused(), "Sequence should be in PAUSED state"

            # Break if deadlines are too far ahead
            # if trigger or ((prefill_available) and (num_seqs < 275) and (seq.get_deadline(self.execution_threshold) >= (now + (0.1 * 2)))):
            #     unsched_decodes.append(seq)
            #     trigger = True
            #     continue

            # No pre-emption support, try-catch the failed allocations
            if not(self.block_manager.can_append_slot()):
                self.done = True
                break
            # try:
            #     assert self.block_manager.can_append_slot()

            # except AssertionError:
            #     print("Failed to allocate slot")
            #     print(num_seqs, num_decs, len(self.waiting), len(self.running))
            self._append_slot(seq)
            running.append(seq)
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq))
            num_batched_tokens += 1
        
        if self.done:
            return SchedulerOutputs(
                id=self.count,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[]
            )
        

        # we want our requests that violated deadlines to be after those that haven't
        # this way if we need to kick out some decodes, we kick the ones that violated deadlines first
        num_prefill_tokens = 0

        # Second pass, add any running prefills
        decodes = []

        while (
            self.running and
            num_prefill_tokens <= self.prompt_limit
        ):

            seq = self.running.pop(0)
            if seq.seq_id == 212:
                print(seq.get_num_prompt_tokens_processed(), seq.get_prompt_len(), 'seq_id')
            if seq.prompt_processing_finished:
                decodes.append(seq)
                continue
            
            assert seq.is_paused()
            next_num_prefill_tokens = min(
                self.prompt_limit - num_prefill_tokens,
                seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()
            )

            # Assumption: No pipeline parallel
            num_batched_tokens += next_num_prefill_tokens
            num_prefill_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens))
            running.append(seq)

            predicted_batch_time = 1e-9

        # Third pass, add any waiting prefills

        while (
            self.waiting and
            num_prefill_tokens <= self.prompt_limit
        ):

            seq = self.waiting[0]
            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                # sleep for the time until the next request arrives
                time.sleep(seq.arrival_time - now)

            # If the sequence cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence
                # there might be other sequences that can be allocated
                self.done = True
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            next_num_prefill_tokens = min(
                self.prompt_limit - num_prefill_tokens,
                seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()
            )

            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            num_batched_tokens += next_num_prefill_tokens
            num_prefill_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens))
            running.append(seq)
            predicted_batch_time = 1e-9
        
        if self.done:
            return SchedulerOutputs(
                id=self.count,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[]
            )



        # Add the unscheduled decodes back to the running queue

        if self.running:
            self.running.extend(running)
        else:
            self.running = running
        
        if decodes:
            self.running.extend(decodes)

        if scheduled_seq_metadata_list:
            self.count += 1
            # print(self.M, self.count, self.begin_decodes, len(self.running), 'M and count')
            if self.count == (4 * self.M):
                self.prompt_limit = self.B
                self.begin_decodes = True

        return SchedulerOutputs(
            id=self.count,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=[],
            scheduled_seq_metadata_list=scheduled_seq_metadata_list
        )
