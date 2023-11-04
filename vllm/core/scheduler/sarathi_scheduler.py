from typing import List

from vllm.config import CacheConfig, SarathiSchedulerConfig
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, Sequence
from vllm.sequence_status import SequenceStatus
from vllm.core.scheduler.base_scheduler import BaseScheduler, SchedulerOutputs
from vllm.core.block_space_manager.sarathi_block_space_manager import SarathiBlockSpaceManager

logger = init_logger(__name__)


class SarathiScheduler(BaseScheduler):

    def __init__(
        self,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(scheduler_config, cache_config)

        self.prompt_limit = self.scheduler_config.max_model_len
        self.chunk_size = self.scheduler_config.chunk_size
        self.enable_rolling_prefills = self.scheduler_config.enable_rolling_prefills
        self.prefill_fitting_tolerance = self.scheduler_config.prefill_fitting_tolerance

    def _get_block_space_manger_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(self, seq: Sequence,
                                         batch_contains_prefill: bool,
                                         num_batched_tokens: int) -> int:
        assert not seq.is_finished()

        if seq.is_prompt_processing_finished():
            return 1

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(),
            self.chunk_size - num_batched_tokens)

        if not batch_contains_prefill:
            return next_num_tokens

        if self.enable_rolling_prefills and num_batched_tokens < self.chunk_size * (
                1 - self.prefill_fitting_tolerance):
            # we can have multiple prefills per batch
            # but the total number of tokens should not exceed
            # the max batch size
            return next_num_tokens
        else:
            # we will only allow one prefill per batch
            return 0

    def _schedule(self) -> SchedulerOutputs:
        ignored_seq_groups: List[SequenceGroup] = []
        prefill_seq_groups: List[SequenceGroup] = []
        decode_seq_groups: List[SequenceGroup] = []
        prefill_prompt_chunk_lens: List[int] = []
        decode_prompt_chunk_lens: List[int] = []  # just a list of zeros
        num_curr_seqs = 0
        num_batched_tokens = 0
        batch_contains_prefill = False
        num_batched_output_tokens = num_curr_seqs

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        for seq_group in self.running:
            seq = seq_group.get_seqs()[0]

            if not seq.is_prompt_processing_finished():
                continue

            num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
            num_curr_seqs += num_seqs
            num_batched_tokens += num_seqs
            num_batched_output_tokens += num_seqs
            decode_seq_groups.append(seq_group)
            decode_prompt_chunk_lens.append(0)

        # now add the requests with prefill incomplete
        for seq_group in self.running:
            seq = seq_group.get_seqs()[0]

            if seq.is_prompt_processing_finished():
                continue

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, batch_contains_prefill, num_batched_tokens)

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            assert next_num_prefill_tokens != 0

            batch_contains_prefill = True
            num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
            num_curr_seqs += num_seqs
            num_batched_tokens += next_num_prefill_tokens
            prefill_seq_groups.append(seq_group)
            prefill_prompt_chunk_lens.append(next_num_prefill_tokens)

        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            seq_group = self.waiting[0]

            assert seq_group.num_seqs() == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_prompt_tokens = seq_group.get_seqs()[0].get_len()
            if num_prompt_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.pop(0)
                continue

            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs
                    > self.scheduler_config.max_num_seqs):
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq_group.get_seqs()[0], batch_contains_prefill,
                num_batched_tokens)

            if next_num_prefill_tokens == 0:
                break

            seq_group = self.waiting.pop(0)
            self._allocate(seq_group)
            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            num_curr_seqs += num_new_seqs
            prefill_prompt_chunk_lens.append(next_num_prefill_tokens)
            prefill_seq_groups.append(seq_group)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = prefill_seq_groups + decode_seq_groups
        prompt_chunk_lens = prefill_prompt_chunk_lens + decode_prompt_chunk_lens

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_chunk_lens=prompt_chunk_lens,
            num_batched_prompt_tokens=sum(prompt_chunk_lens),
            num_batched_output_tokens=num_batched_output_tokens,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            ignored_seq_groups=ignored_seq_groups,
        )
        return scheduler_outputs
