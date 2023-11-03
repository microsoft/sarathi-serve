from typing import List

from vllm.config import CacheConfig, FasterTransformerSchedulerConfig
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup
from vllm.sequence_status import SequenceStatus
from vllm.core.scheduler.base_scheduler import BaseScheduler, SchedulerOutputs
from vllm.core.block_space_manager.faster_transformer_block_space_manager import FasterTransformerBlockSpaceManager

logger = init_logger(__name__)


class FasterTransformerScheduler(BaseScheduler):

    def __init__(
        self,
        scheduler_config: FasterTransformerSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(scheduler_config, cache_config)

        self.prompt_limit = self.scheduler_config.max_model_len

    def _get_block_space_manger_class(self):
        return FasterTransformerBlockSpaceManager

    def _schedule(self) -> SchedulerOutputs:
        ignored_seq_groups: List[SequenceGroup] = []
        prompt_chunk_lens: List[int] = [0] * len(self.running)

        num_curr_seqs = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)
        num_batched_tokens = num_curr_seqs
        num_batched_output_tokens = num_curr_seqs

        if num_curr_seqs > 0:
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

        assert self.running == []

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

            seq_group = self.waiting.pop(0)
            self._allocate(seq_group)
            self.running.append(seq_group)
            num_batched_tokens += num_prompt_tokens
            num_curr_seqs += num_new_seqs
            prompt_chunk_lens.append(num_prompt_tokens)

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
