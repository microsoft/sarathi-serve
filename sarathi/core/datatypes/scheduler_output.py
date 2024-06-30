from typing import List

from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata


class SchedulerOutputs:

    def __init__(
        self,
        id: int,
        ignored_seq_ids: List[str],
        preempted_seq_ids: List[str],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
    ) -> None:
        self.id = id
        self.ignored_seq_ids = ignored_seq_ids
        self.preempted_seq_ids = preempted_seq_ids
        self.scheduled_seq_metadata_list = sorted(
            scheduled_seq_metadata_list, key=lambda x: not x.is_prompt
        )
        self.prompt_chunk_lens = [
            metadata.num_prompt_tokens for metadata in scheduled_seq_metadata_list
        ]
        self.num_batched_prompt_tokens = sum(self.prompt_chunk_lens)
        self.num_batched_output_tokens = sum(
            metadata.num_output_tokens for metadata in scheduled_seq_metadata_list
        )
        self.num_batched_tokens = sum(
            metadata.num_tokens for metadata in scheduled_seq_metadata_list
        )

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return not self.scheduled_seq_metadata_list

    def has_no_output(self) -> bool:
        return (
            not self.scheduled_seq_metadata_list
            and not self.ignored_seq_ids
            and not self.preempted_seq_ids
        )

    @property
    def seq_ids(self) -> List[str]:
        return [metadata.seq_id for metadata in self.scheduled_seq_metadata_list]

    def __repr__(self) -> str:
        return (
            f"SchedulerOutputs(id={self.id}, "
            f"ignored_seq_ids={self.ignored_seq_ids}, "
            f"preempted_seq_ids={self.preempted_seq_ids}, "
            f"scheduled_seq_metadata_list={self.scheduled_seq_metadata_list})"
        )
