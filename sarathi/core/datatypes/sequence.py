"""Sequence and its related classes."""

from typing import List, Optional

from sarathi.core.datatypes.block import LogicalTokenBlock
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence_state import SequenceState
from sarathi.core.datatypes.sequence_status import SequenceStatus


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        eos_token_id: int,
        arrival_time: float,
        sampling_params: SamplingParams,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.arrival_time = arrival_time
        self.sampling_params = sampling_params
        self.prompt_token_ids = prompt_token_ids

        self.output_token_ids: List[int] = []
        self.prompt_tokens_processed = 0
        self.prompt_tokens_stage_processed = 0
        self.prompt_processing_finished = False
        self.prompt_stage_processing_finished = False

        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

        self.state = SequenceState(seq_id, arrival_time, len(prompt_token_ids))

    def get_status(self) -> SequenceStatus:
        return self.state._status

    def set_status(self, status: SequenceStatus) -> None:
        self.state.set_status(status)

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor : cursor + num_empty_slots])
            cursor += num_empty_slots

    def update_prompt_tokens_processed(self, num_tokens: int) -> None:
        assert not self.prompt_processing_finished
        assert num_tokens > 0

        self.prompt_tokens_processed += num_tokens
        assert self.prompt_tokens_processed <= len(self.prompt_token_ids)

        if self.prompt_tokens_processed == len(self.prompt_token_ids):
            self.prompt_processing_finished = True
            self.state.on_prompt_processing_completed()

    def update_prompt_tokens_stage_processed(self, num_tokens: int) -> None:
        assert not self.prompt_processing_finished
        assert not self.prompt_stage_processing_finished
        assert num_tokens > 0
        self.prompt_tokens_stage_processed += num_tokens
        assert self.prompt_tokens_stage_processed <= len(self.prompt_token_ids)
        if self.prompt_tokens_stage_processed == len(self.prompt_token_ids):
            self.prompt_stage_processing_finished = True

    def append_token_id(
        self,
        token_id: int,
    ) -> None:
        # the token need not be appended to the sequence
        # when processing partial prefill chunks
        assert self.prompt_processing_finished

        self.output_token_ids.append(token_id)
        self._append_tokens_to_blocks([token_id])
        self.state.on_token_generated()

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_num_prompt_tokens_processed(self) -> int:
        return self.prompt_tokens_processed

    def get_num_prompt_tokens_stage_processed(self) -> int:
        return self.prompt_tokens_stage_processed

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def get_output_token_ids(self) -> List[int]:
        return self.output_token_ids

    def get_next_prompt_chunk_token_ids(self, chunk_size: int) -> List[int]:
        start = self.prompt_tokens_stage_processed
        end = start + chunk_size
        assert end <= len(self.prompt_token_ids), (
            f"End index {end} is greater than the prompt length "
            f"{len(self.prompt_token_ids)}"
        )
        return self.prompt_token_ids[start:end]

    def get_next_prompt_chunk_len(self, chunk_size: int) -> int:
        return min(
            chunk_size, len(self.prompt_token_ids) - self.prompt_tokens_stage_processed
        )

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.get_status())

    def is_executing(self) -> bool:
        return SequenceStatus.is_executing(self.get_status())

    def is_waiting(self) -> bool:
        return SequenceStatus.is_waiting(self.get_status())

    def is_paused(self) -> bool:
        return SequenceStatus.is_paused(self.get_status())

    def is_running(self) -> bool:
        return SequenceStatus.is_running(self.get_status())

    def reset_for_recompute(self):
        self.set_status(SequenceStatus.WAITING)
        self.prompt_tokens_processed = 0
        self.prompt_tokens_stage_processed = 0
        self.prompt_processing_finished = False
        self.prompt_stage_processing_finished = False
        self.prompt_token_ids = self.prompt_token_ids + self.output_token_ids
        self.output_token_ids = []

    def check_stop(self) -> None:
        """Stop the finished sequences."""
        for stop_str in self.sampling_params.stop:
            if self.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                self.output_text = self.output_text[: -len(stop_str)]
                self.set_status(SequenceStatus.FINISHED_STOPPED)
                return

        # Check if the sequence has reached max_tokens.
        if self.get_output_len() == self.sampling_params.max_tokens:
            self.set_status(SequenceStatus.FINISHED_LENGTH_CAPPED)
            return

        # Check if the sequence has generated the EOS token.
        if (
            not self.sampling_params.ignore_eos
        ) and self.get_last_token_id() == self.eos_token_id:
            self.set_status(SequenceStatus.FINISHED_STOPPED)
            return

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, "
            f"status={self.get_status().name}, "
            f"num_blocks={len(self.logical_token_blocks)}, "
            f"num_prompt_tokens={len(self.prompt_token_ids)}, "
            f"num_output_tokens={len(self.output_token_ids)}, "
            f"prompt_processing_finished={self.prompt_processing_finished}, "
            f"num_prompt_tokens_processed={self.prompt_tokens_processed}, "
            f"num_prompt_tokens_stage_processed={self.prompt_tokens_stage_processed}, "
            f"prompt_stage_processing_finished={self.prompt_stage_processing_finished})"
        )


class SequenceScheduleMetadata:
    """Metadata generated by the scheduler for sequence that has been scheduled.
    This is passed to the worker, and the sequence manger is responsible for
    materializing it into a `SequenceMetadata`.

    Args:
        seq_id: The ID of the request.
        prompt_chunk_len: The size of the prompt chunk.
    """

    def __init__(
        self,
        seq_id: str,
        prompt_chunk_len: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt_chunk_len = prompt_chunk_len

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        if self.prompt_chunk_len > 0:
            return 0
        return 1

    @property
    def num_tokens(self) -> int:
        return max(self.prompt_chunk_len, 1)

    @classmethod
    def from_sequence(
        cls,
        seq: Sequence,
        prompt_chunk_len: Optional[int] = None,
    ) -> "SequenceScheduleMetadata":
        if prompt_chunk_len is None:
            if seq.prompt_stage_processing_finished:
                prompt_chunk_len = 0
            else:
                prompt_chunk_len = seq.get_prompt_len()

        return cls(seq_id=seq.seq_id, prompt_chunk_len=prompt_chunk_len)

    def __str__(self) -> str:
        return (
            f"SequenceScheduleMetadata(seq_id={self.seq_id}, "
            f"prompt_chunk_len={self.prompt_chunk_len})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class SequenceMetadata:
    """Metadata for a sequence. Used to create `SamplerMetadata`.

    Args:
        seq: The sequence object.
        prompt_chunk_len: The size of the prompt chunk.
    """

    def __init__(
        self,
        seq: Sequence,
        block_table: Optional[List[int]],
        prompt_chunk_len: int,
    ) -> None:
        self.seq = seq
        self.block_table = block_table
        self.prompt_chunk_len = prompt_chunk_len

    @property
    def num_prompt_tokens(self) -> int:
        return self.prompt_chunk_len

    @property
    def is_prompt(self) -> bool:
        return self.prompt_chunk_len > 0

    @property
    def num_output_tokens(self) -> int:
        if self.prompt_chunk_len > 0:
            return 0
        return 1

    @property
    def num_tokens(self) -> int:
        return max(self.prompt_chunk_len, 1)

    def __str__(self) -> str:
        return (
            f"SequenceMetadata(seq_id={self.seq.seq_id}, "
            f"prompt_chunk_len={self.prompt_chunk_len})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class SamplerOutput:
    """The model output associated with a sequence.

    Args:
        seq_id: The ID of sequence.
        output_token: The output token ID.
    """

    def __init__(
        self,
        seq_id: str,
        output_token: int,
    ) -> None:
        self.seq_id = seq_id
        self.output_token = output_token

    def __repr__(self) -> str:
        return (
            f"SamplerOutput(seq_id={self.seq_id}, "
            f"output_token={self.output_token}))"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplerOutput):
            raise NotImplementedError()
        return self.seq_id == other.seq_id and self.output_token == other.output_token


SamplerOutputs = List[SamplerOutput]
