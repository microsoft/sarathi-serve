"""Sequence and its related classes."""
import copy
from typing import Dict, List, Optional

from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams
from vllm.sequence_status import SequenceStatus
from vllm.metrics.sequence_metrics import SequenceMetrics


class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0
        self.prompt_tokens_processed = 0
        self.prompt_processing_finished = False

    def append_token_id(self,
                        token_id: int,
                        logprob: float,
                        chunk_size: Optional[int] = None) -> bool:

        if self.is_prompt_processing_finished():
            self.output_token_ids.append(token_id)
            self.cumulative_logprob += logprob
            return True

        assert chunk_size > 0
        self.prompt_tokens_processed += chunk_size
        assert self.prompt_tokens_processed <= len(self.prompt_token_ids)

        if self.prompt_tokens_processed == len(self.prompt_token_ids):
            self.prompt_processing_finished = True
            self.output_token_ids.append(token_id)
            self.cumulative_logprob += logprob
            return True

        return False

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

    def get_next_prompt_chunk_token_ids(self, chunk_size: int) -> List[int]:
        start = self.prompt_tokens_processed
        end = start + chunk_size
        assert end <= len(self.prompt_token_ids)
        return self.prompt_token_ids[start:end]

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def is_prompt_processing_finished(self) -> bool:
        return self.prompt_processing_finished

    def reset_for_recompute(self):
        self.prompt_tokens_processed = 0
        self.prompt_processing_finished = False
        self.prompt_token_ids = self.prompt_token_ids + self.output_token_ids
        self.output_token_ids = []

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


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
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        arrived_at: float,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_probs: List[List[float]] = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self._status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

        self.sequence_metrics = SequenceMetrics(seq_id, arrived_at,
                                                len(prompt_token_ids))

    def get_status(self) -> SequenceStatus:
        return self._status

    def set_status(self, status: SequenceStatus) -> None:
        self._status = status
        self.sequence_metrics.set_status(status)

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
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
        probs: List[float],
        chunk_size: Optional[int] = None,
    ) -> None:
        assert token_id in logprobs
        was_prompt_processing_finished = self.data.is_prompt_processing_finished(
        )

        # the token need not be appended to the sequence
        # when processing partial prefill chunks
        should_append_token = self.data.append_token_id(token_id,
                                                        logprobs[token_id],
                                                        chunk_size=chunk_size)
        if not should_append_token:
            return

        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.output_probs.append(probs)

        self.sequence_metrics.on_token_generated()

        # tokens are appended only when prompt processing completes
        # so if the prompt processing was not complete before this append
        # and it is done now then the token the last prompt chunk
        if not was_prompt_processing_finished:
            self.sequence_metrics.on_prompt_processing_completed()

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_num_prompt_tokens_processed(self) -> int:
        return self.data.prompt_tokens_processed

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # Note: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_prompt_processing_finished(self) -> bool:
        return self.data.is_prompt_processing_finished()

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self._status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def reset_for_recompute(self):
        self.set_status(SequenceStatus.WAITING)
        self.data.reset_for_recompute()

    def on_batch_end(self, model_execution_time: float):
        self.sequence_metrics.on_batch_end(model_execution_time)

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self._status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                # At prompt stage, the sequence group is not yet filled up
                # and only have one sequence running. However, in the
                # generation stage, we will have `best_of` sequences running.
                return self.sampling_params.best_of
            # At sampling stages, return the number of actual sequences
            # that are not finished yet.
            return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values()
                if seq.get_status() == status
            ]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def get_running_or_paused_seqs(self) -> List[Sequence]:
        return self.get_seqs(status=SequenceStatus.RUNNING) + self.get_seqs(
            status=SequenceStatus.PAUSED)

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def is_prompt_processing_finished(self) -> bool:
        count = sum(1 for _, seq in self.seqs_dict.items()
                    if seq.is_prompt_processing_finished())
        if count == 0:
            return False
        elif count == len(self.seqs_dict):
            return True
        else:
            raise ValueError(
                f"Some sequences ({count} out of {len(self.seqs_dict)}) are finished with prompt while others are not."
            )

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())

    def on_batch_end(self, model_execution_time: float):
        for seq in self.get_seqs():
            seq.on_batch_end(model_execution_time)

    def __repr__(self) -> str:
        return (
            f"SequenceGroup(request_id={self.request_id}, "
            f"sampling_params={self.sampling_params}, "
            f"num_seqs={len(self.seqs_dict)}, "
            f"is_prompt_processing_finished={self.is_prompt_processing_finished()}, "
            f"is_finished={self.is_finished()}")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        prompt_chunk_size: int,
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.prompt_chunk_size = prompt_chunk_size


class SequenceOutputs:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
        probs: The probabilities of all the `vocab_size` tokens among which above
            output token is produced.
    """

    def __init__(self, parent_seq_id: int, output_token: int,
                 logprobs: Dict[int, float], probs: List[float]) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs
        self.probs = probs

    def __repr__(self) -> str:
        return (f"SequenceOutputs(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}), "
                f"logprobs={self.logprobs}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutputs):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


# For each sequence group, we generate a list of SequenceOutputs object,
# each of which contains one possible candidate for the next token.
SamplerOutput = List[List[SequenceOutputs]]
