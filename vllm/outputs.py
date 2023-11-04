from dataclasses import dataclass
from typing import Dict, List, Optional

from vllm.sequence import SequenceGroup
from vllm.sequence_status import SequenceStatus


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
    """
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: Optional[List[Dict[int, float]]] = None
    probs: Optional[List[List[float]]] = None
    finish_reason: Optional[str] = None,

    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        # Get the top-n sequences.
        n = seq_group.sampling_params.n
        seqs = seq_group.get_seqs()
        if seq_group.sampling_params.use_beam_search:
            sorting_key = lambda seq: seq.get_beam_search_score(
                seq_group.sampling_params.length_penalty)
        else:
            sorting_key = lambda seq: seq.get_cumulative_logprob()
        sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
        top_n_seqs = sorted_seqs[:n]

        # Create the outputs.
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = seq.output_logprobs
            probs = seq.output_probs
            if seq_group.sampling_params.logprobs is None:
                # NOTE: We need to take care of this case because the sequence
                # always has the logprobs of the sampled tokens even if the
                # logprobs are not requested.
                logprobs = {}
                probs = []
            finshed_reason = SequenceStatus.get_finished_reason(
                seq.get_status())
            output = CompletionOutput(seqs.index(seq), seq.output_text,
                                      seq.get_output_token_ids(),
                                      seq.get_cumulative_logprob(), logprobs,
                                      probs, finshed_reason)
            outputs.append(output)

        # Every sequence in the sequence group should have the same prompt.
        prompt = top_n_seqs[0].prompt
        prompt_token_ids = top_n_seqs[0].data.prompt_token_ids
        finished = seq_group.is_finished()
        return cls(seq_group.request_id, prompt, prompt_token_ids, outputs,
                   finished)
