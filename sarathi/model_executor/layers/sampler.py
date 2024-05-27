"""A layer that samples the next tokens from the model's outputs."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sarathi.core.datatypes.sampling_params import SamplingType
from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    SequenceMetadata,
)
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
)

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, embedding: torch.Tensor, vocab_size: int) -> None:
        super().__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_metadata_list: List[SequenceMetadata],
    ) -> SamplerOutputs:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, seq_metadata_list)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, self.embedding, self.vocab_size)

        # Apply temperature scaling.
        temperatures = _get_temperatures(seq_metadata_list)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks = _get_top_p_top_k(seq_metadata_list, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        return _sample(probs, logprobs, seq_metadata_list)


def _get_logits(
    hidden_states: torch.Tensor, embedding: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    logits = gather_from_tensor_model_parallel_region(logits)
    # Remove paddings in vocab (if any).
    logits = logits[:, :vocab_size]
    return logits


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    seq_metadata_list: List[SequenceMetadata],
) -> torch.Tensor:
    last_token_indices = []
    token_idx = 0
    for seq_metadata in seq_metadata_list:
        if seq_metadata.is_prompt:
            prompt_len = seq_metadata.prompt_chunk_len
            last_token_indices.append(token_idx + prompt_len - 1)
            token_idx += prompt_len
        else:
            last_token_indices.append(token_idx)
            token_idx += 1

    last_token_indices = torch.tensor(
        last_token_indices, dtype=torch.long, device=hidden_states.device
    )
    return hidden_states.index_select(0, last_token_indices)


def _get_temperatures(seq_metadata_list: List[SequenceMetadata]) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for seq_metadata in seq_metadata_list:
        temperature = seq_metadata.seq.sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0
        temperatures.append(temperature)
    return temperatures


def _get_top_p_top_k(
    seq_metadata_list: List[SequenceMetadata],
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for seq_metadata in seq_metadata_list:
        top_p = seq_metadata.seq.sampling_params.top_p
        # k should not be greater than the vocab size.
        top_k = min(seq_metadata.seq.sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        top_ps.append(top_p)
        top_ks.append(top_k)
    return top_ps, top_ks


def _apply_top_p_top_k(
    logits: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))
    return logits


def _greedy_sample(
    logprobs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    return torch.argmax(logprobs, dim=-1).view(-1).cpu().tolist()


def _random_sample(
    probs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    random_samples = (
        torch.multinomial(probs, num_samples=1, replacement=True)
        .view(-1)
        .cpu()
        .tolist()
    )

    return random_samples


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_metadata_list: List[SequenceMetadata],
) -> SamplerOutputs:
    categorized_seq_indices = {t: [] for t in SamplingType}
    category_num_tokens = {t: 0 for t in SamplingType}
    for i, seq_metadata in enumerate(seq_metadata_list):
        sampling_type = seq_metadata.seq.sampling_params.sampling_type
        categorized_seq_indices[sampling_type].append(i)
        category_num_tokens[sampling_type] += 1

    outputs: List[SamplerOutput] = []
    category_start_idx = 0
    for sampling_type in SamplingType:
        seq_indices = categorized_seq_indices[sampling_type]
        seq_ids = [seq_metadata_list[i].seq.seq_id for i in seq_indices]
        num_tokens = category_num_tokens[sampling_type]
        if num_tokens == 0:
            continue
        category_logprobs = logprobs[
            category_start_idx : category_start_idx + num_tokens
        ]
        category_probs = probs[category_start_idx : category_start_idx + num_tokens]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(category_logprobs)
        elif sampling_type == SamplingType.RANDOM:
            sample_results = _random_sample(category_probs)
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

        for seq_id, sample_result in zip(seq_ids, sample_results):
            outputs.append(SamplerOutput(seq_id, sample_result))

    return outputs
