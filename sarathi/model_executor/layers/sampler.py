"""A layer that samples the next tokens from the model's outputs."""

from typing import Dict, List, Optional, Tuple

import flashinfer.sampling
import torch
import torch.nn as nn
from flashinfer.sampling import sampling_from_probs as flashinfer_sampling_from_probs
from flashinfer.sampling import (
    top_k_top_p_sampling_from_logits as flashinfer_top_k_top_p_sampling_from_logits,
)

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
_MAX_TOP_K_ROUND = 32


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

        if not do_top_p and not do_top_k:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            flashinfer_sample_result = _sample_with_flashinfer(probs).cpu()
        else:
            top_ps = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
            top_ks = torch.tensor(top_ks, dtype=torch.int, device=logits.device)

            flashinfer_sample_result = _top_k_top_p_with_flashinfer(
                logits, top_ks, top_ps
            ).cpu()

        return [
            SamplerOutput(seq_metadata_list[i].seq.seq_id, flashinfer_sample_result[i])
            for i in range(len(seq_metadata_list))
        ]


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


def _top_k_top_p_with_flashinfer(
    logits: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    batch_size = logits.shape[0]
    uniform_samples = torch.empty((_MAX_TOP_K_ROUND, batch_size), device=logits.device)
    uniform_samples.uniform_()

    (batch_next_token_ids, success) = flashinfer_top_k_top_p_sampling_from_logits(
        logits, uniform_samples, top_ks, top_ps
    )

    if not success.all():
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs = flashinfer.sampling.top_k_renorm_prob(probs, top_ks)
        probs = flashinfer.sampling.top_p_renorm_prob(probs, top_ps)
        batch_next_token_ids = flashinfer_sampling_from_probs(probs, uniform_samples[0])

    return batch_next_token_ids.view(-1)


def _sample_with_flashinfer(probs: torch.Tensor) -> torch.Tensor:
    batch_size = probs.shape[0]
    uniform_samples = torch.rand(batch_size).to(probs.device)
    samples = flashinfer_sampling_from_probs(probs, uniform_samples)
    return samples
