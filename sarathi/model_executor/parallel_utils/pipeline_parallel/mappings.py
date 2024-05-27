import torch

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)


def send(hidden_states: torch.tensor):
    """Send hidden states to the next pipeline stage."""
    # Bypass the function if we are using only 1 stage.
    if get_pipeline_model_parallel_group().size() == 1:
        return hidden_states

    with CudaTimer(OperationMetrics.NCCL_SEND):
        # Send the tensor.
        torch.distributed.send(
            tensor=hidden_states,
            dst=get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )


def recv(hidden_states: torch.tensor):
    """Receive hidden states from the previous pipeline stage."""
    # Bypass the function if we are using only 1 stage.
    if get_pipeline_model_parallel_group().size() == 1:
        return hidden_states

    # Receive the tensor.
    with CudaTimer(OperationMetrics.NCCL_RECV):
        torch.distributed.recv(
            tensor=hidden_states,
            src=get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )

    return hidden_states
