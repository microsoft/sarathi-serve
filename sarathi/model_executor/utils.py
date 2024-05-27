"""Utils for model executor."""

import random
from typing import List

import numpy as np
import torch

from sarathi.model_executor.parallel_utils.parallel_state import (
    model_parallel_is_initialized,
)
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    model_parallel_cuda_manual_seed,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)


def round_up_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))
