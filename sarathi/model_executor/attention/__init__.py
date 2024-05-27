from enum import Enum
from typing import Union

from sarathi.model_executor.attention.flash_attention_wrapper import (
    FlashAttentionWrapper,
)
from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
    FlashinferAttentionWrapper,
)
from sarathi.model_executor.attention.flashinfer_unpaged_attention_wrapper import (
    FlashinferUnpagedAttentionWrapper,
)
from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)


class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    FLASH_ATTENTION = "FLASH_ATTENTION"
    FLASHINFER_UNPAGED = "FLASHINFER_UNPAGED"
    NO_OP = "NO_OP"


ATTENTION_BACKEND = AttentionBackend.NO_OP


def set_attention_backend(backend: Union[str, AttentionBackend]):
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")

    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend


def get_attention_wrapper():
    if ATTENTION_BACKEND == AttentionBackend.FLASHINFER:
        return FlashinferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FLASHINFER_UNPAGED:
        return FlashinferUnpagedAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FLASH_ATTENTION:
        return FlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.NO_OP:
        return NoOpAttentionWrapper.get_instance()

    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")
