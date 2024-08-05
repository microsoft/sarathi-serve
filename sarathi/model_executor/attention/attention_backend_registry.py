from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
    FlashinferAttentionWrapper,
)
from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)
from sarathi.types import AttentionBackend
from sarathi.utils.base_registry import BaseRegistry


class AttentionBackendRegistry(BaseRegistry):
    pass


AttentionBackendRegistry.register(AttentionBackend.NO_OP, NoOpAttentionWrapper)

AttentionBackendRegistry.register(
    AttentionBackend.FLASHINFER, FlashinferAttentionWrapper
)
