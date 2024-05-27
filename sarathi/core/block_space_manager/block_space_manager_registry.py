from sarathi.config import SchedulerType
from sarathi.core.block_space_manager.faster_transformer_block_space_manager import (
    FasterTransformerBlockSpaceManager,
)
from sarathi.core.block_space_manager.orca_block_space_manager import (
    OrcaBlockSpaceManager,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.block_space_manager.simple_chunking_block_space_manager import (
    SimpleChunkingBlockSpaceManager,
)
from sarathi.core.block_space_manager.vllm_block_space_manager import (
    VLLMBlockSpaceManager,
)
from sarathi.utils.base_registry import BaseRegistry


class BlockSpaceManagerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


BlockSpaceManagerRegistry.register(SchedulerType.VLLM, VLLMBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.ORCA, OrcaBlockSpaceManager)
BlockSpaceManagerRegistry.register(
    SchedulerType.FASTER_TRANSFORMER, FasterTransformerBlockSpaceManager
)
BlockSpaceManagerRegistry.register(SchedulerType.SARATHI, SarathiBlockSpaceManager)
BlockSpaceManagerRegistry.register(
    SchedulerType.SIMPLE_CHUNKING, SimpleChunkingBlockSpaceManager
)
