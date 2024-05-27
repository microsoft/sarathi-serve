from sarathi.core.block_space_manager.base_block_space_manager import (
    BaseBlockSpaceManager,
)
from sarathi.core.datatypes.sequence import Sequence


class VLLMBlockSpaceManager(BaseBlockSpaceManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_num_initial_blocks(self, seq: Sequence) -> int:
        return len(seq.logical_token_blocks)
