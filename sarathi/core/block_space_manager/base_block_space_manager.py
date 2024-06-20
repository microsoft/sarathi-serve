"""A block manager that manages token blocks."""

from abc import ABC, abstractmethod
from typing import Dict, List

from sarathi.core.datatypes.block import PhysicalTokenBlock
from sarathi.core.datatypes.sequence import Sequence


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(block_number=i, block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class BaseBlockSpaceManager(ABC):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(block_size, num_gpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[str, BlockTable] = {}

    @abstractmethod
    def get_num_initial_blocks(self, seq: Sequence) -> int:
        """Returns the number of blocks to allocate for a request initially."""
        pass

    def can_allocate(self, seq: Sequence) -> bool:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        num_required_blocks = self.get_num_initial_blocks(seq)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, seq: Sequence) -> None:
        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        num_initial_blocks = self.get_num_initial_blocks(seq)
        for _ in range(num_initial_blocks):
            block = self.gpu_allocator.allocate()
            block_table.append(block)

        self.block_tables[seq.seq_id] = block_table

    def can_append_slot(self) -> bool:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks > 0

    def append_slot(self, seq: Sequence) -> None:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)

    def _get_physical_blocks(self, seq: Sequence) -> BlockTable:
        assert seq.is_executing()
        return self.block_tables[seq.seq_id]

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            self.gpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def is_allocated(self, seq: Sequence) -> bool:
        return seq.seq_id in self.block_tables
