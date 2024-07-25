from typing import List

from sarathi.config import SystemConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager


class WorkerSequenceManager(BaseSequenceManager):

    def __init__(
        self,
        config: SystemConfig,
    ):
        super().__init__(config)
        # we will have a clone of block manager here, it is supposed
        # to work in sync block manager in scheduler the idea is to avoid
        # sending block table every time to the worker
        self.block_manager = BlockSpaceManagerRegistry.get(
            config.scheduler_config.get_type(),
            config.cache_config.block_size,
            config.cache_config.num_gpu_blocks,
            config.model_config.max_model_len,
        )

    def _free_seq(self, seq_id: str) -> None:
        # ignored sequences might not have been allocated
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        if self.block_manager.is_allocated(seq):
            self.block_manager.free(seq)
        super()._free_seq(seq_id)

    def _preempt_seq(self, seq_id: str) -> None:
        super()._preempt_seq(seq_id)
        seq = self.seq_map[seq_id]
        self.block_manager.free(seq)

    def _on_seq_scheduled(self, seq_sched_metadata: SequenceScheduleMetadata) -> None:
        super()._on_seq_scheduled(seq_sched_metadata)
        seq = self.seq_map[seq_sched_metadata.seq_id]

        if self.block_manager.is_allocated(seq):
            self.block_manager.can_append_slot()
            self.block_manager.append_slot(seq)
        else:
            # lazily allocate memory when a seq
            # is allocated for the first time
            assert self.block_manager.can_allocate(seq)
            self.block_manager.allocate(seq)

    def _on_append_token(self, seq: Sequence) -> None:
        # the engine performs detokenization at this point
        # but we don't need to do anything here on worker side
        pass

    def _get_block_table(self, seq: Sequence) -> List[int]:
        return self.block_manager.get_block_table(seq)
