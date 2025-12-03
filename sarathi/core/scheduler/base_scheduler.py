from abc import ABC, abstractmethod
from typing import List

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore

logger = init_logger(__name__)


class BaseScheduler(ABC):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.metrics_store = MetricsStore.get_instance()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        # we maintain this just for logging purposes
        # 保留它只是为了记录日志
        self._iteration_id = -1

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit = model_config.max_model_len

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # Sequence groups in the WAITING state.
        # wating：候补区
        self.waiting: List[Sequence] = []

        # Sequence groups in the RUNNING state.
        # 上一轮迭代结束时，还在 GPU 上运行的那些请求
        # 它的生命周期贯穿整个系统的运行过程
        # 它只在系统启动（__init__）时被初始化为空列表一次；之后，它就一直保存着系统的“状态”
        # running：正在运行区
        self.running: List[Sequence] = []

    # 重置迭代计数器
    def reset_state(self) -> None:
        self._iteration_id = -1

    # 接受新的请求
    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)

    # 状态查询
    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running

    # 状态查询
    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running)

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        pass

    # 对外的总调度入口
    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        self._iteration_id += 1

        # 如果当前流水线里的 Batch 数量已经填满了流水线的所有阶段（Stage），
        # 调度器就暂停产生新的 Batch，防止流水线溢出。
        if self.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )

        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        return scheduler_outputs

    # 垃圾回收
    def free_finished_seqs(self) -> None:
        # 遍历 self.running，检查每个请求是否结束
        for seq in self.running:
            # 如果结束：调用 _free_seq 释放显存，并将其从运行列表中移除
            if seq.is_finished():
                self._free_seq(seq)
        self.running = [seq for seq in self.running if not seq.is_finished()]

    # 当一个 Batch 在 GPU 上跑完一次迭代后调用
    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.num_running_batches -= 1

    # 初次分配
    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    # 彻底释放
    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    # 增量分配
    # 在 Decode 阶段，每生成一个新的 Token，就需要一个新的 Slot 来存放它的 KV Cache
    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        # 此方法为请求追加分配一个物理块（或在现有块中追加槽位）。
        self.block_manager.append_slot(seq)

    # 强制停止一个正在 GPU 上运行的请求，剥夺它的显存资源，并将其放回等待队列的“头等舱”（最前面），以便下次优先执行
    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self._free_seq(seq)
        self.waiting.insert(0, seq)

    # 辅助校验
    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.prompt_limit:       # 用户输入的 Prompt 长度超过了模型支持的最大长度
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.pop(0)
            return False

        return True
