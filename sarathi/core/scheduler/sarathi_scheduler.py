import time
from typing import List

import numpy as np

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


# 调度器在哪里被创建？ --> sarathi/engine/base_llm_engine.py
class SarathiScheduler(BaseScheduler):
    """
    【Sarathi-Serve 核心调度器】

    该类实现了论文中的核心算法，旨在解决 LLM 推理中吞吐量与延迟的权衡问题。
    它不同于传统的请求级调度（FasterTransformer）或纯迭代级调度（vLLM），
    而是采用了 "Chunked-Prefills"（分块预填充）和 "Stall-free Batching"（无停顿批处理）。

    核心机制：
    1. Token Budget (令牌预算):
       每次调度迭代（Iteration）都有一个计算量的硬性上限（self.chunk_size）。
       调度器会确保每个 Batch 的总 Token 数不超过这个预算。

    2. Stall-free Batching (无停顿调度 - Algorithm 3):
       - 优先权 1: 保证正在运行的 Decode 任务继续执行（低延迟）。
       - 优先权 2: 利用 Decode 阶段剩余的显存和算力预算，"捎带"处理新请求的 Prefill。
       - 结果: 新请求的加入不会导致正在运行的任务暂停（No Generation Stalls）。

    3. Chunked-Prefills (分块预填充):
       如果新请求的 Prompt 很长，超过了剩余的 Token Budget，
       调度器会将其切分（Split），只运行一部分。剩余部分将在后续迭代中继续处理。
    """
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        # 是否启用动态分块调度。如果启用，Budget 不再是固定值，而是根据请求处理进度动态变化的。
        self.enable_dynamic_chunking_schedule = (
            self.scheduler_config.enable_dynamic_chunking_schedule
        )
        # 以下参数仅在动态分块模式下生效
        # next four params apply only when using dynamic schedule
        self.low_chunk_size = self.scheduler_config.low_chunk_size
        self.high_chunk_size = self.scheduler_config.high_chunk_size
        self.chunk_schedule_max_tokens = self.scheduler_config.chunk_schedule_max_tokens
        self.chunk_schedule_stages = self.scheduler_config.chunk_schedule_stages

        if self.enable_dynamic_chunking_schedule:
            assert self.chunk_schedule_stages > 0
            assert self.chunk_schedule_max_tokens > 0
            assert self.low_chunk_size % 32 == 0
            assert self.high_chunk_size % 32 == 0
            self._chunk_sizes = self._compute_chunk_size_schedule()
            self._tokens_per_stage = int(
                np.ceil(self.chunk_schedule_max_tokens / self.chunk_schedule_stages)
            )

    # > 私有方法，给类自己或者子类使用的方法。外部代码不应该直接调用它
    def _compute_chunk_size_schedule(self):
        """
        [Helper] 计算动态分块的策略表。

        策略：生成一个从 high_chunk_size 到 low_chunk_size 递减的数组。
        目的：对于超长 Prompt，开始阶段使用大 Chunk 提高吞吐（高利用率）；
             接近完成时使用小 Chunk，以便灵活填补流水线空隙（减少 Bubbles）。
        """
        # create num_steps equally spaced chunk sizes between low_chunk_size and high_chunk_size
        # 使用 numpy 生成等差数列
        chunk_sizes = np.linspace(
            self.low_chunk_size,
            self.high_chunk_size,
            self.chunk_schedule_stages,
            dtype=np.int32,
        )[::-1] # [::-1] 用于反转数组，使其递减
        # align each chunk size to the nearest multiple of 32 or self.low_chunk_size
        round_of_chunk_sizes = min(32, self.low_chunk_size)
        chunk_sizes = (
            np.round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes
        )
        chunk_sizes = chunk_sizes.astype(np.int64).tolist()

        return chunk_sizes

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    # 计算当前请求在本轮迭代中应该运行多少个 Token。
    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        """
        【Chunking 核心算法】计算当前请求在本轮迭代中应该运行多少个 Token。

        逻辑公式：
            Running Tokens = min(
                Request Remaining Tokens,  (请求还剩多少没跑)
                Chunk Size - Batch Used    (当前 Batch 还剩多少预算)
            )

        Args:
            seq: 目标序列对象
            num_batched_tokens: 当前 Batch 已经累计的 Token 数量

        Returns:
            int: 本次迭代允许该序列运行的 token 数量。如果返回 0，表示预算已耗尽。
        """
        assert not seq.is_finished()

        # 1. 确定当前的 Budget 上限 (Chunk Size)
        if self.enable_dynamic_chunking_schedule:
            # 动态模式：根据请求已处理的长度，计算它处于哪个阶段 (Stage)
            request_stage_idx = int(
                np.ceil(
                    seq.get_num_prompt_tokens_stage_processed()
                    // self._tokens_per_stage
                )
            )
            # 只有当开关为 True 时（意味着 __init__ 里一定创建了该属性）才会去访问 self._chunk_sizes
            assert request_stage_idx < len(self._chunk_sizes)
            # 查表得到当前的 Chunk Size
            chunk_size = self._chunk_sizes[request_stage_idx]
        else:
            # 静态模式：使用固定的 Chunk Size
            chunk_size = self.chunk_size

        # 2. 核心计算：取最小值
        # 这就是 "Piggybacking" 和 "Chunking" 发生的时刻
        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(), # 剩余 Prompt 长度
            chunk_size - num_batched_tokens,                                    # 剩余 Budget（预算）
        )

        return next_num_tokens

    # 这是整个分块机制的“指挥中心”
    def _schedule(self) -> SchedulerOutputs:
        """
        【调度主循环】执行单次调度迭代 (One Scheduling Step)。

        对应论文 Algorithm 3 (Stall-free batching)。

        流程概要：
        1. 排序：按优先级整理运行队列。
        2. Phase 1 (Running):
           - 优先处理纯 Decode 任务（已完成 Prefill）。如果显存不足，执行抢占。
           - 其次处理进行中的 Prefill 任务（上次切分剩下的）。
        3. Phase 2 (Waiting):
           - 尝试接纳新请求。
           - 使用 _get_seq_next_num_prefill_tokens 计算是否能塞入当前 Batch。
           - 如果塞不下（Budget 满），停止接纳。
        """
        # Fix the current time.
        # 记录当前时间用于优先级排序
        now = time.monotonic()

        # 本次迭代将要运行的请求列表 （临时构建区）
        running: List[Sequence] = []        # 与self.running不同，后者是类的属性，这里是本次调度的临时变量
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        # [关键计数器] 记录当前 Batch 已占用的 Token 总数
        # 这就是我们与 Budget 进行比较的基准
        num_batched_tokens: int = 0

        ######################################################################
        # 阶段 1：将已有的运行中序列组加入 batch。(处理正在运行的请求)
        # 存在两种情况：
        # 1. 序列组的 prefill 尚未完成。这类序列的处理流程与 Sarathi 调度器中的逻辑完全一致。
        # 2. 序列组的 prefill 已经完成。在这种情况下，我们需要检查下一段解码 token 的内存可用性，
        #    并在必要时抢占（preempt）一些序列组。
        #
        # 需要注意，被抢占的序列组可能属于上述两类中的任意一种。
        ######################################################################

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        # 注意（woosuk）：只有在没有可用槽位，让所有序列组保持 RUNNING 状态时，才会发生抢占。
        # 此时，策略负责决定抢占哪些序列组。

        # 根据调度策略（如 FCFS）对队列排序
        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        # 首轮处理先所有已完成预填充的请求, 以便准确统计解码 token 的数量

        # 临时列表：用于存放尚未完成 Prefill 的请求
        running_prefills: List[Sequence] = []

        # 1.1 优先处理 Decode 任务
        # 遍历 self.running 队列，找出 seq.prompt_stage_processing_finished 为 True 的请求
        while self.running:
            # seq 是一个正在 Decode 阶段的请求（例如它已经生成了 10 个词，现在要生成第 11 个）
            seq = self.running.pop(0)

            # 如果请求被暂停，暂不处理
            if not seq.is_paused():
                running.append(seq)
                continue

            # 如果是 Prefill 还没跑完的任务，先存起来，稍后处理
            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
                continue

            # [关键逻辑] 显存不足时的抢占机制 (Preemption)
            # 这是一个 "死循环"，直到腾出空间或自我牺牲
            while not self.block_manager.can_append_slot():
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    # 如果显存不足，执行 _preempt 将低优先级请求踢回等待队列，腾出空间
                    victim_seq = self.running.pop(-1) # pop(-1) 意味着取出优先级最低的那个请求（通常是最后加入的）。

                    # 执行抢占：
                    # 1. 释放 victim_seq 占用的所有 KV Cache 块。
                    # 2. 将 victim_seq 的状态重置，并放回 self.waiting 队列头部。
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # 如果 self.running 已经空了，说明：
                    # 1. 也就是当前 Batch 里除了我（seq），其他人都被踢光了。
                    # 2. 即使这样，剩下的空间还是不够我放下一个 Token。
                    # 那么，没办法，只能把自己也停掉。
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            # 只有当 while not 循环是因为条件变为 False（即找到空间了）而正常结束时，才会执行 else 块，如果是通过 break 跳出的（自我牺牲），则不执行。
            else:
                # 终于有空间了（可能是原本就有，也可能是踢人踢出来的）
                # Append new slots to the sequence group.

                # 1. 在物理显存中真正分配这个 Slot
                self._append_slot(seq)  
                # 2. 确认 seq 继续留在本轮的运行列表中
                running.append(seq)
                # 3. 扣除 1 个 Token 的预算（Decode 消耗 1 个）
                num_batched_tokens += 1
                # 4. 记录元数据，告诉底层引擎这个 seq 要执行解码
                # TODO：具体怎么实现的还需要看 sequence.py
                scheduled_seq_metadata_list.append(
                    # SequenceScheduleMetadata.form_sequence 是一个类方法，使用seq 和 解码长度返回一个SequenceScheduleMetadata类实例
                    SequenceScheduleMetadata.from_sequence(seq)
                )

        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
        # 现在加入尚未完成预填充的请求
        # 这些预填充所需的内存早已分配完毕
        # 因此我们应该能够一次性全部运行它们
        # Phase 1.2 处理那些“上次没跑完，这次得接着跑”的 Prefill 任务（即 Chunked Prefill 的后续部分）。
        # 对应论文 `if not is_prefill_complete(R)`
        for seq in running_prefills:
            # 断言：确保这里面装的确实是还没跑完 Prefill 的任务
            assert not seq.prompt_stage_processing_finished

            # 对应算法三的 get_next_chunk_size
            # 求 min(剩余Prompt, chunk_size - 当前已用的Budget)
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            # 只要该请求先前能放入整个批次，现在也应该能放得下。
            # 因此在非流水线场景下，这个条件永远为假；
            # 但在流水线场景里，不同微批次之间请求的分组可能变化，所以并不保证永远成立。
            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue

            # 对应算法三中的 `n_t = n_t + c`
            num_batched_tokens += next_num_prefill_tokens
            # 记录元数据，告诉底层引擎这个 seq 要执行解码
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        ######################################################################
        # Phase 2: 处理等待队列中的新请求 (New Requests)
        # 目标：Piggybacking (捎带执行)，实现 Stall-free Batching
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        # 只要还有 Budget (num_batched_tokens < chunk_size)，就尝试加入新请求
        while self.waiting:
            seq = self.waiting[0]

            # This is required to handle benchmarking where we set request arrival time ahead of time
            # 处理 Benchmarking 的特殊情况（模拟请求到达时间）
            if seq.arrival_time > now:
                break

            # 检查 Prompt 是否超过模型最大支持长度
            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            # [Memory Check] 检查是否有足够显存启动该请求
            # 注意：如果显存不够，这里不会抢占，而是直接不调度该新请求（流控）
            if not self.block_manager.can_allocate(seq):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                break

            # 检查最大并发序列数限制
            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # 这些检查（Prompt长度、显存要求、最大并发序列数要求）都是算法三第14行 `can_allocate_request()`的实现

            # check if we can fit the prefill in the batch
            # [Chunking Logic] 计算这个新请求能分到多少 Budget
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # [Stop Condition] 预算已耗尽
            # 如果计算结果为 0，说明 num_batched_tokens 已经达到了 chunk_size
            # 此时停止接纳新请求
            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)       # 1. 真正从等待队列移除
            self._allocate(seq)             # 2. 在 Block Manager 中正式分配显存
            num_batched_tokens += next_num_prefill_tokens   # 3. 扣除预算
            # 4. 记录元数据：告诉引擎，这个新请求本轮只跑 next_num_prefill_tokens 这么长
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            # 5. 加入本轮运行名单
            running.append(seq)

        ######################################################################
        # 收尾
        ######################################################################
        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        # 将本轮构建好的 running 列表（包含 Phase 1 的未处理完的老请求和 Phase 2 的新请求）赋值给 self.running，作为系统的最新状态。
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
