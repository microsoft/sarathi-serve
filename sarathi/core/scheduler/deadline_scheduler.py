import time
from typing import Dict, List, Type
from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    DeadlineSchedulerConfig,
)
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.logger import init_logger
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.block_space_manager.deadline_block_space_manager import DeadlineBlockSpaceManager
import numpy as np
import warnings


TOKEN_SEARCH_SPACE = [128] + list(range(256, 2049, 256)) + [2552]
LOW_MEMORY_TOKEN_SEARCH_SPACE = list(range(128, 513, 128))

import heapq

logger = init_logger(__name__)

DEADLINE_MAX = 1e18

class DeadlineScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: DeadlineSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        self.execution_threshold = self.scheduler_config.execution_threshold
        self.execution_threshold_batched = self.scheduler_config.execution_threshold_batched
        self.scheduler_type = self.scheduler_config.scheduler_type
        self.hybrid_prioritization_param = self.scheduler_config.hybrid_prioritization_param
        self.expected_prefill_throughput = 8500.0  # tokens per second, initialized value
        self.prefill_throughput_weight_decay = 0.995
        self.output_len_pred = self.scheduler_config.output_len_pred
        self._prefill_queue: heapq = []
        heapq.heapify(self._prefill_queue)
        self.count = 0
        self.prefill_token_limit = self.chunk_size
        self.max_time = 0
        self.min_time = 1e18
        self.pred_thres = 1.2
        self.prev_call_time = -1
        self.last_prefill_size = 0

    
    def _predict_batch_time(self, batch_decode_context, batch_prefill_context, 
                        batch_num_prefill_tokens, batch_num_decode_tokens):
        """
        Predict execution time using two stratified linear models:
        - tokens_0_512:      0 < total_tokens <= 512
        - tokens_513_plus:   total_tokens > 512
        """
        # Total tokens
        batch_num_tokens = batch_num_prefill_tokens + batch_num_decode_tokens

        # Stratified models from latest training logs, memory_bound v/s compute_bound regime
        tokens_0_512 = {
            "coefficients": {
                "batch_num_tokens":      5.882475e-05,
                "batch_num_prefill_tokens": 2.247956e-05,
                "batch_num_decode_tokens":  3.634519e-05,
                "batch_decode_context":   1.111258e-07,
                "batch_prefill_context":  7.554773e-07,
            },
            "intercept": 0.014936,
        }

        tokens_513_plus = {
            "coefficients": {
                "batch_num_tokens":      2.719050e-05,
                "batch_num_prefill_tokens": 6.863043e-05,
                "batch_num_decode_tokens": -4.143993e-05,
                "batch_decode_context":   1.663409e-07,
                "batch_prefill_context":  2.727223e-06,
            },
            "intercept": 0.005004,
        }

        # Select bin (ensure 513 maps to the 513+ model)
        model = tokens_0_512 if (batch_num_tokens <= 512) else tokens_513_plus

        # Linear prediction
        c = model["coefficients"]
        y = model["intercept"]
        y += c["batch_num_tokens"]        * batch_num_tokens
        y += c["batch_num_prefill_tokens"]* batch_num_prefill_tokens
        y += c["batch_num_decode_tokens"] * batch_num_decode_tokens
        y += c["batch_decode_context"]    * batch_decode_context
        y += c["batch_prefill_context"]   * batch_prefill_context

        return y

    
    def _get_prefill_size_by_slack(self, min_slack, batch_decode_context, batch_prefill_context, batch_num_decode_tokens):
        """Find max prefill tokens that can be processed within min_slack time.
        Search based on total tokens (prefill + decode) but return prefill tokens,
        while respecting memory budget constraints."""
        # Token Search Space from 128 to 8192 in steps of 128
        token_search_space = TOKEN_SEARCH_SPACE
        
        # Get memory budget information
        free_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
        block_size = self.block_manager.block_size

        # We say that the memory budget is the total number of tokens whose kv cache can be stored divided by 32
        # This way if we can ensure that next few batches have sufficient memory budget, we can avoid OOM errors
        memory_budget = (free_blocks * block_size) // 32
        if memory_budget < 1000:
            # Under memory constrained conditions, range goes from 128 to 512
            token_search_space = LOW_MEMORY_TOKEN_SEARCH_SPACE
        
        # Binary search through the predefined token space
        best_total_tokens = token_search_space[0]
        best_total_tokens_pred_time = self._predict_batch_time(
            batch_decode_context, batch_prefill_context,
            max(0, best_total_tokens - batch_num_decode_tokens), batch_num_decode_tokens
        )
        if min_slack <= 0:
            return max(0, best_total_tokens - batch_num_decode_tokens), best_total_tokens_pred_time

        lo, hi = 0, len(token_search_space) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            total_tokens = token_search_space[mid]
            prefill_tokens = max(0, total_tokens - batch_num_decode_tokens)

            predicted_time = self._predict_batch_time(
                batch_decode_context, batch_prefill_context,
                prefill_tokens, batch_num_decode_tokens
            )

            # Adjust predicted time based on pred_thres
            adjusted_predicted_time = predicted_time * self.pred_thres
            if adjusted_predicted_time <= min_slack:
                # feasible -> record and try to push higher (more tokens within slack)
                best_total_tokens = total_tokens
                best_total_tokens_pred_time = predicted_time
                lo = mid + 1
            else:
                # infeasible -> need fewer total tokens
                hi = mid - 1
         # -------------------- Buffered logging --------------------
        enable_logging = False
        if enable_logging:
            if not hasattr(self, "_log_dc_buffer"):
                self._log_dc_buffer = []
            log_line = f'{batch_decode_context},{batch_prefill_context},' \
                    f'{batch_num_decode_tokens},{min_slack},' \
                    f'{best_total_tokens},{best_total_tokens_pred_time}\n'
            self._log_dc_buffer.append(log_line)
            
            # Flush every 200 rows
            if len(self._log_dc_buffer) >= 200:
                with open(f'dc_{self.pred_thres}.csv', 'a') as f:
                    f.writelines(self._log_dc_buffer)
                self._log_dc_buffer.clear()
            # -----------------------------------------------------------
        # Convert final total tokens to prefill tokens
        return max(0, best_total_tokens - batch_num_decode_tokens), best_total_tokens_pred_time

    def get_block_space_manager_class(self):
        return DeadlineBlockSpaceManager

    def _sorting_key(self, seq, now):
        if seq.prompt_processing_finished:
            slack = seq._get_TBT_deadline(self.execution_threshold, self.execution_threshold_batched) - now
            seq.last_slack = slack
            return (0, -slack)  # Primary key 0 for False, sort by negative slack
        else:
            return (1, seq.arrival_time + seq.ttft_deadline)  # Primary key 1 for True, sort by ttft_deadline
    
    def _schedule(self) -> SchedulerOutputs:

        # Fix the current time
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []
        scheduled_decode_seq_list: List[Sequence] = []
        scheduled_prefill_seq_list: List[Sequence] = []
        num_batched_tokens: int = 0
        batch_decode_context: int = 0
        batch_prefill_context: int = 0
        batch_size: int = 0
        batch_num_decode_tokens: int = 0
        batch_num_prefill_tokens: int = 0

        # Sort the running queue in non-descending order of deadlines
        # Correct policy, decodes should be ahead of prefills
        self.running = sorted(self.running, key=lambda seq: self._sorting_key(seq, now))
        
        while(self.waiting and self.waiting[0].arrival_time <= now):
            seq = self.waiting[0]
            if (seq.seq_id == 0) or (seq.seq_id == '0'):
                self.min_time = min(self.min_time, seq.arrival_time)
            self.max_time = max(self.max_time, seq.arrival_time)
            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                RuntimeError("Ignore Sequence")
                continue

            if not self.block_manager.can_allocate(seq):
                RuntimeError("Memory Error")
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                RuntimeError("Many Sequence error")
                break

            seq = self.waiting.pop(0)
            seq.scheduler_type = self.scheduler_config.scheduler_type
            seq.hybrid_prioritization_param = self.hybrid_prioritization_param
            seq.batch_id_arrived = self._iteration_id
            # For TTLT Tier sequences, we add extra time for decoding the output tokens
            if seq.tier >= 1:
                # If we are provided the number of decode tokens, we use that
                if seq.num_decode_tokens == -1:
                    seq.ttft_deadline = seq.tier_deadline - (self.output_len_pred * self.execution_threshold_batched)
                else:
                    seq.ttft_deadline = seq.tier_deadline - (seq.num_decode_tokens * self.execution_threshold_batched)
            heapq.heappush(self._prefill_queue, seq)
        
        # First pass, add all decodes in the running queue
        min_slack = 1e18
        while (
            self.running and
            num_batched_tokens < self.chunk_size
        ):

            # Stop once we reach a prefill
            if not self.running[0].prompt_processing_finished:
                break

            seq = self.running.pop(0)
            assert seq.is_paused(), "Sequence should be in PAUSED state"
            # No pre-emption support, try-catch the failed allocations
            assert self.block_manager.can_append_slot()
            try:
                assert self.block_manager.can_append_slot()
            except AssertionError:
                print("Failed to allocate slot")

            seq_decode_slack = seq._get_TBT_deadline(self.execution_threshold, self.execution_threshold_batched) - now
            min_slack = min(min_slack, seq_decode_slack)
            self._append_slot(seq)
            running.append(seq)
            scheduled_decode_seq_list.append(seq)
            batch_decode_context += seq._get_context_size()
            num_batched_tokens += 1
            batch_size += 1
            batch_num_decode_tokens += 1
        
        self.prefill_token_limit = self.chunk_size - num_batched_tokens

        # Dynamic Chunk Sizing based on slack, Dynamic Chunking
        top_prefill_seq = None
        estimated_chunk_latency = self.execution_threshold
        
        if self._prefill_queue:
            if self.scheduler_type == 'deadline':
                top_seq = heapq.heappop(self._prefill_queue)
                heapq.heappush(self._prefill_queue, top_seq)
                # We estimate the batch prefill context based on the top sequence, can assume max context size for conservative estimate
                batch_prefill_context_estimate = top_seq._get_context_size()
                self.prefill_token_limit, estimated_chunk_latency = self._get_prefill_size_by_slack(
                    min_slack, batch_decode_context, batch_prefill_context_estimate, batch_num_decode_tokens)
            else:
                self.prefill_token_limit = self.chunk_size - num_batched_tokens
        temp_prefill_queue = []

        # Update expected prefill throughput using exponential moving average
        if self.last_prefill_size != 0:
            throughput_observation = self.last_prefill_size / (now - self.prev_call_time)
            self.expected_prefill_throughput = (
                self.prefill_throughput_weight_decay * self.expected_prefill_throughput +
                (1 - self.prefill_throughput_weight_decay) * throughput_observation
            )
        self.last_prefill_size = 0
        self.prev_call_time = now
        while self._prefill_queue:
            seq = heapq.heappop(self._prefill_queue)
            remaining_prefill_tokens = seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()

            if self.scheduler_type == 'deadline' or self.scheduler_type == 'deadline_no_dynamic_chunking':
                # We use estimated prefill throughput to determine if we can finish the prefill within deadline
                service_time_required = remaining_prefill_tokens / self.expected_prefill_throughput

                # Eager Relegation: If we cannot finish the request within its deadline, we drop it and re-add to the prefill queue
                if (seq.drop == 0) and (seq.mem_allocated == False) and ((now + service_time_required) > (seq.arrival_time + seq.ttft_deadline)):
                    seq.drop += 1
                    heapq.heappush(self._prefill_queue, seq)
                    continue

            next_num_prefill_tokens = min(
                self.prefill_token_limit - batch_num_prefill_tokens,
                remaining_prefill_tokens
            )

            if remaining_prefill_tokens != next_num_prefill_tokens:
                heapq.heappush(temp_prefill_queue, seq)
            
            if next_num_prefill_tokens == 0:
                break
            num_batched_tokens += next_num_prefill_tokens
            self.last_prefill_size += next_num_prefill_tokens
            batch_num_prefill_tokens += next_num_prefill_tokens
            if not(seq.mem_allocated):
                self._allocate(seq)
                seq.mem_allocated = True
                running.append(seq)
            scheduled_prefill_seq_list.append((seq, next_num_prefill_tokens))
            batch_prefill_context += seq._get_context_size()
            batch_size += 1
        
        
        while temp_prefill_queue:
            heapq.heappush(self._prefill_queue, heapq.heappop(temp_prefill_queue))
        
        for seq in scheduled_decode_seq_list:
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq))
        
        for seq, num_tokens in scheduled_prefill_seq_list:
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=num_tokens))

        if self.running:
            self.running.extend(running)
        else:
            self.running = running
        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list
        )
