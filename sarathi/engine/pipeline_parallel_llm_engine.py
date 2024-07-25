import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import List, Tuple

import zmq

from sarathi.config import SystemConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, SequenceMetadata
from sarathi.core.datatypes.step_inputs import StepInputs
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized

logger = init_logger(__name__)

SCHEDULER_LOOP_DELAY = 0.01


@dataclass
class ScheduleStageOutputs:
    ignored_seqs: List[SequenceMetadata]
    seq_metadata_list: List[SequenceMetadata]
    scheduler_outputs: SchedulerOutputs
    start_time: float


class PipelineParallelLLMEngine(BaseLLMEngine):
    """An LLM engine that receives requests and generates texts.

    This is the main class for the Sarathi engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    Args:
        config; System Config: The system configuration for the engine.
    """

    def __init__(
        self,
        config: SystemConfig,
    ) -> None:
        super().__init__(config)

        # Create the request queue.
        self.has_started_execution_loops = False
        self.scheduler_output_queue = Queue()
        self.output_queue = Queue()
        self.schedule_event = Event()
        self.microbatch_watch_event = Event()
        self.schedule_thread = Thread(target=self._schedule_loop, daemon=True)
        self.microbatch_watch_thread = Thread(
            target=self._microbatch_watch_loop, daemon=True
        )
        self.output_thread = Thread(target=self._output_loop, daemon=True)
        self.scheduler_timer_thread = Thread(
            target=self._scheduler_timer_loop, daemon=True
        )

        self.pending_step_outputs: List[Tuple[SchedulerOutputs, SamplerOutputs]] = []

    def _init_zmq_sockets(self):
        super()._init_zmq_sockets()
        # PULL socket for microbatch completion signal
        self.microbatch_socket = self.zmq_context.socket(zmq.PULL)
        self.microbatch_socket.bind(f"tcp://*:{self.comm_info.microbatch_socket_port}")

    def _validate_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size > 1

    def start_execution_loops(self) -> None:
        """Starts the execution loop."""
        self.has_started_execution_loops = True
        self.schedule_event.set()
        self.schedule_thread.start()
        self.output_thread.start()
        self.scheduler_timer_thread.start()
        self.microbatch_watch_thread.start()

    @exit_on_error
    def _scheduler_timer_loop(self) -> None:
        while True:
            time.sleep(SCHEDULER_LOOP_DELAY)
            self.schedule_event.set()

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.pipeline_parallel_worker import PipelineParallelWorker

        return PipelineParallelWorker

    @synchronized
    def _append_pending_step_output(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.pending_step_outputs.append((scheduler_outputs, sampler_outputs))

    @synchronized
    def _get_pending_step_outputs(
        self,
    ) -> List[Tuple[SchedulerOutputs, SamplerOutputs]]:
        pending_step_outputs = self.pending_step_outputs
        self.pending_step_outputs = []
        return pending_step_outputs

    @exit_on_error
    def _schedule_loop(self) -> None:
        while True:
            self.schedule_event.wait()
            self.schedule_event.clear()

            start_time = time.perf_counter()

            scheduler_outputs = self.scheduler.schedule()

            if scheduler_outputs.has_no_output():
                continue

            ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
                scheduler_outputs
            )

            self.scheduler_output_queue.put(
                ScheduleStageOutputs(
                    ignored_seqs,
                    seq_metadata_list,
                    scheduler_outputs,
                    start_time,
                )
            )

            end_time = time.perf_counter()

            if not scheduler_outputs.is_empty():
                self.microbatch_watch_event.set()
                self.enqueue_socket.send_pyobj(
                    StepInputs(
                        scheduler_outputs,
                        new_seqs=self._get_new_seqs(),
                        pending_step_outputs=self._get_pending_step_outputs(),
                    )
                )

            self.metrics_store.on_schedule(seq_metadata_list, start_time, end_time)

    @exit_on_error
    def _microbatch_watch_loop(self) -> None:
        while True:
            self.microbatch_watch_event.wait()
            self.microbatch_socket.recv_pyobj()
            self.schedule_event.set()

    @exit_on_error
    def _output_loop(self) -> None:
        while True:
            scheduler_stage_output = self.scheduler_output_queue.get()

            sampler_outputs = self.output_socket.recv_pyobj()

            self._append_pending_step_output(
                scheduler_stage_output.scheduler_outputs, sampler_outputs
            )

            all_request_outputs = self._on_step_completed(
                scheduler_stage_output.scheduler_outputs,
                scheduler_stage_output.ignored_seqs,
                scheduler_stage_output.seq_metadata_list,
                sampler_outputs,
                scheduler_stage_output.start_time,
            )
            self.schedule_event.set()

            self.output_queue.put(all_request_outputs)

    def step(self, block: bool = True) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine.
        This version does everything asynchronously and returns the results
        """
        if not self.has_started_execution_loops:
            self.start_execution_loops()

        if block:
            return self.output_queue.get()

        try:
            return self.output_queue.get(block=False)
        except Empty:
            return []
