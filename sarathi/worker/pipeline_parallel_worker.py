"""A GPU worker class."""

from typing import Tuple

import torch
import torch.distributed
import zmq

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized
from sarathi.worker.base_worker import BaseWorker

logger = init_logger(__name__)


class PipelineParallelWorker(BaseWorker):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def _init_zmq_sockets(self):
        super()._init_zmq_sockets()

        self.microbatch_socket = self.zmq_context.socket(zmq.PUSH)
        self.microbatch_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.microbatch_socket_port}"
        )

    def _verify_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size > 1

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        # in pipeline parallel case, each stage won't have sampler output
        # so we don't do anything here
        pass

    @synchronized
    def on_sampling_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)

    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)

        self.worker_ready_event.set()

        while True:
            step_inputs = self.enqueue_socket.recv_pyobj()

            for new_seq in step_inputs.new_seqs:
                self.seq_manager.add_seq(new_seq)

            for pending_step_output in step_inputs.pending_step_outputs:
                self.seq_manager.on_step_completed(
                    pending_step_output[0], pending_step_output[1]
                )

            output = self.execute_model(step_inputs.scheduler_outputs)

            if not self.is_tensor_parallel_rank_zero:
                continue

            if self.is_last_pipeline_stage:
                self.output_socket.send_pyobj(output)
            elif self.is_first_pipeline_stage:
                self.microbatch_socket.send_pyobj(None)

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank
