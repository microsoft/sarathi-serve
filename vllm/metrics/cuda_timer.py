import torch

from vllm.metrics.constants import OperationMetrics
from vllm.metrics.metrics_store import MetricsStore


class CudaTimer:

    def __init__(self, name: OperationMetrics, layer_id: int = 0):
        self.name = name
        self.metrics_store = MetricsStore(None)
        self.disabled = (not self.metrics_store.is_op_enabled(self.name)
                         or layer_id != 0)

        if self.disabled:
            return

        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=self.handle_trace,
        )

    def __enter__(self):
        if self.disabled:
            return

        self.profiler.__enter__()
        return self

    def handle_trace(self, trace):
        total_cuda_time = sum(
            [e.cuda_time_total for e in trace.key_averages()])
        self.metrics_store.push_operation_metrics(
            self.name,
            total_cuda_time * 1e-3  # convert to ms
        )

    def __exit__(self, *args):
        if self.disabled:
            return

        self.profiler.__exit__(*args)
