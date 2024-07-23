from typing import Optional

import torch

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.metrics_store import MetricsStore


class CudaTimer:

    def __init__(
        self,
        name: OperationMetrics,
        layer_id: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.name = name
        self.metrics_store = MetricsStore.get_instance()
        self.layer_id = layer_id
        self.disabled = (name is None) or not self.metrics_store.is_op_enabled(
            metric_name=self.name, layer_id=layer_id, rank=rank
        )

        if self.disabled:
            return

        self.use_cuda_events = False

        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=self.handle_trace,
        )
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.disabled:
            return

        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.profiler.__enter__()

        return self

    def handle_trace(self, trace):
        total_cuda_time = sum([e.cuda_time_total for e in trace.key_averages()])

        self.metrics_store.push_operation_metrics(
            self.name,
            total_cuda_time * 1e-3,  # convert to ms
        )

    def __exit__(self, *args):
        if self.disabled:
            return

        if self.use_cuda_events:
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()
            self.metrics_store.push_operation_metrics_events(
                self.name, self.start_event, self.end_event
            )
        else:
            self.profiler.__exit__(*args)
