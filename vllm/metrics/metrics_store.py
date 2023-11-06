from dataclasses import asdict
import os
import json
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Any

import pandas as pd
import plotly.express as px
import wandb
import logging

from vllm.config import MetricsConfig
from vllm.core.scheduler.base_scheduler import SchedulerOutputs
from vllm.metrics.constants import (
    TokenMetricsTimeDistribution,
    CpuOperationMetrics,
    HighLevelCudaOperationMetrics,
    OperationMetrics,
    SequenceMetricsTimeDistributions,
    SequenceMetricsHistogram,
    CompletionMetricsTimeSeries,
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
)
from vllm.metrics.data_series import DataSeries
from vllm.metrics.cdf_sketch import CDFSketch
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup, SequenceGroupMetadata
from vllm.singleton import Singleton

logger = logging.getLogger(__name__)


def if_write_metrics(func):

    def wrapper(self, *args, **kwargs):
        if self._should_write_metrics and self._initial_memory_profiling_done:
            return func(self, *args, **kwargs)

    return wrapper


def check_enabled(func):

    def wrapper(self, *args, **kwargs):
        if self._disabled:
            return
        return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
TIME_STR_MS = "Time (ms)"
OPERATION_STR = "Operation"


class MetricsStore(metaclass=Singleton):

    def __init__(self, metrics_config: MetricsConfig):
        self._disabled = False
        self._config = metrics_config

        if not metrics_config or not metrics_config.write_metrics:
            self._disabled = True
            return

        self._initial_memory_profiling_done = False
        self._should_write_metrics = metrics_config.write_metrics
        self._output_dir = metrics_config.output_dir
        self._subsamples = metrics_config.subsamples
        self._save_table_to_wandb = metrics_config.save_table_to_wandb

        self._wandb_project = metrics_config.wandb_project
        self._wandb_group = metrics_config.wandb_group
        self._wandb_run_name = metrics_config.wandb_run_name

        self._enable_op_level_metrics = metrics_config.enable_op_level_metrics
        self._enable_chrome_trace = metrics_config.enable_chrome_trace
        self._enable_request_outputs = metrics_config.enable_request_outputs
        self._enable_cpu_op_level_metrics = metrics_config.enable_cpu_op_level_metrics
        self._enable_high_level_cuda_metrics = metrics_config.enable_high_level_cuda_metrics
        self._tensor_parallel_size = metrics_config.tensor_parallel_size
        self._model_num_layers = metrics_config.model_num_layers

        self.reset()
        self._init_wandb()

    @property
    def disabled(self):
        return self._disabled

    @property
    def enable_op_level_metrics(self):
        return self._enable_op_level_metrics

    def is_op_enabled(self, metric_name: Any) -> bool:
        if self._disabled:
            return False

        if metric_name in self._operation_metrics:
            return self._enable_op_level_metrics
        elif metric_name in self._high_level_cuda_operation_metrics:
            return self._enable_high_level_cuda_metrics
        elif metric_name in self._cpu_operation_metrics:
            return self._enable_cpu_op_level_metrics
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")

    @check_enabled
    def reset(self):
        # Initialise request metrics
        self._seq_metrics_time_distributions: Dict[
            SequenceMetricsTimeDistributions, DataSeries] = {}
        for metric_name in SequenceMetricsTimeDistributions:
            self._seq_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._seq_metrics_histogram: Dict[SequenceMetricsHistogram,
                                          DataSeries] = {}
        for metric_name in SequenceMetricsHistogram:
            self._seq_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        # Initialise batch metrics
        self._batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, CDFSketch] = {}
        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, CDFSketch] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        # to measure the time wasted between the last batch and the next batch
        self._last_batch_end_time = None

        # Initialise completion metrics
        self._completion_metrics_time_series: Dict[CompletionMetricsTimeSeries,
                                                   DataSeries] = {}
        for metric_name in CompletionMetricsTimeSeries:
            self._completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        self._operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._high_level_cuda_operation_metrics: Dict[
            HighLevelCudaOperationMetrics, CDFSketch] = {}
        for metric_name in HighLevelCudaOperationMetrics:
            self._high_level_cuda_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, CDFSketch] = {}
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._chrome_trace: List[Dict[str, Any]] = []
        self._requests_outputs: List[RequestOutput] = []

    def _init_wandb(self):
        if (not self._should_write_metrics or not self._wandb_project
                or not self._wandb_group):
            return

        logger.info(
            f"Initializing wandb with project: {self._wandb_project}, group: {self._wandb_group}, run_name: {self._wandb_run_name}"
        )
        wandb.init(
            project=self._wandb_project,
            group=self._wandb_group,
            name=self._wandb_run_name,
        )

    def get_config_for_worker(self):
        config = deepcopy(self._config)
        config.wandb_project = None
        config.wandb_group = None

        return config

    @check_enabled
    def mark_initial_memory_profiling_done(self):
        self._initial_memory_profiling_done = True

    @check_enabled
    @if_write_metrics
    def on_request_arrival(self, sequence_group: SequenceGroup) -> None:
        assert sequence_group.num_seqs() == 1
        seq = sequence_group.get_seqs()[0]
        seq_metrics = seq.sequence_metrics

        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL].put(
                seq_metrics.arrived_at, 1)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY].put_delta(
                seq.seq_id,
                seq_metrics.arrived_at,
            )

    @if_write_metrics
    def _on_request_end(self, seq_group: SequenceGroup) -> None:
        # Treating every seq inside a sequence as an individual request
        # TODO(nitinke): Currently only support sequence groups with one sequence

        assert seq_group.num_seqs() == 1
        assert seq_group.is_finished()

        seq = seq_group.get_seqs()[0]
        seq_metrics = seq.sequence_metrics

        assert seq_metrics.is_completed

        # log request outputs and completion metrics regardless of whether the request is ignored or not
        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_COMPLETION].put(
                seq_metrics.completed_at, 1)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_NUM_IGNORED].put(
                seq_metrics.id, int(seq_metrics.is_ignore_finished))

        if seq_metrics.is_ignore_finished:
            # do not log metrics for ignored requests, they can skew the results
            return

        if self._enable_request_outputs:
            self._requests_outputs.append(
                RequestOutput.from_seq_group(seq_group))

        # first log all the histograms
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_NUM_TOKENS].put(
                seq.seq_id, seq_metrics.num_total_tokens)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_PREFILL_TOKENS].put(
                seq.seq_id, seq_metrics.num_prompt_tokens)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_DECODE_TOKENS].put(
                seq.seq_id, seq_metrics.num_output_tokens)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_PD_RATIO].put(
                seq.seq_id,
                seq_metrics.num_prompt_tokens / seq_metrics.num_output_tokens)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_NUM_RESTARTS].put(
                seq.seq_id, seq_metrics.num_restarts)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_NUM_SWAPS].put(
                seq.seq_id, seq_metrics.num_swaps)
        self._seq_metrics_histogram[
            SequenceMetricsHistogram.REQUEST_NUM_PAUSES].put(
                seq.seq_id, seq_metrics.num_pauses)

        # then log all the time distributions
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_E2E_TIME].put(
                seq.seq_id, seq_metrics.e2e_time)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED].put(
                seq.seq_id, seq_metrics.e2e_time_normalized)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.
            REQUEST_EXECUTION_PLUS_PREEMPTION_TIME].put(
                seq.seq_id,
                seq_metrics.execution_plus_preemption_time,
            )
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY].put(
                seq.seq_id,
                seq_metrics.scheduling_delay,
            )
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_EXECUTION_TIME].put(
                seq.seq_id, seq_metrics.execution_time)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.
            REQUEST_EXECUTION_TIME_NORMALIZED].put(
                seq.seq_id, seq_metrics.execution_time_normalized)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_PREEMPTION_TIME].put(
                seq.seq_id, seq_metrics.preempted_time)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.PREFILL_TIME_E2E].put(
                seq.seq_id, seq_metrics.e2e_prefill_time)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.
            PREFILL_TIME_EXECUTION_PLUS_PREEMPTION].put(
                seq.seq_id, seq_metrics.prefill_execution_plus_preemption_time)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.
            PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED].put(
                seq.seq_id,
                seq_metrics.prefill_execution_plus_preemption_time_normalized,
            )
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.
            DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED].put(
                seq.seq_id,
                seq_metrics.decode_execution_plus_preemption_time_normalized,
            )
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME].put(
                seq.seq_id, seq_metrics.model_execution_time)
        self._seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.
            REQUEST_MODEL_EXECUTION_TIME_NORMALIZED].put(
                seq.seq_id, seq_metrics.model_execution_time_normalized)

    def _update_per_token_execution_times(
        self,
        batch_end_time: float,
        seq_group: SequenceGroup,
    ) -> None:
        assert seq_group.num_seqs() == 1

        seq = seq_group.get_seqs()[0]
        seq_metrics = seq.sequence_metrics

        # determine if this was prefill or decode token
        if not seq.is_prompt_processing_finished():
            return

        # if prefill has just finished in this iteration, update the prefill completion timeseries
        if seq.get_output_len() == 1:
            self._completion_metrics_time_series[
                CompletionMetricsTimeSeries.PREFILL_COMPLETIONS].put(
                    batch_end_time,
                    seq_metrics.num_prompt_tokens,
                )

        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.DECODE_COMPLETIONS].put(
                batch_end_time, 1)

        self._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.
            DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME].put(
                seq_metrics.last_token_generation_time, )

    @check_enabled
    @if_write_metrics
    def on_batch_end(
        self,
        replica_id: int,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        scheduler_outputs: SchedulerOutputs,
        batch_start_time: float,
        batch_end_time: float,
        model_execution_time: float,
    ) -> None:
        execution_time = batch_end_time - batch_start_time

        for seq in scheduler_outputs.scheduled_seq_groups:
            seq.on_batch_end(model_execution_time)

            if seq.is_finished():
                self._on_request_end(seq)

        if self._last_batch_end_time is not None:
            self._batch_metrics_time_distribution[
                BatchMetricsTimeDistribution.INTER_BATCH_DELAY].put(
                    batch_start_time - self._last_batch_end_time, )
        self._last_batch_end_time = batch_end_time

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS].put(
                scheduler_outputs.num_batched_prompt_tokens +
                scheduler_outputs.num_batched_output_tokens, )
        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS].put(
                scheduler_outputs.num_batched_prompt_tokens)
        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS].put(
                scheduler_outputs.num_batched_output_tokens)

        batch_size = 0
        for seq_group_metadata in seq_group_metadata_list:
            batch_size += (1 if seq_group_metadata.is_prompt else len(
                seq_group_metadata.seq_data))
        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_SIZE].put(batch_size, )
        # add the only time distribution we have for batch
        self._batch_metrics_time_distribution[
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME].put(
                execution_time)

        for seq_group in scheduler_outputs.scheduled_seq_groups:
            self._update_per_token_execution_times(batch_end_time, seq_group)

        if self._enable_chrome_trace:
            self._chrome_trace.append(
                scheduler_outputs.to_chrome_trace_dict(replica_id,
                                                       batch_start_time,
                                                       batch_end_time))

    @check_enabled
    @if_write_metrics
    def push_operation_metrics(
        self,
        metrics_name: OperationMetrics,
        time: float,
    ):
        if not self._enable_op_level_metrics and not self._enable_high_level_cuda_metrics:
            return

        if metrics_name in self._operation_metrics:
            self._operation_metrics[metrics_name].put(time)
        elif metrics_name in self._high_level_cuda_operation_metrics:
            self._high_level_cuda_operation_metrics[metrics_name].put(time)

    @check_enabled
    @if_write_metrics
    def push_cpu_operation_metrics(
        self,
        metrics_name: CpuOperationMetrics,
        time: float,
    ):
        if not self._enable_cpu_op_level_metrics:
            return

        self._cpu_operation_metrics[metrics_name].put(time)

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)

        merged_request_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=[key_to_join], how="outer"),
            [dataseries._to_df() for dataseries in dataseries_list],
        )
        merged_request_df.to_csv(f"{base_path}/{file_name}.csv", index=False)

    def _store_bar_plot(self, base_path: str, plot_name: str, x_label: str,
                        y_label: str, data: Dict[str, float]):
        fig = px.bar(x=list(data.keys()),
                     y=list(data.values()),
                     labels={
                         "x": x_label,
                         "y": y_label
                     })

        if wandb.run:
            wandb.log(
                {
                    plot_name:
                    wandb.plot.bar(
                        wandb.Table(dataframe=pd.DataFrame(
                            data=data.items(), columns=[x_label, y_label])),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{base_path}/{plot_name}.png")

    def _store_request_outputs(self):
        if not self._enable_request_outputs:
            return

        self._requests_outputs.sort(key=lambda x: int(x.request_id))
        with open(f"{self._output_dir}/responses.json", "w") as f:
            json.dump(
                [asdict(response) for response in self._requests_outputs],
                f,
                indent="\t")

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._enable_op_level_metrics and \
                not self._enable_cpu_op_level_metrics and \
                not self._enable_high_level_cuda_metrics:
            return

        total_operation_runtimes: Dict[str, float] = {}
        total_operation_runtimes["op_level_cuda"] = 0
        for dataseries in self._operation_metrics.values():
            dataseries.plot_cdf(base_plot_path,
                                f"{dataseries._metric_name}_execution_time",
                                TIME_STR_MS)
            normalized_time = dataseries.sum / self._tensor_parallel_size
            if "mlp" in dataseries._metric_name \
                or "attn" in dataseries._metric_name \
                or "rms_norm" == dataseries._metric_name \
                or "add" == dataseries._metric_name:
                normalized_time = normalized_time * self._model_num_layers

            total_operation_runtimes[dataseries._metric_name] = normalized_time
            total_operation_runtimes["op_level_cuda"] += normalized_time

        total_operation_runtimes["high_level_cuda"] = 0
        for dataseries in self._high_level_cuda_operation_metrics.values():
            dataseries.plot_cdf(base_plot_path,
                                f"{dataseries._metric_name}_execution_time",
                                TIME_STR_MS)
            normalized_time = dataseries.sum / self._tensor_parallel_size
            if "mlp" in dataseries._metric_name \
                or "attn" in dataseries._metric_name:
                normalized_time = normalized_time * self._model_num_layers
            total_operation_runtimes[dataseries._metric_name] = normalized_time
            total_operation_runtimes["high_level_cuda"] += normalized_time

        total_operation_runtimes["cpu"] = 0
        for dataseries in self._cpu_operation_metrics.values():
            dataseries.plot_cdf(base_plot_path,
                                f"{dataseries._metric_name}_execution_time",
                                TIME_STR_MS)
            if dataseries._metric_name == "schedule" or dataseries._metric_name == "process_model_outputs":
                normalized_time = dataseries.sum
            else:
                normalized_time = dataseries.sum / self._tensor_parallel_size

            total_operation_runtimes[dataseries._metric_name] = normalized_time

            # skip adding time for kernel launch to avoid double counting
            if "launch" not in dataseries._metric_name:
                total_operation_runtimes["cpu"] += normalized_time

        self._store_bar_plot(base_plot_path, "total_operation_runtimes",
                             OPERATION_STR, TIME_STR_MS,
                             total_operation_runtimes)

    def _store_seq_metrics(self, base_plot_path: str):
        # all_seq_metrics = list(
        #     self._seq_metrics_time_distributions.values()) + list(
        #         self._seq_metrics_histogram.values())

        # self._save_as_csv(
        #     dataseries_list=all_seq_metrics,
        #     key_to_join=REQUEST_ID_STR,
        #     base_path=self._output_dir,
        #     file_name="sequence_metrics",
        # )

        for dataseries in self._seq_metrics_histogram.values():
            dataseries.plot_histogram(base_plot_path, dataseries._y_name)

        for dataseries in self._seq_metrics_time_distributions.values():
            dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)

    def _store_batch_metrics(self, base_plot_path: str):
        for dataseries in self._batch_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name,
                                TIME_STR)

        for dataseries in self._batch_metrics_count_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name,
                                COUNT_STR)

    def _store_completion_metrics(self, base_plot_path: str):
        for dataseries in self._token_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name,
                                TIME_STR)

        first_request_arrival_time = self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL].min_x

        for dataseries in self._completion_metrics_time_series.values():
            # subtract the first request arrival time from all the completion times
            dataseries.plot_step(base_plot_path,
                                 f"{dataseries._y_name}_time_series",
                                 COUNT_STR,
                                 start_time=first_request_arrival_time)

    @check_enabled
    def plot(self):
        base_plot_path = f"{self._output_dir}/plots/"
        os.makedirs(base_plot_path, exist_ok=True)

        self._store_seq_metrics(base_plot_path)
        self._store_batch_metrics(base_plot_path)
        self._store_completion_metrics(base_plot_path)
        self._store_request_outputs()
        self._store_operation_metrics(base_plot_path)
        self._store_chrome_trace()

    def _store_chrome_trace(self):
        if not self._enable_chrome_trace:
            return

        file_path = f"{self._output_dir}/chrome_trace.json"
        with open(file_path, "w") as f:
            json.dump(self._chrome_trace, f)

    @check_enabled
    def merge(self, other: "MetricsStore"):
        for metric_name in SequenceMetricsTimeDistributions:
            self._seq_metrics_time_distributions[metric_name].merge(
                other._seq_metrics_time_distributions[metric_name])

        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name].merge(
                other._token_metrics_time_distribution[metric_name])

        for metric_name in SequenceMetricsHistogram:
            self._seq_metrics_histogram[metric_name].merge(
                other._seq_metrics_histogram[metric_name])

        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name].merge(
                other._batch_metrics_count_distribution[metric_name])

        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name].merge(
                other._batch_metrics_time_distribution[metric_name])

        for metric_name in CompletionMetricsTimeSeries:
            self._completion_metrics_time_series[metric_name].merge(
                other._completion_metrics_time_series[metric_name])

        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name].merge(
                other._operation_metrics[metric_name])

        for metric_name in HighLevelCudaOperationMetrics:
            self._high_level_cuda_operation_metrics[metric_name].merge(
                other._high_level_cuda_operation_metrics[metric_name])

        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name].merge(
                other._cpu_operation_metrics[metric_name])

        self._chrome_trace.extend(other._chrome_trace)
        self._requests_outputs.extend(other._requests_outputs)
