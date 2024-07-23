import json
import logging
import os
import zipfile
from copy import deepcopy
from dataclasses import asdict
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import torch
import wandb

from sarathi.config import MetricsConfig, ModelConfig, ReplicaConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.metrics.cdf_sketch import CDFSketch
from sarathi.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CompletionMetricsTimeSeries,
    CpuOperationMetrics,
    OperationMetrics,
    SequenceMetricsHistogram,
    SequenceMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
    TokenMetricsTimeList,
)
from sarathi.metrics.data_series import DataSeries

logger = logging.getLogger(__name__)


def if_write_metrics(func):

    def wrapper(self, *args, **kwargs):
        if self.config.write_metrics and self.initial_memory_profiling_done:
            return func(self, *args, **kwargs)

    return wrapper


def check_enabled(func):

    def wrapper(self, *args, **kwargs):
        if self.disabled:
            return
        return func(self, *args, **kwargs)

    return wrapper


PROFILE_LAYER_ID = 10
BATCH_ID_STR = "Batch Id"
REQUEST_ID_STR = "Request Id"
DECODE_TOKEN_ID_STR = "Decode Token Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
TIME_STR_MS = "Time (ms)"
OPERATION_STR = "Operation"


class MetricsStore:

    def __init__(
        self,
        replica_config: ReplicaConfig,
        model_config: ModelConfig,
        metrics_config: MetricsConfig,
    ):
        self.disabled = False

        if not metrics_config or not metrics_config.write_metrics:
            logger.info("MetricsStore disabled")
            self.disabled = True
            return

        self.config = metrics_config
        self.replica_id = replica_config.replica_id
        self.output_dir = replica_config.output_dir
        self.initial_memory_profiling_done = False
        self.model_num_layers = model_config.get_total_num_layers()

        self.reset()
        self._init_wandb()

    @classmethod
    def get_or_create_instance(
        cls,
        replica_config: ReplicaConfig,
        model_config: ModelConfig,
        metrics_config: MetricsConfig,
    ):
        cls._instance = cls(replica_config, model_config, metrics_config)
        return cls._instance

    @classmethod
    def get_instance(cls):
        return cls._instance

    def is_op_enabled(
        self,
        metric_name: Any,
        rank: Optional[int] = None,
        layer_id: Optional[int] = None,
    ) -> bool:
        if self.disabled:
            return False

        if metric_name in self.operation_metrics:
            return self.config.enable_op_level_metrics and layer_id == PROFILE_LAYER_ID
        elif metric_name in self.cpu_operation_metrics:
            if not self.config.enable_cpu_op_level_metrics:
                return False
            if metric_name in [
                CpuOperationMetrics.SCHEDULE,
                CpuOperationMetrics.PROCESS_MODEL_OUTPUTS,
            ]:
                assert rank is None
                return True
            elif metric_name in [
                CpuOperationMetrics.PREPARE_INPUTS_E2E,
                CpuOperationMetrics.MODEL_EXECUTION_E2E,
                CpuOperationMetrics.SAMPLER_E2E,
            ]:
                return rank == 0
        raise ValueError(f"Unknown metric name: {metric_name}")

    def reset(self):
        if self.disabled:
            return

        # Initialise request metrics
        self.seq_metrics_time_distributions: Dict[
            SequenceMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in SequenceMetricsTimeDistributions:
            self.seq_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
            )

        self.token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, CDFSketch
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self.token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                relative_accuracy=0.001,
                num_quantiles_in_df=1001,
            )

        self.token_metrics_time_list: Dict[TokenMetricsTimeList, DataSeries] = {}
        for metric_name in TokenMetricsTimeList:
            self.token_metrics_time_list[metric_name] = DataSeries(
                DECODE_TOKEN_ID_STR,
                metric_name.value,
            )

        self.seq_metrics_histogram: Dict[SequenceMetricsHistogram, DataSeries] = {}
        for metric_name in SequenceMetricsHistogram:
            self.seq_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
            )

        # to measure the time interval between the last request and the next request
        self.last_request_arrived_at = None

        # Initialise batch metrics
        self.batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, Union[DataSeries, CDFSketch]
        ] = {}
        for metric_name in BatchMetricsCountDistribution:
            self.batch_metrics_count_distribution[metric_name] = (
                DataSeries(
                    BATCH_ID_STR,
                    metric_name.value,
                )
                if self.config.keep_individual_batch_metrics
                else CDFSketch(
                    metric_name.value,
                )
            )

        self.batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, Union[DataSeries, CDFSketch]
        ] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self.batch_metrics_time_distribution[metric_name] = (
                DataSeries(
                    BATCH_ID_STR,
                    metric_name.value,
                )
                if self.config.keep_individual_batch_metrics
                else CDFSketch(
                    metric_name.value,
                )
            )

        # to measure the time wasted between the last batch and the next batch
        self.last_batch_end_time = None
        self.next_batch_id = 0

        # Initialise completion metrics
        self.completion_metrics_time_series: Dict[
            CompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in CompletionMetricsTimeSeries:
            self.completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
            )

        self.operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        self.operation_metrics_per_batch: Dict[OperationMetrics, DataSeries] = {}
        self.operation_metrics_per_batch_events: Dict[
            OperationMetrics, List[Tuple[torch.cuda.Event]]
        ] = {}
        for metric_name in OperationMetrics:
            self.operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
            )
            self.operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
            )
            self.operation_metrics_per_batch_events[metric_name] = []

        self.cpu_operation_metrics: Dict[
            CpuOperationMetrics, Union[CDFSketch, DataSeries]
        ] = {}
        for metric_name in CpuOperationMetrics:
            self.cpu_operation_metrics[metric_name] = (
                DataSeries(
                    BATCH_ID_STR,
                    metric_name.value,
                )
                if self.config.keep_individual_batch_metrics
                else CDFSketch(
                    metric_name.value,
                )
            )

        self.chrome_trace: List[Dict[str, Any]] = []
        self.requests_outputs: List[RequestOutput] = []

    def _init_wandb(self):
        if (
            not self.config.write_metrics
            or not self.config.wandb_project
            or not self.config.wandb_group
        ):
            return

        logger.info(
            f"Initializing wandb with project: {self.config.wandb_project}, group: {self.config.wandb_group}, run_name: {self.config.wandb_run_name}"
            f", sweep_id: {self.config.wandb_sweep_id}, run_id: {self.config.wandb_run_id}"
        )
        if self.config.wandb_sweep_id or self.config.wandb_run_id:
            logger.warn("wandb_sweep_id and wandb_run_id are not supported yet.")

        wandb.init(
            project=self.config.wandb_project,
            group=self.config.wandb_group,
            name=self.config.wandb_run_name,
        )

    @check_enabled
    def get_config_for_worker(self):
        config = deepcopy(self.config)
        config.wandb_project = None
        config.wandb_group = None

        return config

    @check_enabled
    def mark_initial_memory_profiling_done(self):
        self.initial_memory_profiling_done = True

    def _get_seq_id(self, seq_id: str) -> str:
        return f"{self.replica_id}_{seq_id}"

    @check_enabled
    @if_write_metrics
    def on_request_arrival(self, seq: Sequence) -> None:
        self.completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put(seq.state.arrived_at, 1)
        if self.last_request_arrived_at is not None:
            self.seq_metrics_histogram[
                SequenceMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
            ].put(
                self._get_seq_id(seq.seq_id),
                seq.state.arrived_at - self.last_request_arrived_at,
            )
        self.last_request_arrived_at = seq.state.arrived_at

    @if_write_metrics
    def _on_request_end(self, seq: Sequence) -> None:
        assert seq.is_finished()
        assert seq.state.is_completed

        # log request outputs and completion metrics regardless of whether the request is ignored or not
        self.completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put(seq.state.completed_at, 1)
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_NUM_IGNORED].put(
            self._get_seq_id(seq.seq_id), int(seq.state.is_ignore_finished)
        )

        if seq.state.is_ignore_finished:
            # do not log metrics for ignored requests, they can skew the results
            return

        if self.config.enable_request_outputs:
            self.requests_outputs.append(RequestOutput.from_seq(seq))

        # first log all the histograms
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_NUM_TOKENS].put(
            self._get_seq_id(seq.seq_id), seq.state.num_total_tokens
        )
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_PREFILL_TOKENS].put(
            self._get_seq_id(seq.seq_id), seq.state.num_prompt_tokens
        )
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_DECODE_TOKENS].put(
            self._get_seq_id(seq.seq_id), seq.state.num_output_tokens
        )
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_PD_RATIO].put(
            self._get_seq_id(seq.seq_id),
            seq.state.num_prompt_tokens / seq.state.num_output_tokens,
        )
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_NUM_RESTARTS].put(
            self._get_seq_id(seq.seq_id), seq.state.num_restarts
        )
        self.seq_metrics_histogram[SequenceMetricsHistogram.REQUEST_NUM_PAUSES].put(
            self._get_seq_id(seq.seq_id), seq.state.num_pauses
        )

        # then log all the time distributions
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(self._get_seq_id(seq.seq_id), seq.state.e2e_time)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(self._get_seq_id(seq.seq_id), seq.state.e2e_time_normalized)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_E2E_TIME_PIECEWISE_NORMALIZED
        ].put(self._get_seq_id(seq.seq_id), seq.state.e2e_time_piecewise_normalized)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.execution_plus_preemption_time,
        )
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.execution_plus_preemption_time_normalized,
        )
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.scheduling_delay,
        )
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(self._get_seq_id(seq.seq_id), seq.state.execution_time)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(self._get_seq_id(seq.seq_id), seq.state.execution_time_normalized)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(self._get_seq_id(seq.seq_id), seq.state.preempted_time)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(self._get_seq_id(seq.seq_id), seq.state.e2e_prefill_time)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.PREFILL_TIME_E2E_NORMALIZED
        ].put(self._get_seq_id(seq.seq_id), seq.state.e2e_prefill_time_normalized)
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.PREFILL_TIME_E2E_PIECEWISE_NORMALIZED
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.e2e_prefill_time_piecewise_normalized,
        )
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.prefill_execution_plus_preemption_time,
        )
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.prefill_execution_plus_preemption_time_normalized,
        )
        self.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            self._get_seq_id(seq.seq_id),
            seq.state.decode_execution_plus_preemption_time_normalized,
        )

    def _update_per_token_execution_times(
        self,
        batch_end_time: float,
        seq: Sequence,
    ) -> None:
        # determine if this was prefill or decode token
        if not seq.prompt_processing_finished:
            return

        # if prefill has just finished in this iteration, update the prefill completion timeseries
        if seq.get_output_len() == 1:
            self.completion_metrics_time_series[
                CompletionMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                batch_end_time,
                seq.state.num_prompt_tokens,
            )

        self.token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(
            seq.state.last_token_generation_time,
        )

        if self.config.keep_individual_batch_metrics:
            self.completion_metrics_time_series[
                CompletionMetricsTimeSeries.DECODE_COMPLETIONS
            ].put(batch_end_time, 1)
            self.token_metrics_time_list[
                TokenMetricsTimeList.DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME_LIST
            ].put(
                f"{self._get_seq_id(seq.seq_id)}_{seq.state.num_output_tokens - 1}",
                seq.state.last_token_generation_time,
            )

    @check_enabled
    @if_write_metrics
    def on_schedule(
        self,
        seq_metadata_list: List[SequenceMetadata],
        start_time: float,
        end_time: float,
    ) -> None:
        if not self.config.enable_chrome_trace:
            return

        trace = self._to_chrome_trace_dict(
            seq_metadata_list,
            0,  # tensor_parallel_rank
            "scheduler",  # pipeline_parallel_rank - used as tid
            start_time,
            end_time,
        )

        if trace:
            self.chrome_trace.append(trace)

    @check_enabled
    @if_write_metrics
    def on_batch_stage_end(
        self,
        seq_metadata_list: List[SequenceMetadata],
        scheduler_outputs: SchedulerOutputs,
        tensor_parallel_rank: int,
        pipeline_parallel_rank: int,
        start_time: float,
        end_time: float,
    ) -> None:
        self._process_individual_batch_metrics()

        self.next_batch_id = scheduler_outputs.id + 1

        if not self.config.enable_chrome_trace or len(seq_metadata_list) == 0:
            return

        trace = self._to_chrome_trace_dict(
            seq_metadata_list,
            tensor_parallel_rank,
            pipeline_parallel_rank,
            start_time,
            end_time,
        )

        if trace:
            self.chrome_trace.append(trace)

    @check_enabled
    @if_write_metrics
    def on_batch_end(
        self,
        seq_metadata_list: List[SequenceMetadata],
        scheduler_outputs: SchedulerOutputs,
        batch_start_time: float,
        batch_end_time: float,
    ) -> None:
        self._process_individual_batch_metrics()
        self.next_batch_id = scheduler_outputs.id + 1
        execution_time = batch_end_time - batch_start_time

        for seq_metadata in seq_metadata_list:
            self._update_per_token_execution_times(batch_end_time, seq_metadata.seq)
            if seq_metadata.seq.is_finished():
                self._on_request_end(seq_metadata.seq)

        if self.last_batch_end_time is not None:
            self.batch_metrics_time_distribution[
                BatchMetricsTimeDistribution.INTER_BATCH_DELAY
            ].put_pair(
                scheduler_outputs.id,
                batch_start_time - self.last_batch_end_time,
            )
        self.last_batch_end_time = batch_end_time

        self.batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS
        ].put_pair(
            scheduler_outputs.id,
            scheduler_outputs.num_batched_prompt_tokens
            + scheduler_outputs.num_batched_output_tokens,
        )
        self.batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS
        ].put_pair(scheduler_outputs.id, scheduler_outputs.num_batched_prompt_tokens)
        self.batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS
        ].put_pair(scheduler_outputs.id, scheduler_outputs.num_batched_output_tokens)

        self.batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_SIZE
        ].put_pair(scheduler_outputs.id, len(seq_metadata_list))
        # add the only time distribution we have for batch
        self.batch_metrics_time_distribution[
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME
        ].put_pair(scheduler_outputs.id, execution_time)

    def _to_chrome_trace_dict(
        self,
        seq_metadata_list: List[SequenceMetadata],
        tensor_parallel_rank: int,
        pipeline_parallel_rank: int,
        start_time: float,
        end_time: float,
    ) -> Optional[Dict[str, Any]]:

        if tensor_parallel_rank != 0:
            return None

        seq_ids = [seq_metadata.seq.seq_id for seq_metadata in seq_metadata_list]
        prompt_chunk_lens = [
            seq_metadata.prompt_chunk_len for seq_metadata in seq_metadata_list
        ]

        num_batched_prompt_tokens = sum(prompt_chunk_lens)
        num_batched_output_tokens = len(
            [
                seq_metadata
                for seq_metadata in seq_metadata_list
                if not seq_metadata.is_prompt
            ]
        )

        num_batched_tokens = num_batched_prompt_tokens + num_batched_output_tokens

        return {
            "name": f"{seq_ids}",
            "ph": "X",
            "ts": start_time * 1e6,
            "dur": (end_time - start_time) * 1e6,
            "pid": self.replica_id,
            "tid": pipeline_parallel_rank,
            "args": {
                "batch_size": len(seq_metadata_list),
                "request_ids": seq_ids,
                "num_batched_tokens": num_batched_tokens,
                "num_batched_prompt_tokens": num_batched_prompt_tokens,
                "num_batched_output_tokens": num_batched_output_tokens,
                "prompt_chunk_lens": prompt_chunk_lens,
            },
        }

    def clear_individual_batch_metrics(self):
        for metrics_name, _ in self.operation_metrics_per_batch_events.items():
            self.operation_metrics_per_batch_events[metrics_name] = []

    def _process_individual_batch_metrics(self):
        for metrics_name, events in self.operation_metrics_per_batch_events.items():
            for event in events:
                start_event, end_event = event
                time = start_event.elapsed_time(end_event)
                self.push_operation_metrics(metrics_name, time)
            self.operation_metrics_per_batch_events[metrics_name] = []

    @check_enabled
    @if_write_metrics
    def push_operation_metrics_events(
        self,
        metrics_name: OperationMetrics,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ):
        if not self.config.enable_op_level_metrics:
            return
        if self.config.keep_individual_batch_metrics:
            self.operation_metrics_per_batch_events[metrics_name].append(
                [start_event, end_event]
            )

    @check_enabled
    @if_write_metrics
    def push_operation_metrics(
        self,
        metrics_name: OperationMetrics,
        time: float,
    ):
        if not self.config.enable_op_level_metrics:
            return
        self.operation_metrics[metrics_name].put(time)
        if self.config.keep_individual_batch_metrics:
            self.operation_metrics_per_batch[metrics_name].put(self.next_batch_id, time)

    @check_enabled
    @if_write_metrics
    def push_cpu_operation_metrics(
        self,
        metrics_name: CpuOperationMetrics,
        time: float,
    ):
        if not self.config.enable_cpu_op_level_metrics:
            return
        self.cpu_operation_metrics[metrics_name].put_pair(self.next_batch_id, time)

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)

        dataseries_dfs = [dataseries.to_df() for dataseries in dataseries_list]
        assert [
            df[key_to_join].is_unique and pd.notnull(df[key_to_join])
            for df in dataseries_dfs
        ]
        merged_df = reduce(
            lambda left, right: left.merge(right, on=key_to_join, how="outer"),
            dataseries_dfs,
        )
        merged_df.to_csv(f"{base_path}/{file_name}.csv", index=False)

    def _store_bar_plot(
        self,
        base_path: str,
        plot_name: str,
        x_label: str,
        y_label: str,
        data: Dict[str, float],
    ):
        fig = px.bar(
            x=list(data.keys()),
            y=list(data.values()),
            labels={"x": x_label, "y": y_label},
        )

        if wandb.run:
            wandb.log(
                {
                    plot_name: wandb.plot.bar(
                        wandb.Table(
                            dataframe=pd.DataFrame(
                                data=data.items(), columns=[x_label, y_label]
                            )
                        ),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{base_path}/{plot_name}.png")

    def _store_request_outputs(self):
        if not self.config.enable_request_outputs:
            return

        self.requests_outputs.sort(key=lambda x: int(x.seq_id))

        with open(f"{self.output_dir}/responses.json", "w") as f:
            json.dump(
                [asdict(response) for response in self.requests_outputs], f, indent="\t"
            )

    def _store_operation_metrics(self, base_plot_path: str):
        if (
            not self.config.enable_op_level_metrics
            and not self.config.enable_cpu_op_level_metrics
        ):
            return

        total_operation_runtimes: Dict[str, float] = {}

        for dataseries in self.operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries.metric_name}_execution_time", TIME_STR_MS
            )
            # In `is_op_enabled` we take operations from one of the layers and only rank 0 is considered.
            total_operation_runtimes[dataseries.metric_name] = (
                dataseries.sum * self.model_num_layers
            )

        for dataseries in self.cpu_operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries.metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries.metric_name] = dataseries.sum

        self._store_bar_plot(
            base_plot_path,
            "total_operation_runtimes",
            OPERATION_STR,
            TIME_STR_MS,
            total_operation_runtimes,
        )

        if not self.config.keep_individual_batch_metrics:
            return

        for dataseries in self.operation_metrics_per_batch.values():
            dataseries.consolidate()
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries.metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        operations_dataseries_list = list(self.operation_metrics_per_batch.values())
        self._save_as_csv(
            dataseries_list=operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self.output_dir,
            file_name="operation_metrics",
        )

        for dataseries in self.cpu_operation_metrics.values():
            dataseries.consolidate()
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries.metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        cpu_operations_dataseries_list = list(self.cpu_operation_metrics.values())
        self._save_as_csv(
            dataseries_list=cpu_operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self.output_dir,
            file_name="cpu_operation_metrics",
        )

    def _store_seq_metrics(self, base_plot_path: str):
        all_seq_metrics = list(self.seq_metrics_time_distributions.values()) + list(
            self.seq_metrics_histogram.values()
        )

        self._save_as_csv(
            dataseries_list=all_seq_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self.output_dir,
            file_name="sequence_metrics",
        )

        for dataseries in self.seq_metrics_histogram.values():
            dataseries.plot_histogram(base_plot_path, dataseries.y_name)

        for dataseries in self.seq_metrics_time_distributions.values():
            dataseries.plot_cdf(base_plot_path, dataseries.y_name, TIME_STR)

    def _store_batch_metrics(self, base_plot_path: str):
        if self.config.keep_individual_batch_metrics:
            all_batch_metrics = list(
                self.batch_metrics_count_distribution.values()
            ) + list(self.batch_metrics_time_distribution.values())

            self._save_as_csv(
                dataseries_list=all_batch_metrics,
                key_to_join=BATCH_ID_STR,
                base_path=self.output_dir,
                file_name="batch_metrics",
            )

        for dataseries in self.batch_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries.metric_name, TIME_STR)
            if self.config.keep_individual_batch_metrics:
                dataseries.plot_step(
                    base_plot_path,
                    f"{dataseries.metric_name}_per_batch",
                    y_axis_label=TIME_STR,
                    y_cumsum=False,
                ),

        for dataseries in self.batch_metrics_count_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries.metric_name, COUNT_STR)
            if self.config.keep_individual_batch_metrics:
                dataseries.plot_step(
                    base_plot_path,
                    f"{dataseries.metric_name}_per_batch",
                    y_axis_label=COUNT_STR,
                    y_cumsum=False,
                ),

    def _store_completion_metrics(self, base_plot_path: str):
        for dataseries in self.token_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries.metric_name, TIME_STR)
        if self.config.keep_individual_batch_metrics:
            for dataseries in self.token_metrics_time_list.values():
                dataseries.save_df(
                    path=base_plot_path, plot_name=dataseries.metric_name
                )

        first_request_arrival_time = self.completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].min_x

        for dataseries in self.completion_metrics_time_series.values():
            # subtract the first request arrival time from all the completion times
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries.y_name}_time_series",
                COUNT_STR,
                start_time=first_request_arrival_time,
            )

    def _store_chrome_trace(self):
        if not self.config.enable_chrome_trace:
            return

        file_path = f"{self.output_dir}/chrome_trace.json"
        with open(file_path, "w") as f:
            json.dump(self.chrome_trace, f)

        if wandb.run:
            zip_file_path = f"{self.output_dir}/chrome_trace.zip"
            with zipfile.ZipFile(
                zip_file_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.writestr(
                    "chrome_trace.json",
                    json.dumps(self.chrome_trace),
                )
            wandb.save(zip_file_path, policy="now")

    @check_enabled
    @if_write_metrics
    def plot(self):
        base_plot_path = f"{self.output_dir}/plots/"
        os.makedirs(base_plot_path, exist_ok=True)

        self._store_seq_metrics(base_plot_path)
        self._store_batch_metrics(base_plot_path)
        self._store_completion_metrics(base_plot_path)
        self._store_chrome_trace()
        self._store_request_outputs()
        self._store_operation_metrics(base_plot_path)

    @check_enabled
    def merge(self, other: "MetricsStore"):
        for metric_name in SequenceMetricsTimeDistributions:
            self.seq_metrics_time_distributions[metric_name].merge(
                other.seq_metrics_time_distributions[metric_name]
            )

        for metric_name in TokenMetricsTimeDistribution:
            self.token_metrics_time_distribution[metric_name].merge(
                other.token_metrics_time_distribution[metric_name]
            )

        if self.config.keep_individual_batch_metrics:
            for metric_name in TokenMetricsTimeList:
                self.token_metrics_time_list[metric_name].merge(
                    other.token_metrics_time_list[metric_name]
                )

        for metric_name in SequenceMetricsHistogram:
            self.seq_metrics_histogram[metric_name].merge(
                other.seq_metrics_histogram[metric_name]
            )

        for metric_name in BatchMetricsCountDistribution:
            self.batch_metrics_count_distribution[metric_name].merge(
                other.batch_metrics_count_distribution[metric_name]
            )

        for metric_name in BatchMetricsTimeDistribution:
            self.batch_metrics_time_distribution[metric_name].merge(
                other.batch_metrics_time_distribution[metric_name]
            )

        for metric_name in CompletionMetricsTimeSeries:
            self.completion_metrics_time_series[metric_name].merge(
                other.completion_metrics_time_series[metric_name]
            )

        for metric_name in OperationMetrics:
            if (
                metric_name in self.operation_metrics
                and metric_name in other.operation_metrics
            ):
                self.operation_metrics[metric_name].merge(
                    other.operation_metrics[metric_name]
                )

        for metric_name in OperationMetrics:
            self.operation_metrics_per_batch[metric_name].elementwise_merge(
                other.operation_metrics_per_batch[metric_name]
            )

        for metric_name in CpuOperationMetrics:
            self.cpu_operation_metrics[metric_name].merge(
                other.cpu_operation_metrics[metric_name]
            )

        self.chrome_trace.extend(other.chrome_trace)
        self.requests_outputs.extend(other.requests_outputs)
