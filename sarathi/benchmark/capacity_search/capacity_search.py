import argparse
import glob
import json
import os
import shlex
from subprocess import Popen

import pandas as pd
import ray
import wandb

from sarathi.benchmark.capacity_search.config import BenchmarkConfig, JobConfig
from sarathi.benchmark.capacity_search.ray_utils import ResourceManager, get_ip
from sarathi.logger import init_logger
from sarathi.types import ReplicaResourceMapping

logger = init_logger(__name__)


def release_resources_on_failure(func):

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in search: {e}")
            self.release_resources()

    return wrapper


class CapacitySearch:

    def __init__(
        self,
        job_config: JobConfig,
        args: argparse.Namespace,
        resource_manager: ResourceManager,
        resource_mapping: ReplicaResourceMapping,
    ):
        self.node_ip = get_ip()
        self.job_config = job_config
        self.args = args
        self.resource_manager = resource_manager
        self.resource_mapping = resource_mapping

    def release_resources(self):
        if not self.resource_mapping:
            return

        ray.get(self.resource_manager.release_resources.remote(self.resource_mapping))

    def _generate_run_command(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        resource_mapping_arg = (
            f"--replica_resource_mapping '{json.dumps(self.resource_mapping)}'"
        )
        command = f"python -m sarathi.benchmark.main {benchmark_config.to_args()} {resource_mapping_arg}"
        logger.debug(f"Running command: {command}")

        return command

    def _get_result_file(self, run_dir: str, metric_name: str) -> str:
        result_file = glob.glob(f"{run_dir}/*/*/plots/{metric_name}.csv")
        if len(result_file) == 0:
            return

        return result_file[0]

    def _is_under_sla(
        self,
        scheduling_delay_file: str,
        tbt_file: str,
        benchmark_config: BenchmarkConfig,
    ) -> tuple[bool, float, float, str]:
        scheduling_delay_df = pd.read_csv(scheduling_delay_file)
        scheduling_delay = scheduling_delay_df["request_scheduling_delay"].quantile(
            self.args.scheduling_delay_slo_quantile
        )

        tbt_df = pd.read_csv(tbt_file)
        tbt = tbt_df["decode_token_execution_plus_preemption_time"].quantile(
            self.args.tbt_slo_quantile
        )

        is_under_scheduling_delay_sla = (
            scheduling_delay <= self.args.scheduling_delay_slo_value
            and tbt <= self.args.tbt_slo_value
        )

        logger.info(
            f"{benchmark_config.to_human_readable_name()} - "
            f"Scheduling delay (P{self.args.scheduling_delay_slo_quantile}): {scheduling_delay}"
            f" - TBT (P{self.args.tbt_slo_quantile}): {tbt}"
        )
        return (
            is_under_scheduling_delay_sla,
            scheduling_delay,
            tbt,
            benchmark_config.get_run_id(),
        )

    def is_under_sla(self, qps: float) -> tuple[bool, float, float, str]:
        benchmark_config = BenchmarkConfig(
            output_dir=self.args.output_dir,
            wandb_project=self.args.wandb_project,
            wandb_group=self.job_config.get_key(),
            wandb_sweep_id=self.args.wandb_sweep_id,
            qps=qps,
            time_limit=self.args.time_limit,
            job_config=self.job_config,
        )
        run_dir = benchmark_config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        cached_scheduling_delay_file = self._get_result_file(
            run_dir, "request_scheduling_delay"
        )
        cached_tbt_file = self._get_result_file(
            run_dir, "decode_token_execution_plus_preemption_time"
        )

        if cached_scheduling_delay_file is not None and cached_tbt_file is not None:
            return self._is_under_sla(
                cached_scheduling_delay_file, cached_tbt_file, benchmark_config
            )

        command = self._generate_run_command(benchmark_config)

        output_file = open(f"{run_dir}/output.log", "w")

        # write command to a file
        output_file.write(f"Running command: {command}\n")

        args = shlex.split(command)
        p = Popen(args, stdout=output_file, stderr=output_file)
        p.wait()

        scheduling_delay_file = self._get_result_file(
            run_dir, "request_scheduling_delay"
        )
        tbt_file = self._get_result_file(
            run_dir, "decode_token_execution_plus_preemption_time"
        )
        assert (
            scheduling_delay_file is not None and tbt_file is not None
        ), f"Result file not found for {benchmark_config.to_human_readable_name()}"
        return self._is_under_sla(scheduling_delay_file, tbt_file, benchmark_config)

    @release_resources_on_failure
    def search(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """
        logger.info(f"Starting search for {self.job_config.get_human_readable_name()}")

        left = 0
        right = self.job_config.start_qps * 2
        qps = 0
        last_qps = 0
        max_qps_under_sla = None
        min_qps_over_sla = 2**32

        scheduling_delay_at_max_qps = None
        tbt_at_max_qps = None
        best_run_id = None
        found_valid_qps = False

        for _ in range(self.args.max_iterations):
            logger.info(f"Searching between {left} and {right}")
            # stopping condition - we have reached the minimum granularity
            if abs(left - right) < self.args.min_search_granularity * qps / 100:
                break

            qps = (left + right) / 2
            # round to 2 decimal places
            qps = round(qps, 2)

            if qps == last_qps:
                break

            last_qps = qps

            print(f"Searching between {left} and {right} - qps: {qps}")

            is_under_sla, scheduling_delay, tbt, run_id = self.is_under_sla(qps)

            if scheduling_delay is None:
                break

            if is_under_sla:
                found_valid_qps = True
                max_qps_under_sla = qps
                scheduling_delay_at_max_qps = scheduling_delay
                tbt_at_max_qps = tbt
                best_run_id = run_id

                if scheduling_delay < self.args.scheduling_delay_slo_value / 8:
                    # if the scheduling delay is very low, we can increase the QPS more aggressively
                    right = min(right * 4, min_qps_over_sla)
                elif scheduling_delay < self.args.scheduling_delay_slo_value / 4:
                    right = min(right * 2, min_qps_over_sla)
                elif qps > 0.8 * right:
                    right = min(right * 2, min_qps_over_sla)

                left = qps
            else:
                if scheduling_delay > 500:
                    right = qps / 2
                elif scheduling_delay > 1000:
                    right = qps / 4
                else:
                    right = qps

                min_qps_over_sla = min(min_qps_over_sla, qps)

        if not found_valid_qps:
            logger.info(
                f"No valid QPS found for {self.job_config.get_human_readable_name()}"
            )
            return {}

        logger.info(
            f"Max QPS under SLO for {self.job_config.get_human_readable_name()} - "
            f"QPS: {max_qps_under_sla}, Scheduling delay: {scheduling_delay_at_max_qps}, TBT: {tbt_at_max_qps}"
        )
        best_run = wandb.Api().run(f"{self.args.wandb_project}/{best_run_id}")
        best_run.tags.append("BEST_CONFIG")
        best_run.update()

        return {
            **self.job_config.to_config_dict(),
            "max_qps_under_sla": max_qps_under_sla,
            "scheduling_delay_at_max_qps": scheduling_delay_at_max_qps,
            "tbt_at_max_qps": tbt_at_max_qps,
        }
