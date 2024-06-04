import argparse
import glob
import os
import shlex
import json
from subprocess import Popen

import pandas as pd
import ray

from sarathi.benchmark.capacity_search.config import (
    JobConfig,
    BenchmarkConfig,
)
from sarathi.benchmark.capacity_search.ray_utils import (
    ResourceManager,
    get_ip,
)
from sarathi.benchmark.types import ReplicaResourceMapping


def release_resources_on_failure(func):

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(f"Error in search: {e}", flush=True)
            self.release_resources()

    return wrapper


class IsoQPSSearch:

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

        ray.get(
            self.resource_manager.release_resources.remote(
                self.resource_mapping))

    def _generate_run_command(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        resource_mapping_arg = f"--replica_resource_mapping '{json.dumps(self.resource_mapping)}'"
        command = f"python -m sarathi.benchmark.main {benchmark_config.to_args()} {resource_mapping_arg}"
        if self.args.debug:
            print(f"Running command: {command}", flush=True)

        return command

    def _get_result_file(self, run_dir: str, metric_name: str) -> str:
        result_file = glob.glob(f"{run_dir}/*/*/plots/{metric_name}.csv")
        if len(result_file) == 0:
            return

        return result_file[0]

    def _extract_stats(
        self,
        ttft_file: str,
        tbt_file: str,
        benchmark_config: BenchmarkConfig,
    ) -> tuple[bool, float, float, str]:
        ttft_df = pd.read_csv(ttft_file)
        ttft = ttft_df["prefill_e2e_time"].quantile(self.args.ttft_slo_quantile)

        tbt_df = pd.read_csv(tbt_file)
        tbt = tbt_df["decode_token_execution_plus_preemption_time"].quantile(
            self.args.tbt_slo_quantile)

        print(
            f"{benchmark_config.to_human_readable_name()} - "
            f"TTFT (P{self.args.ttft_slo_quantile}): {ttft}"
            f" - TBT (P{self.args.tbt_slo_quantile}): {tbt}",
            flush=True,
        )
        return ttft, tbt

    def _run_job(self, qps: float) -> tuple[bool, float, float, str]:
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

        cached_ttft_file = self._get_result_file(
            run_dir, "prefill_e2e_time")
        cached_tbt_file = self._get_result_file(
            run_dir, "decode_token_execution_plus_preemption_time")

        if cached_ttft_file is not None and cached_tbt_file is not None:
            return self._extract_stats(cached_ttft_file,
                                      cached_tbt_file,
                                      benchmark_config)

        command = self._generate_run_command(benchmark_config)

        output_file = open(f"{run_dir}/output.log", "w")

        # write command to a file
        output_file.write(f"Running command: {command}\n")

        args = shlex.split(command)
        p = Popen(args, stdout=output_file, stderr=output_file)
        p.wait()

        ttft_file = self._get_result_file(
            run_dir, "prefill_e2e_time")
        tbt_file = self._get_result_file(
            run_dir, "decode_token_execution_plus_preemption_time")
        assert (
            ttft_file is not None and tbt_file is not None
        ), f"Result file not found for {benchmark_config.to_human_readable_name()}"
        return self._extract_stats(ttft_file, tbt_file, benchmark_config)

    @release_resources_on_failure
    def run(self):
        """
        Run config for all QPS values
        """
        print(
            f"Starting run for {self.job_config.get_human_readable_name()}",
            flush=True,
        )

        tbt_values = []
        ttft_values = []

        for qps in self.args.qps_values:
            ttft, tbt = self._run_job(qps)
            tbt_values.append(tbt)
            ttft_values.append(ttft)

        return {
            **self.job_config.to_config_dict(),
            "qps_values": self.args.qps_values,
            "tbt_values": tbt_values,
            "ttft_values": ttft_values,
        }
