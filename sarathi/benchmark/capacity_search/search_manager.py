import argparse

import ray

from sarathi.benchmark.capacity_search.capacity_search import CapacitySearch
from sarathi.benchmark.capacity_search.config import JobConfig
from sarathi.benchmark.capacity_search.ray_utils import (
    RayParallelRunner,
    ResourceManager,
)
from sarathi.logger import init_logger
from sarathi.types import ReplicaResourceMapping

logger = init_logger(__name__)


def run_search(
    job_config: JobConfig,
    args: argparse.Namespace,
    resource_manager: ResourceManager,
    replica_resource_mapping: ReplicaResourceMapping,
):
    capacity_search = CapacitySearch(
        job_config,
        args,
        resource_manager,
        replica_resource_mapping,
    )
    return capacity_search.search()


class SearchManager:

    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
    ):
        self.args = args
        self.config = config

        ray.init(ignore_reinit_error=True)

    def run(self):
        job_configs = JobConfig.generate_job_configs(self.config)

        for job_config in job_configs:
            logger.info(f"Running search for {job_config}")

        ray_parallel_runner = RayParallelRunner()

        remote_func = (
            lambda resource_manager, replica_resource_mapping, job_config: run_search(
                job_config,
                self.args,
                resource_manager,
                replica_resource_mapping,
            )
        )
        all_results = ray_parallel_runner.map(
            remote_func,
            job_configs,
        )
        return all_results
