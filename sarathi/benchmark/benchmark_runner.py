import logging
import os
import time

import ray
import wandb
from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import ReplicaConfig
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.types import ReplicaResourceMapping, ResourceMapping
from sarathi.utils import get_ip

logger = logging.getLogger(__name__)


class BenchmarkRunner:

    def __init__(
        self,
        replica_id: int,
        config: BenchmarkConfig,
        resource_mapping: ResourceMapping,
    ) -> None:
        self.replica_id = replica_id
        self.config = config

        replica_config = ReplicaConfig(
            replica_id,
            self.config.output_dir,
            resource_mapping,
        )
        os.makedirs(replica_config.output_dir, exist_ok=True)

        set_seeds(self.config.seed)
        request_generator = RequestGeneratorRegistry.get(
            self.config.request_generator_config.get_type(),
            self.config.request_generator_config,
        )
        self.requests = request_generator.generate()

        # select every nth request for this replica
        # e.g. if there are 4 replicas, and this is the 2nd replica, then
        # we will select the 2nd, 6th, 10th, ... requests
        # round robin scheduling
        self.requests = self.requests[self.replica_id :: self.config.num_replicas]

        if self.config.num_replicas > 1:
            # disable per-replica wandb logging for multi-replica runs
            # so that we can aggregate metrics across all replicas
            self.config.metrics_config.wandb_project = None

        system_config = self.config.create_system_config(replica_config)
        self.llm_engine = LLMEngine.from_system_config(system_config)

        if wandb.run is not None:
            wandb.config.update(self.config.to_dict())

    def _get_input_params(
        self, request: Request, first_request_time: float
    ) -> SamplingParams:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,
            temperature=0,
            top_p=1.0,
        )
        prompt_token_ids = [1] * request.num_prefill_tokens

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "arrival_time": first_request_time + request.arrived_at,
        }

    def warmup(self) -> None:
        self.llm_engine.add_request(**self._get_input_params(self.requests[0], 0))

        is_completed = False
        while not is_completed:
            step_outputs = self.llm_engine.step()
            is_completed = step_outputs[0].finished

        self.llm_engine.reset_metrics()

    def _run(self) -> None:
        if self.config.enable_profiling:
            self.llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=len(self.requests),
            desc=f"Replica {self.replica_id} processed requests",
        )
        start_time = time.monotonic()

        # Run the engine.
        while num_processed_requests < len(self.requests):
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > self.config.time_limit:
                break

            step_outputs = self.llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)

        end_time = time.monotonic()
        pbar.close()

        logger.info(
            f"Replica {self.replica_id} exiting after processing {len(self.requests)} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )

        if self.config.enable_profiling:
            self.llm_engine.stop_profiling()

    def _add_requests(self) -> None:
        index = 0
        first_request_time = time.monotonic()
        while index < len(self.requests):
            request = self.requests[index]
            self.llm_engine.add_request(
                **self._get_input_params(request, first_request_time)
            )
            index += 1

    def run(self) -> None:
        self.llm_engine.reset_metrics()
        self._add_requests()
        self._run()
        self.llm_engine.pull_worker_metrics()
        metric_store = self.llm_engine.get_metric_store()
        return metric_store


class BenchmarkRunnerLauncher:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.is_multi_replica = self.config.num_replicas > 1

        ray.init(ignore_reinit_error=True)

        self._validate_cluster_resources()
        self.runners = self._create_runners()

        if self.is_multi_replica:
            self.aggregate_metric_store = self._create_aggregate_metric_store()

    def _validate_cluster_resources(self):
        num_replicas = self.config.num_replicas
        num_gpus_required = num_replicas * self.config.parallel_config.world_size

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
        if self.config.replica_resource_mapping:
            assert len(self.config.replica_resource_mapping) == self.config.num_replicas
            logger.info(
                f"Replica resource mapping: {self.config.replica_resource_mapping}"
            )
            return self.config.replica_resource_mapping

        cluster_resources_keys = list(ray.available_resources().keys())
        num_gpus = ray.available_resources()["GPU"]
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]

        runner_ip = f"node:{get_ip()}"

        ip_addresses.remove(runner_ip)
        ip_addresses.insert(0, runner_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        assert (
            num_gpus % num_nodes == 0
        ), f"Number of GPUs ({num_gpus}) is not a multiple of number of nodes ({num_nodes})"
        num_gpus_per_node = int(num_gpus // num_nodes)
        num_replicas = self.config.num_replicas
        num_gpus_per_replica = self.config.parallel_config.world_size

        assert (
            num_gpus >= num_replicas * num_gpus_per_replica
        ), f"Insufficient GPUs. Required: {num_replicas * num_gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = []

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in reversed(range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))

        for _ in range(num_replicas):
            resource_mapping = []
            for _ in range(num_gpus_per_replica):
                resource_mapping.append(available_gpus.pop(0))
            replica_resource_mapping.append(resource_mapping)

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping

    def _create_runners(self):
        replica_resource_mapping = self._get_replica_resource_mapping()

        if not self.is_multi_replica:
            return [BenchmarkRunner(0, self.config, replica_resource_mapping[0])]

        runner_class = ray.remote(num_cpus=1)(BenchmarkRunner)

        runners = []

        for replica_id in range(self.config.num_replicas):
            runners.append(
                runner_class.options(
                    resources={
                        replica_resource_mapping[replica_id][0][0]: 0.01,
                    },
                ).remote(replica_id, self.config, replica_resource_mapping[replica_id])
            )

        return runners

    def _create_aggregate_metric_store(self):
        replica_config = ReplicaConfig(
            replica_id=0,  # dummy replica id
            output_dir=self.config.output_dir,
        )
        metrics_store = MetricsStore.get_instance(
            replica_config,
            self.config.model_config,
            self.config.metrics_config,
        )

        if wandb.run is not None:
            wandb.config.update(self.config.to_dict())

        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store

    def run(self):
        if self.is_multi_replica:
            ray.get([runner.warmup.remote() for runner in self.runners])

            runner_metrics = ray.get([runner.run.remote() for runner in self.runners])

            for runner_metric in runner_metrics:
                self.aggregate_metric_store.merge(runner_metric)

            if wandb.run is not None:
                wandb.config.update(self.config.__dict__)

            self.aggregate_metric_store.plot()
        else:
            metric_store = self.runners[0].run()
            metric_store.plot()

        wandb.finish()
