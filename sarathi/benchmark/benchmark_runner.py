import json
import logging
import os
import time

import ray
import wandb
from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import Config
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.types import ReplicaResourceMapping, ResourceMapping
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import MetricsConfig
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.utils import get_ip

logger = logging.getLogger(__name__)


class BenchmarkRunner:

    def __init__(
        self,
        replica_id: int,
        config: Config,
        replica_resource_mapping: ResourceMapping = [],
    ) -> None:
        self._replica_id = replica_id
        self._config = config
        self._num_replicas = self._config.cluster_num_replicas

        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        output_dir = f"{self._config.output_dir}/replica_{replica_id}"
        os.makedirs(output_dir, exist_ok=True)

        set_seeds(config.seed)
        request_generator = RequestGeneratorRegistry.get_from_str(
            self._config.request_generator_provider, self._config
        )
        self._requests = request_generator.generate()

        # select every nth request for this replica
        # e.g. if there are 4 replicas, and this is the 2nd replica, then
        # we will select the 2nd, 6th, 10th, ... requests
        # round robin scheduling
        self._requests = self._requests[self._replica_id :: self._num_replicas]

        if self._num_replicas == 1:
            wandb_project = self._config.metrics_store_wandb_project
            wandb_group = self._config.metrics_store_wandb_group
            wandb_run_name = self._config.metrics_store_wandb_run_name
        else:
            wandb_project = None
            wandb_group = None
            wandb_run_name = None

        chunk_size = None
        if self._config.replica_scheduler_provider == "sarathi":
            chunk_size = self._config.sarathi_scheduler_chunk_size
        elif self._config.replica_scheduler_provider == "simple_chunking":
            chunk_size = self._config.simple_chunking_scheduler_chunk_size

        self._llm_engine = LLMEngine.from_engine_args(
            # replica config
            replica_id=replica_id,
            replica_resource_mapping=replica_resource_mapping,
            output_dir=output_dir,
            # model config
            model=self._config.model_name,
            tokenizer=self._config.model_name,
            tensor_parallel_size=self._config.model_tensor_parallel_degree,
            pipeline_parallel_size=self._config.model_pipeline_parallel_degree,
            attention_backend=self._config.model_attention_backend,
            seed=self._config.seed,
            dtype="float16",
            load_format=self._config.model_load_format,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
            max_model_len=self._config.model_max_model_len,
            # scheduler config
            scheduler_type=self._config.replica_scheduler_provider,
            max_num_seqs=self._config.replica_scheduler_max_batch_size,
            # sarathi scheduler config
            chunk_size=chunk_size,
            enable_dynamic_chunking_schedule=self._config.sarathi_scheduler_enable_dynamic_chunking_schedule,
            low_chunk_size=self._config.sarathi_scheduler_low_chunk_size,
            high_chunk_size=self._config.sarathi_scheduler_high_chunk_size,
            chunk_schedule_max_tokens=self._config.sarathi_scheduler_chunk_schedule_max_tokens,
            chunk_schedule_stages=self._config.sarathi_scheduler_chunk_schedule_stages,
            # vllm scheduler config
            max_num_batched_tokens=self._config.vllm_scheduler_max_tokens_in_batch,
            # wandb config
            write_metrics=self._config.write_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_run_name=wandb_run_name,
            wandb_sweep_id=self._config.metrics_store_wandb_sweep_id,
            wandb_run_id=self._config.metrics_store_wandb_run_id,
            # metrics config
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=self._config.metrics_store_keep_individual_batch_metrics,
            # engine config
            trust_remote_code=True,
        )

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
        # warmup the engine
        self._llm_engine.add_request(
            **self._get_input_params(self._requests[0], time.monotonic())
        )

        is_completed = False
        while not is_completed:
            step_outputs = self._llm_engine.step()
            is_completed = step_outputs[0].finished

        self._llm_engine.reset_metrics()

    def _run(self) -> None:
        if self._config.enable_profiling:
            self._llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=len(self._requests),
            desc=f"Replica {self._replica_id} processed requests",
        )
        start_time = time.monotonic()

        # Run the engine.
        while num_processed_requests < len(self._requests):
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > self._time_limit:
                break

            step_outputs = self._llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)
        end_time = time.monotonic()
        pbar.close()

        logger.info(
            f"Replica {self._replica_id} exiting after processing {len(self._requests)} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )

        if self._config.enable_profiling:
            self._llm_engine.stop_profiling()

    def _add_requests(self) -> None:
        index = 0
        first_request_time = time.monotonic()
        while index < len(self._requests):
            request = self._requests[index]
            self._llm_engine.add_request(
                **self._get_input_params(request, first_request_time)
            )
            index += 1

    def run(self) -> None:
        self._llm_engine.reset_metrics()
        self._add_requests()
        self._run()
        self._llm_engine.pull_worker_metrics()
        metric_store = self._llm_engine.get_metric_store()
        return metric_store


class BenchmarkRunnerLauncher:

    def __init__(self, config: Config) -> None:
        self._config = config
        self._is_multi_replica = self._config.cluster_num_replicas > 1

        ray.init(ignore_reinit_error=True)

        if self._is_multi_replica:
            self._validate_cluster_resources()
            self._runners = self._create_runners()
            self._aggregate_metric_store = self._create_aggregate_metric_store()
        else:
            replica_resource_mapping = self._get_replica_resource_mapping()
            assert len(replica_resource_mapping) == 1
            self._runner = BenchmarkRunner(
                0, self._config, replica_resource_mapping["0"]
            )

        if wandb.run is not None:
            wandb.config.update(self._config.__dict__)

    def _validate_cluster_resources(self):
        num_replicas = self._config.cluster_num_replicas
        tp_degree = self._config.model_tensor_parallel_degree
        pp_degree = self._config.model_pipeline_parallel_degree
        num_gpus_required = num_replicas * tp_degree * pp_degree

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
        if self._config.replica_resource_mapping:
            replica_resource_mapping = json.loads(self._config.replica_resource_mapping)
            logger.info(f"Replica resource mapping: {replica_resource_mapping}")
            return replica_resource_mapping

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
        num_replicas = self._config.cluster_num_replicas
        num_gpus_per_replica = (
            self._config.model_tensor_parallel_degree
            * self._config.model_pipeline_parallel_degree
        )

        assert (
            num_gpus >= num_replicas * num_gpus_per_replica
        ), f"Insufficient GPUs. Required: {num_replicas * num_gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = {}

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in reversed(range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))

        for replica_id in range(num_replicas):
            replica_resource_mapping[str(replica_id)] = []
            for _ in range(num_gpus_per_replica):
                replica_resource_mapping[str(replica_id)].append(available_gpus.pop(0))

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping

    def _create_runners(self):
        assert (
            self._config.model_tensor_parallel_degree > 1
            or self._config.model_pipeline_parallel_degree > 1
        )

        replica_resource_mapping = self._get_replica_resource_mapping()

        runner_class = ray.remote(num_cpus=1)(BenchmarkRunner)

        runners = []

        for replica_id in range(self._config.cluster_num_replicas):
            runners.append(
                runner_class.options(
                    resources={
                        replica_resource_mapping[str(replica_id)][0][0]: 0.01,
                    },
                ).remote(
                    replica_id, self._config, replica_resource_mapping[str(replica_id)]
                )
            )

        return runners

    def _create_aggregate_metric_store(self):
        metric_config = MetricsConfig(
            replica_id=0,  # dummy replica id
            write_metrics=self._config.write_metrics,
            output_dir=self._config.output_dir,
            wandb_project=self._config.metrics_store_wandb_project,
            wandb_group=self._config.metrics_store_wandb_group,
            wandb_run_name=self._config.metrics_store_wandb_run_name,
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=self._config.metrics_store_keep_individual_batch_metrics,
        )
        metrics_store = MetricsStore(metric_config)
        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store

    def run(self):
        if self._is_multi_replica:
            ray.get([runner.warmup.remote() for runner in self._runners])

            runner_metrics = ray.get([runner.run.remote() for runner in self._runners])

            for runner_metric in runner_metrics:
                self._aggregate_metric_store.merge(runner_metric)

            if wandb.run is not None:
                wandb.config.update(self._config.__dict__)

            self._aggregate_metric_store.plot()
        else:
            metric_store = self._runner.run()
            metric_store.plot()

        wandb.finish()
