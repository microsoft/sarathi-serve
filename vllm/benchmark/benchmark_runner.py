import time
import logging
import threading
import os

import ray
from tqdm import tqdm

from vllm.benchmark.config import Config
from vllm.benchmark.request_generator import RequestGeneratorRegistry
from vllm.benchmark.entities import Request
from vllm import LLM, SamplingParams
from vllm.config import MetricsConfig
from vllm.metrics.metrics_store import MetricsStore
from vllm.benchmark.utils.random import set_seeds

logger = logging.getLogger(__name__)


class BenchmarkRunner:

    def __init__(self, replica_id: int, config: Config) -> None:
        self._replica_id = replica_id
        self._config = config

        self._num_replicas = self._config.cluster_num_replicas

        output_dir=f"{self._config.output_dir}/replica_{replica_id}"
        os.makedirs(output_dir, exist_ok=True)

        set_seeds(config.seed)
        request_generator = RequestGeneratorRegistry.get_from_str(
            self._config.request_generator_provider, self._config)
        self._requests = request_generator.generate()

        # select every nth request for this replica
        # e.g. if there are 4 replicas, and this is the 2nd replica, then
        # we will select the 2nd, 6th, 10th, ... requests
        # round robin scheduling
        self._requests = self._requests[self._replica_id::self._num_replicas]

        self._llm = LLM(
            # replica config
            replica_id=replica_id,
            output_dir=output_dir,
            # model config
            model=self._config.model_name,
            tokenizer=self._config.model_tokenizer,
            tensor_parallel_size=self._config.model_tensor_parallel_degree,
            seed=self._config.seed,
            dtype="float16",
            # scheduler config
            scheduler_type=self._config.replica_scheduler_provider,
            max_num_seqs=self._config.replica_scheduler_max_batch_size,
            # sarathi scheduler config
            chunk_size=self._config.sarathi_scheduler_chunk_size,
            enable_rolling_prefills=self._config.
            sarathi_scheduler_enable_rolling_prefills,
            prefill_fitting_tolerance=self._config.
            sarathi_scheduler_prefill_fitting_tolerance,
            # vllm scheduler config
            max_num_batched_tokens=self._config.
            vllm_scheduler_max_tokens_in_batch,
            # wandb config
            write_metrics=self._config.write_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            wandb_project=None,
            wandb_group=None,
            wandb_run_name=None,
            subsamples=self._config.metrics_store_subsamples,
            save_table_to_wandb=self._config.metrics_store_save_table_to_wandb,
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_high_level_cuda_metrics=self._config.metrics_store_enable_high_level_cuda_metrics,
            # vllm engine config
            disable_log_stats=True,
            trust_remote_code=True,
        )

    def _get_input_params(self, request: Request) -> SamplingParams:
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,
        )
        prompt_token_ids = [1] * request.num_prefill_tokens

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
        }

    def _run_replica_thread(self) -> None:
        if self._config.enable_profiling:
            self._llm.llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(total=len(self._requests), desc=f"Replica {self._replica_id} processed requests")
        start_time = time.time()

        # Run the engine.
        while num_processed_requests < len(self._requests):
            step_outputs = self._llm.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)

        end_time = time.time()
        pbar.close()

        logger.info(f"Replica {self._replica_id} exiting after processing {len(self._requests)} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds")

        if self._config.enable_profiling:
            self._llm.stop_profiling()

    def warmup(self) -> None:
        # warmup the engine
        self._llm._add_request(**self._get_input_params(self._requests[0]))

        is_completed = False
        while not is_completed:
            step_outputs = self._llm.step()
            is_completed = step_outputs[0].finished

        self._llm.reset_metrics()

    def run(self) -> None:
        self._llm.reset_metrics()

        # launch the engine in a separate background thread
        # Create a thread and target the function you want to run in the background
        thread = threading.Thread(target=self._run_replica_thread)
        # Start the thread
        thread.start()

        last_request_arrival_time = 0

        for i, request in enumerate(self._requests):
            time.sleep(request.arrived_at - last_request_arrival_time)

            self._llm._add_request(**self._get_input_params(request))
            last_request_arrival_time = request.arrived_at

        thread.join()

        self._llm.pull_worker_metrics()

        return self._llm.get_metric_store()


class BenchmarkRunnerLauncher:
    def __init__(self, config: Config) -> None:
        self._config = config
        
        ray.init()

        self._validate_cluster_resources()
        self._runners = self._create_runners()
        self._aggregate_metric_store = self._create_aggregate_metric_store()

    def _validate_cluster_resources(self):
        num_replicas = self._config.cluster_num_replicas
        tp_degree = self._config.model_tensor_parallel_degree
        num_gpus_required = num_replicas * tp_degree

        available_resources = ray.available_resources()

        assert available_resources["GPU"] >= num_gpus_required, \
            f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _create_runners(self):
        # compute the number of cpu cores for each runner
        # we want to split all cpus evenly among all runners
        # so that they get allocated to different nodes
        total_num_cpus = ray.available_resources()["CPU"]
        num_cpus = total_num_cpus // self._config.cluster_num_replicas

        num_gpus = 0
        if self._config.model_tensor_parallel_degree == 1:
            num_gpus = 1

        runner_class = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(BenchmarkRunner).remote

        runners = []

        for replica_id in range(self._config.cluster_num_replicas):
            runners.append(runner_class(replica_id, self._config))

        return runners

    def _create_aggregate_metric_store(self):
        metric_config = MetricsConfig(
            write_metrics=self._config.write_metrics,
            output_dir=self._config.output_dir,
            wandb_project=self._config.metrics_store_wandb_project,
            wandb_group=self._config.metrics_store_wandb_group,
            wandb_run_name=self._config.metrics_store_wandb_run_name,
            subsamples=self._config.metrics_store_subsamples,
            save_table_to_wandb=self._config.metrics_store_save_table_to_wandb,
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_high_level_cuda_metrics=self._config.metrics_store_enable_high_level_cuda_metrics,
            tensor_parallel_size=self._config.model_tensor_parallel_degree,
            model_num_layers=self._config.model_num_layers,
        )

        return MetricsStore(metric_config)

    def run(self):
        ray.get([runner.warmup.remote() for runner in self._runners])

        runner_metrics = ray.get([runner.run.remote() for runner in self._runners])
        
        for runner_metric in runner_metrics:
            self._aggregate_metric_store.merge(runner_metric)

        self._aggregate_metric_store.plot()
