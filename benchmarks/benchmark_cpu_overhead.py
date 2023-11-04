import time
import logging
import datetime
import os
import gc
import binascii
from itertools import product

import ray

import pandas as pd
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.metrics.constants import CpuOperationMetrics

logger = logging.getLogger(__name__)

PREFILL_SIZE = 1024
NUM_DECODES = 50

#MODELS = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-70b-hf", "codellama/CodeLlama-34b-Instruct-hf", "tiiuae/falcon-7b", "tiiuae/falcon-40b"]
MODELS = [
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-34b-Instruct-hf"
]
# MODELS = ["codellama/CodeLlama-34b-Instruct-hf"]
BATCH_SIZES = list(range(8, 64 + 1, 8))
TENSOR_PARALLEL_DEGREES = [1, 2, 4, 8]

OUTPUT_DIR = "cpu_overhead_benchmark_results"


def hex_to_binary(hex_identifier):
    return binascii.unhexlify(hex_identifier)


class BenchmarkRunner:

    def __init__(self, model_name, batch_size, tensor_parallel_degree,
                 output_dir) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._tensor_parallel_degree = tensor_parallel_degree
        self._output_dir = output_dir

        self._config_name = f"{model_name}_{batch_size}_{tensor_parallel_degree}"

        self._llm = LLM(
            replica_id=0,
            # model config
            model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=tensor_parallel_degree,
            max_num_batched_tokens=4096,
            dtype="float16",
            # scheduler config
            scheduler_type="vllm",
            max_num_seqs=batch_size,
            write_metrics=True,
            enable_cpu_op_level_metrics=True,
            output_dir=output_dir,
            skip_hidden_layers=True,
        )

    def _get_input_params(self) -> SamplingParams:
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=NUM_DECODES,
        )
        prompt_token_ids = [1] * PREFILL_SIZE

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
        }

    def run(self):
        self._warmup()

        num_requests = self._batch_size * 5

        for _ in range(num_requests):
            self._llm._add_request(**self._get_input_params())

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(total=num_requests,
                    desc=f"{self._config_name} processed requests")

        self._llm.reset_metrics()

        start_time = time.time()

        # Run the engine.
        while num_processed_requests < num_requests:
            step_outputs = self._llm.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)

        end_time = time.time()
        pbar.close()

        logger.info(
            f"{self._config_name} exiting after processing {num_requests} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )

        self._llm.pull_worker_metrics()

        metric_store = self._llm.get_metric_store()

        metrics_means = {
            f"{k.name}_MEAN": v.mean
            for k, v in metric_store._cpu_operation_metrics.items()
        }

        metrics_medians = {
            f"{k.name}_MEDIAN": v.median
            for k, v in metric_store._cpu_operation_metrics.items()
        }

        metrics = {**metrics_means, **metrics_medians}

        total_recorded_cpu_time = \
            (
                metric_store._cpu_operation_metrics[CpuOperationMetrics.SCHEDULE].sum +
                metric_store._cpu_operation_metrics[CpuOperationMetrics.PROCESS_MODEL_OUTPUTS].sum
            ) + (
                metric_store._cpu_operation_metrics[CpuOperationMetrics.SAMPLER_E2E].sum +
                metric_store._cpu_operation_metrics[CpuOperationMetrics.PREPARE_INPUTS_E2E].sum +
                metric_store._cpu_operation_metrics[CpuOperationMetrics.MODEL_EXECUTION_E2E].sum +
                metric_store._cpu_operation_metrics[CpuOperationMetrics.POST_PREPARE_INPUTS_BARRIER].sum
            ) / self._tensor_parallel_degree

        total_recorded_cpu_time *= 1e-3  # convert to seconds
        ray_comm_time = (
            (end_time - start_time) - total_recorded_cpu_time) / num_steps
        ray_comm_time *= 1e3  # convert to ms

        metrics.update({
            "model_name": self._model_name,
            "batch_size": self._batch_size,
            "tensor_parallel_degree": self._tensor_parallel_degree,
            "total_time": end_time - start_time,
            "num_steps": num_steps,
            "RAY_COMM_TIME": ray_comm_time,
        })

        del self._llm
        # trigger garbage collection
        gc.collect()

        return metrics

    def _warmup(self) -> None:
        # warmup the engine
        self._llm._add_request(**self._get_input_params())

        is_completed = False
        while not is_completed:
            step_outputs = self._llm.step()
            is_completed = step_outputs[0].finished

        self._llm.reset_metrics()


class BenchmarkRunnerLauncher:

    def _create_runner(self, model_name, batch_size, tensor_parallel_degree,
                       output_dir) -> BenchmarkRunner:
        placement_group_ids = list(ray.util.placement_group_table().keys())
        for placement_group_id in placement_group_ids:
            ray._private.worker.global_worker.core_worker.remove_placement_group(
                ray.PlacementGroupID(hex_to_binary(placement_group_id)))

        print(ray.available_resources())

        num_gpus = 0
        if tensor_parallel_degree == 1:
            num_gpus = 1

        runner_class = ray.remote(num_gpus=num_gpus)(BenchmarkRunner).remote

        return runner_class(model_name, batch_size, tensor_parallel_degree,
                            output_dir)

    def run(self):
        results = []

        output_dir = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(output_dir, exist_ok=True)

        params = product(MODELS, TENSOR_PARALLEL_DEGREES)
        for model_name, tensor_parallel_degree in params:
            if model_name == "meta-llama/Llama-2-70b-hf" and tensor_parallel_degree < 4:
                continue

            for batch_size in BATCH_SIZES:
                try:
                    runner = self._create_runner(model_name, batch_size,
                                                 tensor_parallel_degree,
                                                 output_dir)
                    results.append(ray.get(runner.run.remote()))
                    del runner
                    # trigger garbage collection
                    gc.collect()
                except Exception as e:
                    logger.error(
                        f"Failed to run {model_name}_{batch_size}_{tensor_parallel_degree} due to {e}"
                    )
                    break

        df = pd.DataFrame(results)
        # write results to a csv file
        df.to_csv(f"{output_dir}/results.csv")


if __name__ == "__main__":
    launcher = BenchmarkRunnerLauncher()
    launcher.run()
