import glob
import shutil
import json
import os

import pandas as pd
import pytest

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.config import BenchmarkConfig, SyntheticRequestGeneratorConfig, \
    DatasetRequestLengthGeneratorConfig, CorrectnessTestConfig
from sarathi.config.config import ModelConfig, ParallelConfig


# pytest -k "perf_test"
@pytest.mark.parametrize(
    "model, max_model_len, pp_size, tp_size, dataset, request_pattern, baseline_run, test_file",
    [
        ("meta-llama/Meta-Llama-3-8B", 4096, 1, 1, "OpenGVLab/ShareGPT-4o", "uniform", False, None),
    ]
)
def test_correctness(model: str, max_model_len: int, pp_size: int, tp_size: int, dataset: str, request_pattern: str, baseline_run: bool, baseline_file: str):
    # TODO: Test over 3d space
    model_config = ModelConfig(
        model=model,
        max_model_len=max_model_len
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size
    )
    request_generator_config = SyntheticRequestGeneratorConfig(
        length_generator_config=DatasetRequestLengthGeneratorConfig(dataset=dataset)
    )
    correctness_test_config = CorrectnessTestConfig(
        run_correctness_tests=True,
        run_correctness_baseline=baseline_run,
        correctness_test_file=baseline_file
    )
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "benchmark_output")
    benchmark_config = BenchmarkConfig(
        log_level="error",
        output_dir=output_dir,
        model_config=model_config,
        parallel_config=parallel_config,
        request_generator_config=request_generator_config,
        test_config=correctness_test_config,
    )
    BenchmarkRunnerLauncher(benchmark_config).run()

    print("correctness_test finished.")