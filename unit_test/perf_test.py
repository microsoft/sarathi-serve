import glob
import shutil
import json
import os

import pandas as pd
import pytest

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.config import BenchmarkConfig, SyntheticRequestGeneratorConfig, \
    DatasetRequestLengthGeneratorConfig
from sarathi.config.config import ModelConfig, ParallelConfig


# pytest -k "perf_test"
@pytest.mark.parametrize(
    "model, max_model_len, pp_size, tp_size, dataset, request_pattern",
    [
        ("meta-llama/Meta-Llama-3-8B", 4096, 1, 1, "OpenGVLab/ShareGPT-4o", "uniform"),
    ]
)
def test_perf(model: str, max_model_len: int, pp_size: int, tp_size: int, dataset: str, request_pattern: str):
    # TODO: Test over 3d space
    model_config = ModelConfig(
        model=model,
        max_model_len=max_model_len)
    parallel_config = ParallelConfig(
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size)
    request_generator_config = SyntheticRequestGeneratorConfig(
        length_generator_config=DatasetRequestLengthGeneratorConfig(dataset=dataset)
    )
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "benchmark_output")
    benchmark_config = BenchmarkConfig(
        log_level="error",
        output_dir=output_dir,
        model_config=model_config,
        parallel_config=parallel_config,
        request_generator_config=request_generator_config
    )
    BenchmarkRunnerLauncher(benchmark_config).run()

    key = model + "_" + "pp" + str(pp_size) + "_" + "tp" + str(tp_size) + "_" + dataset + "_" + request_pattern
    # TODO: convert to build key for json method

    perf_json_path = os.path.join(cwd, "perf_test.json")
    _build_perf_data(key, output_dir, perf_json_path)

    # clean up benchmark runner output now
    _delete_directory(output_dir)


def _build_perf_data(key: str, output_dir: str, perf_json_path: str):
    p50_quantile = .50
    p90_quantile = .90
    perf_data = {}
    ttft_file = _get_result_file(
        output_dir, "prefill_e2e_time"
    )
    tbt_file = _get_result_file(
        output_dir, "decode_token_execution_plus_preemption_time"
    )

    ttft_delay_df = pd.read_csv(ttft_file)
    ttft_p50 = ttft_delay_df["request_scheduling_delay"].quantile(
         p50_quantile
    )
    perf_data["ttft_p50"] = ttft_p50
    ttft_p90 = ttft_delay_df["request_scheduling_delay"].quantile(
        p90_quantile
    )
    perf_data["ttft_p90"] = ttft_p90

    tbt_df = pd.read_csv(tbt_file)
    tbt_p50 = tbt_df["decode_token_execution_plus_preemption_time"].quantile(
        p50_quantile
    )
    perf_data["tbt_p50"] = tbt_p50
    tbt_p90 = tbt_df["decode_token_execution_plus_preemption_time"].quantile(
        p90_quantile
    )
    perf_data["tbt_p90"] = tbt_p90

    _write_or_append_json(key, perf_data, filename=perf_json_path)


def _get_result_file(run_dir: str, metric_name: str) -> str:
    result_file = glob.glob(f"{run_dir}/*/*/plots/{metric_name}.csv")
    assert len(result_file) > 0, f"No benchmark results found for path {run_dir}"

    return result_file[0]


def _write_or_append_json(key, data, filename):
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    existing_data[key] = data

    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)


def _delete_directory(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)

