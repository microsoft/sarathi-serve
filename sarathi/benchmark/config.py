import datetime
from dataclasses import dataclass, field
from typing import Optional

from sarathi.config import BaseEndpointConfig
from sarathi.config.base_poly_config import BasePolyConfig
from sarathi.config.flat_dataclass import create_flat_dataclass
from sarathi.logger import init_logger
from sarathi.types import (
    ReplicaResourceMapping,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request interval generator."}
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request length generator."}
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace file for request intervals."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00", metadata={"help": "Start time for the trace."}
    )
    end_time: str = field(
        default="1970-01-04 15:00:00", metadata={"help": "End time for the trace."}
    )
    time_scale_factor: float = field(
        default=0.3,
        metadata={"help": "Factor to scale the time intervals in the trace."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0,
        metadata={"help": "Queries per second for the Poisson distribution."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0, metadata={"help": "Queries per second for the Gamma distribution."}
    )
    cv: float = field(
        default=0.5,
        metadata={"help": "Coefficient of variation for the Gamma distribution."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace file for request lengths."},
    )
    prefill_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6, metadata={"help": "Theta parameter for the Zipf distribution."}
    )
    scramble: bool = field(
        default=False, metadata={"help": "Whether to scramble the Zipf distribution."}
    )
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=4096, metadata={"help": "Number of prefill tokens."}
    )
    decode_tokens: int = field(
        default=512, metadata={"help": "Number of decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED



@dataclass
class DatasetRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    dataset: str = field(
        default="ccdv/arxiv-summarization",
        metadata={"help": "Path to the trace file for request lengths."},
    )
    meta_prompt: str = field(
        default=None,
        metadata={"help": "Meta prompt for the dataset."},
    )
    max_prompt_len: int = field(
        default=4096, metadata={"help": "Maximum prompt length allowed."}
    )
    max_num_prompts: int = field(
        default=300, metadata={"help": "Maximum number of prompts to use."}
    )
    max_decode_tokens: int = field(
        default=512, metadata={"help": "Maximum number of decode tokens."}
    )
    tokenizer_model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "Name or path of the huggingface model to use for the tokenizer."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.DATASET


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request generator."}
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: int = field(
        default=64, metadata={"help": "Number of requests to generate."}
    )
    duration: float = field(
        default=None, metadata={"help": "Duration of the synthetic request generation."}
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sydney_enterprise.csv",
        metadata={"help": "Path to the trace file for request generation."},
    )
    date: str = field(
        default="2023-08-21", metadata={"help": "Date for the trace data."}
    )
    prefill_scale_factor: float = field(
        default=0.3, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    time_scale_factor: float = field(
        default=0.04, metadata={"help": "Scale factor for time intervals."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed."}
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE

@dataclass
class CorrectnessTestConfig(BaseTestConfig):
    run_correctness_tests: bool = field(
        default=False, metadata={"help": "Collect correctness data in this run"}
    )
    run_correctness_baseline: bool = field(
        default=False, metadata={"help": "Make this correctness ground truth for correctness tests"}
    )
    correctness_test_file: bool = field(
        default=None, metadata={"help": "Ground truth file for model output. If run_correctness_baseline is True, then the model output will be saved to \
                                this file to be used as a ground truth file. Otherwise, the test will read from this file to be used as ground truth for \
                                the correctness test."}
    )

@dataclass
class BenchmarkConfig(BaseEndpointConfig):
    seed: int = field(default=42, metadata={"help": "Random seed for the benchmark."})
    output_dir: str = field(
        default="benchmark_output",
        metadata={"help": "Directory to store benchmark output."},
    )
    write_json_trace: bool = field(
        default=True, metadata={"help": "Whether to write JSON trace output."}
    )
    enable_profiling: bool = field(
        default=False, metadata={"help": "Whether to enable profiling."}
    )
    time_limit: Optional[int] = field(
        default=None, metadata={"help": "Time limit for the benchmark in seconds."}
    )
    num_replicas: int = field(
        default=1, metadata={"help": "Number of replicas to use."}
    )
    replica_resource_mapping: Optional[ReplicaResourceMapping] = field(
        default=None, metadata={"help": "Mapping of replicas to resources."}
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig
    )
    correctness_test_config: Optional[CorrectnessTestConfig] = field(
        default_factory=CorrectnessTestConfig
    )

    def __post_init__(self):
        super().__post_init__()

        if not self.time_limit:
            self.time_limit = float("inf")
