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
    seed: int = 42


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = 42


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = (
        "data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv"
    )
    start_time: str = "1970-01-04 12:00:00"
    end_time: str = "1970-01-04 15:00:00"
    time_scale_factor: float = 0.3

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = 1.0

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = 1.0
    cv: float = 0.5

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
    trace_file: str = (
        "data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv"
    )
    prefill_scale_factor: float = 1
    decode_scale_factor: float = 1
    max_tokens: int = 4096

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = 0.6
    scramble: bool = False
    min_tokens: int = 1024
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 20.0

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = 1024
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 20.0

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = 4096
    decode_tokens: int = 512

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = 42


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: int = 64
    duration: float = None

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = "data/processed_traces/sydney_enterprise.csv"
    date: str = "2023-08-21"
    prefill_scale_factor: float = 0.3
    decode_scale_factor: float = 1
    time_scale_factor: float = 0.04
    max_tokens: int = 4096

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE


@dataclass
class BenchmarkConfig(BaseEndpointConfig):
    seed: int = 42
    output_dir: str = "benchmark_output"
    write_json_trace: bool = True
    enable_profiling: bool = False
    time_limit: Optional[int] = None
    num_replicas: int = 1
    replica_resource_mapping: Optional[ReplicaResourceMapping] = None
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig
    )

    def __post_init__(self):
        super().__post_init__()

        if not self.time_limit:
            self.time_limit = float("inf")
