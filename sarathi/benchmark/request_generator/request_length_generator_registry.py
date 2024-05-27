from sarathi.benchmark.request_generator.fixed_request_length_generator import (
    FixedRequestLengthGenerator,
)
from sarathi.benchmark.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from sarathi.benchmark.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from sarathi.benchmark.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)
from sarathi.benchmark.types import RequestLengthGeneratorType
from sarathi.utils.base_registry import BaseRegistry


class RequestLengthGeneratorRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.from_str(key_str)


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.FIXED, FixedRequestLengthGenerator
)
