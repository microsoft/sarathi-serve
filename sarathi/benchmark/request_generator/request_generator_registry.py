from sarathi.benchmark.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from sarathi.benchmark.request_generator.trace_request_generator import (
    TraceRequestGenerator,
)
from sarathi.types import RequestGeneratorType
from sarathi.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
