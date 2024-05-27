from sarathi.benchmark.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from sarathi.benchmark.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)
from sarathi.benchmark.types import RequestGeneratorType
from sarathi.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestGeneratorType:
        return RequestGeneratorType.from_str(key_str)


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_REPLAY, TraceReplayRequestGenerator
)
