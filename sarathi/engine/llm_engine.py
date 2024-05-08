from sarathi.engine.arg_utils import EngineArgs
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine


class LLMEngine:

    @classmethod
    def from_engine_args(cls, **kwargs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = EngineArgs(**kwargs).create_engine_configs()
        parallel_config = engine_configs[2]
        if parallel_config.pipeline_parallel_size > 1:
            engine = PipelineParallelLLMEngine(*engine_configs)
        else:
            engine = BaseLLMEngine(*engine_configs)

        return engine
