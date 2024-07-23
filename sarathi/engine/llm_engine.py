from sarathi.config import SystemConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if config.parallel_config.pipeline_parallel_size > 1:
            engine = PipelineParallelLLMEngine(config)
        else:
            engine = BaseLLMEngine(config)

        return engine
