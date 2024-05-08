from typing import Optional, List, Tuple
from abc import ABC

import torch
from transformers import PretrainedConfig

from sarathi.logger import init_logger
from sarathi.transformers_utils.config import get_config
from sarathi.utils.base_int_enum import BaseIntEnum

logger = init_logger(__name__)


class SchedulerType(BaseIntEnum):
    VLLM = 1
    ORCA = 2
    FASTER_TRANSFORMER = 3
    SARATHI = 4
    SIMPLE_CHUNKING = 5


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.        
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: Optional[str],
        load_format: str,
        dtype: str,
        seed: int,
        revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        attention_backend: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.attention_backend = attention_backend

        self.hf_config = get_config(model, trust_remote_code, revision)

        # support fschat to load model which uses dynamic ntk (e.g Qwen)
        use_dynamic_ntk = getattr(self.hf_config, "use_dynamic_ntk", None)
        if use_dynamic_ntk is not None:
            self.hf_config.max_sequence_length = 16384

        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.hf_config.dtype = self.dtype
        self.max_model_len = _get_and_verify_max_len(self.hf_config,
                                                     max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in [
                "auto", "pt", "safetensors", "npcache", "dummy"
        ]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")
        self.load_format = load_format

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'.")
        self.tokenizer_mode = tokenizer_mode

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size}).")

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            return 1
        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return (self.hf_config.n_head_kv //
                    parallel_config.tensor_parallel_size)
        # For Falcon-40b/Falcon-180b:
        if getattr(self.hf_config, "num_kv_heads", None) is not None:
            return (self.hf_config.num_kv_heads //
                    parallel_config.tensor_parallel_size)
        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (self.hf_config.num_key_value_heads //
                    parallel_config.tensor_parallel_size)
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_q_heads(self, parallel_config: "ParallelConfig") -> int:
        if getattr(self.hf_config, "num_attention_heads", None) is not None:
            return (self.hf_config.num_attention_heads //
                    parallel_config.tensor_parallel_size)
        raise ValueError(
            "num_attention_heads is not defined in the model config")

    def get_max_model_len(self) -> int:
        return self.max_model_len

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_total_num_layers(self) -> int:
        return self.hf_config.num_hidden_layers

class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            Sarathi execution.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._verify_args()

        # Will be set after profiling.
        self.num_gpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        replica_resource_mapping: List[Tuple[str, int]] = [],
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size

        if not replica_resource_mapping:
            replica_resource_mapping = [
                (None, i)
                for i in range(pipeline_parallel_size * tensor_parallel_size)
            ]

        self.replica_resource_mapping = replica_resource_mapping

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self._verify_args()

    def _verify_args(self) -> None:
        pass


class BaseSchedulerConfig(ABC):
    """BaseScheduler configuration.

    Args:
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration. Aka batch size.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.num_pipeline_stages = num_pipeline_stages

    @property
    def max_num_batched_tokens(self):
        pass

    @property
    def type(self):
        pass


class VLLMSchedulerConfig(BaseSchedulerConfig):
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
            This only takes into account number of tokens
            moving from WAITING to RUNNING states.
    """

    def __init__(self, max_num_seqs: int, max_model_len: int,
                 num_pipeline_stages: int,
                 max_num_batched_tokens: int) -> None:
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self._max_num_batched_tokens = (max_num_batched_tokens
                                        if max_num_batched_tokens else
                                        max_model_len)
        # Requests with context length upto max_model_len must be schedulable.
        assert max_model_len <= self._max_num_batched_tokens

    @property
    def max_num_batched_tokens(self):
        return self._max_num_batched_tokens

    @property
    def type(self):
        return SchedulerType.VLLM


class SimpleChunkingSchedulerConfig(BaseSchedulerConfig):

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
        chunk_size: Optional[int],
    ) -> None:
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self.chunk_size = chunk_size

    @property
    def max_num_batched_tokens(self):
        return self.chunk_size

    @property
    def type(self):
        return SchedulerType.SIMPLE_CHUNKING


class OrcaSchedulerConfig(BaseSchedulerConfig):

    @property
    def max_num_batched_tokens(self):
        return self.max_num_seqs * self.max_model_len

    @property
    def type(self):
        return SchedulerType.ORCA


class FasterTransformerSchedulerConfig(BaseSchedulerConfig):

    @property
    def max_num_batched_tokens(self):
        return self.max_num_seqs * self.max_model_len

    @property
    def type(self):
        return SchedulerType.FASTER_TRANSFORMER


class SarathiSchedulerConfig(BaseSchedulerConfig):

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
        chunk_size: Optional[int],
        enable_dynamic_chunking_schedule: bool,
        low_chunk_size: Optional[int],
        high_chunk_size: Optional[int],
        chunk_schedule_max_tokens: Optional[int],
        chunk_schedule_stages: Optional[int],
    ) -> None:
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self.chunk_size = chunk_size
        self.enable_dynamic_chunking_schedule = enable_dynamic_chunking_schedule
        self.low_chunk_size = low_chunk_size
        self.high_chunk_size = high_chunk_size
        self.chunk_schedule_max_tokens = chunk_schedule_max_tokens
        self.chunk_schedule_stages = chunk_schedule_stages

    @property
    def max_num_batched_tokens(self):
        # Sarathi never schedules more than chunk_size tokens in one iteration.
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size
        else:
            return self.chunk_size

    @property
    def type(self):
        return SchedulerType.SARATHI


class MetricsConfig:
    """Metric configuration."""

    def __init__(self, replica_id: int, write_metrics: bool, output_dir: str,
                 wandb_project: str, wandb_group: str, wandb_run_name: str,
                 wandb_sweep_id: str, wandb_run_id: str,
                 enable_op_level_metrics: bool, enable_cpu_op_level_metrics: bool,
                 enable_chrome_trace: bool, enable_request_outputs: bool,
                 keep_individual_batch_metrics: bool, model_num_layers: int) -> None:
        self.replica_id = replica_id
        self.write_metrics = write_metrics
        self.output_dir = output_dir
        self.wandb_project = wandb_project
        self.wandb_sweep_id = wandb_sweep_id
        self.wandb_run_id = wandb_run_id
        self.wandb_group = wandb_group
        self.wandb_run_name = wandb_run_name
        self.enable_op_level_metrics = enable_op_level_metrics
        self.enable_cpu_op_level_metrics = enable_cpu_op_level_metrics
        self.enable_chrome_trace = enable_chrome_trace
        self.enable_request_outputs = enable_request_outputs
        self.keep_individual_batch_metrics = keep_individual_batch_metrics
        self.model_num_layers = model_num_layers

    def __str__(self) -> str:
        return (f"MetricsConfig(replica_id={self.replica_id}, "
                f"write_metrics={self.write_metrics}, "
                f"output_dir={self.output_dir}, "
                f"wandb_project={self.wandb_project}, "
                f"wandb_group={self.wandb_group}, "
                f"wandb_run_name={self.wandb_run_name}, "
                f"enable_op_level_metrics={self.enable_op_level_metrics}, "
                f"enable_cpu_op_level_metrics={self.enable_cpu_op_level_metrics}, "
                f"enable_chrome_trace={self.enable_chrome_trace}, "
                f"enable_request_outputs={self.enable_request_outputs}, "
                f"keep_individual_batch_metrics="
                f"{self.keep_individual_batch_metrics})")


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str,
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    dtype = dtype.lower()
    if dtype == "auto":
        if config_dtype == torch.float32:
            # Following the common practice, we use float16 for float32 models.
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if derived_max_model_len == float("inf"):
            raise ValueError(
                "When using rope_scaling, the model's config.json must "
                "contain one of the following keys to determine the original "
                f"maximum length of the model: {possible_keys}")
        assert "factor" in rope_scaling
        scaling_factor = rope_scaling["factor"]
        if rope_scaling["type"] == "yarn":
            derived_max_model_len = rope_scaling[
                "original_max_position_embeddings"]
        derived_max_model_len *= scaling_factor

    if max_model_len is None:
        print(
            f"Using the derived maximum model length: {derived_max_model_len}")
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        print(f"Applying rope_scaling to the maximum model length: "
              f"{derived_max_model_len} -> {max_model_len}")
        # force rope_scaling
        scaling_factor = max_model_len / derived_max_model_len
        rope_scaling = {"type": "linear", "factor": scaling_factor}
        hf_config.rope_scaling = rope_scaling

    return max_model_len
