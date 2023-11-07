import argparse
import dataclasses
from dataclasses import asdict, dataclass
import os
from typing import Optional, Tuple

import yaml

from vllm.config import (CacheConfig, MetricsConfig, ModelConfig,
                         ParallelConfig, BaseSchedulerConfig,
                         VLLMSchedulerConfig, OrcaSchedulerConfig,
                         FasterTransformerSchedulerConfig,
                         SarathiSchedulerConfig, DSarathiSchedulerConfig)


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    replica_id: int = 0
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    seed: int = 0
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    disable_log_stats: bool = False
    revision: Optional[str] = None
    quantization: Optional[str] = None
    # scheduler parameters
    scheduler_type: str = 'vllm'
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    # vllm scheduler parameters
    max_num_batched_tokens: Optional[int] = None
    # sarathi scheduler parameters
    chunk_size: Optional[int] = None
    enable_rolling_prefills: bool = True
    prefill_fitting_tolerance: float = 0.2
    # Metrics store parameters
    write_metrics: bool = True
    output_dir: str = '.'
    subsamples: int = 1000
    save_table_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_run_name: Optional[str] = None
    enable_op_level_metrics: bool = False
    enable_chrome_trace: bool = False
    enable_request_outputs: bool = False
    enable_cpu_op_level_metrics: bool = False
    enable_high_level_cuda_metrics: bool = False
    skip_hidden_layers: bool = False

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
        assert (self.max_model_len <= self.max_num_batched_tokens
                if self.max_model_len is not None else True)
        if self.write_metrics:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(f'{self.output_dir}/config.yml', 'w') as f:
                yaml.dump(asdict(self),
                          f,
                          default_flow_style=False,
                          sort_keys=False)

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument('--replica-id',
                            type=int,
                            default=0,
                            help='replica ID')
        parser.add_argument(
            '--model',
            type=str,
            default='facebook/opt-125m',
            help='name or path of the huggingface model to use')
        parser.add_argument(
            '--tokenizer',
            type=str,
            default=EngineArgs.tokenizer,
            help='name or path of the huggingface tokenizer to use')
        parser.add_argument(
            '--revision',
            type=str,
            default=None,
            help='the specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument('--tokenizer-mode',
                            type=str,
                            default=EngineArgs.tokenizer_mode,
                            choices=['auto', 'slow'],
                            help='tokenizer mode. "auto" will use the fast '
                            'tokenizer if available, and "slow" will '
                            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='trust remote code from huggingface')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=['auto', 'pt', 'safetensors', 'npcache', 'dummy'],
            help='The format of the model weights to load. '
            '"auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available. '
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading. '
            '"dummy" will initialize the weights with random values, '
            'which is mainly for profiling.')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=None,
                            help='model context length. If unspecified, '
                            'will be automatically derived from the model.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='random seed')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the model executor')
        parser.add_argument('--scheduler-type',
                            type=str,
                            default=EngineArgs.scheduler_type,
                            help='type of scheduler to use')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--chunk-size',
                            type=int,
                            default=EngineArgs.chunk_size,
                            help='size of each prefill chunk used in sarathi')
        parser.add_argument('--enable-rolling-prefills',
                            action='store_true',
                            default=EngineArgs.enable_rolling_prefills,
                            help='enable rolling prefill in sarathi')
        parser.add_argument(
            '--prefill-fitting-tolerance',
            type=float,
            default=EngineArgs.prefill_fitting_tolerance,
            help=
            'maximum fraction of prefill chunk that can be left empty in sarathi'
        )

        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=str,
                            choices=['awq', None],
                            default=None,
                            help='Method used to quantize the weights')
        # Metrics settings
        parser.add_argument('--write-metrics',
                            action='store_true',
                            help='Capture metrics and export them')
        parser.add_argument('--output-dir',
                            type=str,
                            default=EngineArgs.output_dir,
                            help='directory to save captured metrics')
        parser.add_argument('--subsamples',
                            type=int,
                            default=EngineArgs.subsamples,
                            help='number of subsamples to use for metrics')
        parser.add_argument('--save-table-to-wandb',
                            action='store_true',
                            help='save captured metrics to wandb')
        parser.add_argument('--wandb-project',
                            type=str,
                            default=EngineArgs.wandb_project,
                            help='wandb project name')
        parser.add_argument('--wandb-group',
                            type=str,
                            default=EngineArgs.wandb_group,
                            help='wandb group name')
        parser.add_argument('--wandb-run-name',
                            type=str,
                            default=EngineArgs.wandb_run_name,
                            help='wandb run name')
        parser.add_argument('--enable-op-level-metrics',
                            action='store_true',
                            default=EngineArgs.enable_op_level_metrics,
                            help='enable op-level metrics')
        parser.add_argument('--enable-cpu-op-level-metrics',
                            action='store_true',
                            default=EngineArgs.enable_cpu_op_level_metrics,
                            help='enable cpu op-level metrics')
        parser.add_argument('--enable-high-level-cuda-metrics',
                            action='store_true',
                            default=EngineArgs.enable_high_level_cuda_metrics,
                            help='enable high-level cuda op metrics')
        parser.add_argument('--enable-chrome-trace',
                            action='store_true',
                            default=EngineArgs.enable_chrome_trace,
                            help='enable chrome trace')
        parser.add_argument('--enable-request-outputs',
                            action='store_true',
                            default=EngineArgs.enable_request_outputs,
                            help='enable request outputs')
        parser.add_argument('--skip-hidden-layers',
                            action='store_true',
                            default=EngineArgs.skip_hidden_layers,
                            help='skip hidden layers')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def _get_scheduler_config(
            self, model_config: ModelConfig) -> BaseSchedulerConfig:
        if self.scheduler_type == 'vllm':
            scheduler_config = VLLMSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                self.max_num_batched_tokens,
            )
        elif self.scheduler_type == 'orca':
            scheduler_config = OrcaSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
            )
        elif self.scheduler_type == 'faster_transformer':
            scheduler_config = FasterTransformerSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
            )
        elif self.scheduler_type == 'sarathi':
            scheduler_config = SarathiSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                self.chunk_size,
                self.enable_rolling_prefills,
                self.prefill_fitting_tolerance,
            )
        elif self.scheduler_type == 'dsarathi':
            scheduler_config = DSarathiSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                self.chunk_size,
                self.enable_rolling_prefills,
                self.prefill_fitting_tolerance,
            )
        else:
            raise ValueError(
                f'Unsupported scheduler type: {self.scheduler_type}')

        return scheduler_config

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, BaseSchedulerConfig,
               MetricsConfig]:
        model_config = ModelConfig(self.model, self.tokenizer,
                                   self.tokenizer_mode, self.trust_remote_code,
                                   self.download_dir, self.load_format,
                                   self.dtype, self.seed, self.revision,
                                   self.max_model_len, self.quantization,
                                   self.skip_hidden_layers)
        cache_config = CacheConfig(self.block_size,
                                   self.gpu_memory_utilization,
                                   self.swap_space)
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray)
        scheduler_config = self._get_scheduler_config(model_config)
        metrics_config = MetricsConfig(
            self.write_metrics, self.output_dir, self.subsamples,
            self.save_table_to_wandb, self.wandb_project, self.wandb_group,
            self.wandb_run_name, self.enable_op_level_metrics,
            self.enable_chrome_trace, self.enable_request_outputs,
            self.enable_cpu_op_level_metrics,
            self.enable_high_level_cuda_metrics, self.tensor_parallel_size,
            model_config.hf_config.num_hidden_layers)
        return model_config, cache_config, parallel_config, scheduler_config, metrics_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray',
                            action='store_true',
                            help='use Ray to start the LLM engine in a '
                            'separate process as the server process.')
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='disable logging requests')
        parser.add_argument('--max-log-len',
                            type=int,
                            default=None,
                            help='max number of prompt characters or prompt '
                            'ID numbers being printed in log. '
                            'Default: unlimited.')
        return parser
