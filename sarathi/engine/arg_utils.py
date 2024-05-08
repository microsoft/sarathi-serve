import dataclasses
from dataclasses import asdict, dataclass
import os
from typing import Optional, Tuple, List

import yaml

from sarathi.config import (CacheConfig, MetricsConfig, ModelConfig,
                            ParallelConfig, BaseSchedulerConfig,
                            VLLMSchedulerConfig, OrcaSchedulerConfig,
                            FasterTransformerSchedulerConfig,
                            SimpleChunkingSchedulerConfig,
                            SarathiSchedulerConfig, SchedulerType)


@dataclass
class EngineArgs:
    """Arguments for Sarathi engine."""
    model: str
    replica_id: int = 0
    replica_resource_mapping: List[Tuple[str, int]] = dataclasses.field(
        default_factory=list)
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    seed: int = 0
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    gpu_memory_utilization: float = 0.90
    revision: Optional[str] = None
    # scheduler parameters
    scheduler_type: str = 'sarathi'
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    # vllm scheduler parameters
    max_num_batched_tokens: Optional[int] = None
    # sarathi scheduler parameters
    chunk_size: Optional[int] = None
    enable_dynamic_chunking_schedule: bool = False
    low_chunk_size: Optional[int] = None
    high_chunk_size: Optional[int] = None
    chunk_schedule_max_tokens: Optional[int] = None
    chunk_schedule_stages: Optional[int] = None
    # Metrics store parameters
    write_metrics: bool = True
    output_dir: str = '.'
    wandb_project: Optional[str] = None
    wandb_sweep_id: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_run_name: Optional[str] = None
    enable_op_level_metrics: bool = False
    enable_cpu_op_level_metrics: bool = False
    enable_chrome_trace: bool = False
    enable_request_outputs: bool = False
    keep_individual_batch_metrics: bool = False
    attention_backend: str = 'flash_attention'

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.write_metrics:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(f'{self.output_dir}/config.yml', 'w') as f:
                yaml.dump(asdict(self),
                          f,
                          default_flow_style=False,
                          sort_keys=False)

    def _get_scheduler_config(self, model_config: ModelConfig,
                              num_pipeline_stages: int) -> BaseSchedulerConfig:
        if self.scheduler_type == SchedulerType.VLLM.name.lower():
            scheduler_config = VLLMSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.max_num_batched_tokens,
            )
        elif self.scheduler_type == SchedulerType.ORCA.name.lower():
            scheduler_config = OrcaSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
            )
        elif self.scheduler_type == SchedulerType.FASTER_TRANSFORMER.name.lower(
        ):
            scheduler_config = FasterTransformerSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
            )
        elif self.scheduler_type == SchedulerType.SARATHI.name.lower():
            scheduler_config = SarathiSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.chunk_size,
                self.enable_dynamic_chunking_schedule,
                self.low_chunk_size,
                self.high_chunk_size,
                self.chunk_schedule_max_tokens,
                self.chunk_schedule_stages,
            )
        elif self.scheduler_type == SchedulerType.SIMPLE_CHUNKING.name.lower():
            scheduler_config = SimpleChunkingSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.chunk_size,
            )
        else:
            raise ValueError(
                f'Unsupported scheduler type: {self.scheduler_type}')

        return scheduler_config

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, BaseSchedulerConfig,
               MetricsConfig]:
        model_config = ModelConfig(model=self.model,
                                   tokenizer=self.tokenizer,
                                   tokenizer_mode=self.tokenizer_mode,
                                   trust_remote_code=self.trust_remote_code,
                                   download_dir=self.download_dir,
                                   load_format=self.load_format,
                                   dtype=self.dtype,
                                   seed=self.seed,
                                   revision=self.revision,
                                   max_model_len=self.max_model_len,
                                   attention_backend=self.attention_backend)
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            replica_resource_mapping=self.replica_resource_mapping)
        scheduler_config = self._get_scheduler_config(
            model_config=model_config,
            num_pipeline_stages=self.pipeline_parallel_size)
        metrics_config = MetricsConfig(
            replica_id=self.replica_id,
            write_metrics=self.write_metrics,
            output_dir=self.output_dir,
            wandb_project=self.wandb_project,
            wandb_group=self.wandb_group,
            wandb_run_name=self.wandb_run_name,
            wandb_sweep_id=self.wandb_sweep_id,
            wandb_run_id=self.wandb_run_id,
            enable_op_level_metrics=self.enable_op_level_metrics,
            enable_cpu_op_level_metrics=self.enable_cpu_op_level_metrics,
            enable_chrome_trace=self.enable_chrome_trace,
            enable_request_outputs=self.enable_request_outputs,
            keep_individual_batch_metrics=self.keep_individual_batch_metrics,
            model_num_layers=model_config.get_total_num_layers(),
        )
        return model_config, cache_config, parallel_config, scheduler_config, metrics_config
