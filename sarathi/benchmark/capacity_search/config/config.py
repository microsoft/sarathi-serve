import hashlib
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional


def _get_hash(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


@dataclass
class ModelConfig:
    name: str
    identifier: str
    parallel_specs: List[str] = field(default_factory=list)
    scheduler_specs: List[str] = field(default_factory=list)
    traces: List[str] = field(default_factory=list)

    def get_key(self):
        return self.name

    def get_human_readable_name(self):
        return f"Model: {self.name}"

    def to_config_dict(self):
        return {
            "model_config_model": self.identifier,
        }

    def is_parallel_spec_valid(self, spec_name: str):
        return not self.parallel_specs or spec_name in self.parallel_specs

    def is_scheduler_spec_valid(self, spec_name: str):
        return not self.scheduler_specs or spec_name in self.scheduler_specs

    def is_traces_valid(self, trace_name: str):
        return not self.traces or trace_name in self.traces


@dataclass
class TraceConfig:
    name: str
    trace_file: str
    max_seq_len: int
    num_requests: int
    start_qps: float

    def get_key(self):
        return f"{self.name}_tk{self.max_seq_len}_rq{self.num_requests}"

    def get_human_readable_name(self):
        return f"Trace: {self.name}, Max Seq Len: {self.max_seq_len}, Num Requests: {self.num_requests}, Start QPS: {self.start_qps}"

    def to_config_dict(self):
        return {
            "request_generator_config_type": "SYNTHETIC",
            "length_generator_config_type": "TRACE",
            "interval_generator_config_type": "POISSON",
            "trace_request_length_generator_config_max_tokens": self.max_seq_len,
            "model_config_max_model_len": self.max_seq_len,
            "trace_request_length_generator_config_trace_file": self.trace_file,
            "trace_request_length_generator_config_prefill_scale_factor": 1,
            "trace_request_length_generator_config_decode_scale_factor": 1,
            "synthetic_request_generator_config_num_requests": self.num_requests,
            "vllm_scheduler_config_max_batched_tokens": self.max_seq_len,
        }


@dataclass
class SchedulerConfig:
    name: str
    scheduler: str
    batch_size: int
    chunk_size: Optional[int] = None

    def get_key(self):
        key = f"{self.scheduler}_bs{self.batch_size}"

        if self.chunk_size is not None:
            key += f"_cs{self.chunk_size}"

        return key

    def get_human_readable_name(self):
        return f"Scheduler: {self.scheduler}, Batch Size: {self.batch_size}, Chunk Size: {self.chunk_size}"

    def to_config_dict(self):
        if self.scheduler == "vllm":
            return {
                "scheduler_config_type": "VLLM",
                "vllm_scheduler_config_max_num_seqs": self.batch_size,
            }
        elif self.scheduler == "orca":
            return {
                "scheduler_config_type": "ORCA",
                "orca_scheduler_config_max_num_seqs": self.batch_size,
            }
        elif self.scheduler == "sarathi":
            assert self.chunk_size is not None
            return {
                "scheduler_config_type": "SARATHI",
                "sarathi_scheduler_config_max_num_seqs": self.batch_size,
                "sarathi_scheduler_config_chunk_size": self.chunk_size,
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")


@dataclass
class ParallelConfig:
    name: str
    tp_dimension: int
    pp_dimension: int

    def get_key(self):
        return f"tp{self.tp_dimension}_pp{self.pp_dimension}"

    def get_human_readable_name(self):
        return f"TP: {self.tp_dimension}, PP: {self.pp_dimension}"

    def get_num_gpus(self):
        return self.tp_dimension * self.pp_dimension

    def to_config_dict(self):
        return {
            "parallel_config_tensor_parallel_size": self.tp_dimension,
            "parallel_config_pipeline_parallel_size": self.pp_dimension,
        }


class JobConfig:

    def __init__(
        self,
        model_config: ModelConfig,
        trace_config: TraceConfig,
        scheduler_config: SchedulerConfig,
        parallel_config: ParallelConfig,
    ):
        self.model_config = model_config
        self.trace_config = trace_config
        self.scheduler_config = scheduler_config
        self.parallel_config = parallel_config

        self.start_qps = self.trace_config.start_qps

    def get_key(self):
        config_keys = [
            self.model_config.get_key(),
            self.trace_config.get_key(),
            self.scheduler_config.get_key(),
            self.parallel_config.get_key(),
        ]

        return "_".join(config_keys)

    def get_wandb_run_name(self):
        substrings = [
            self.model_config.get_wandb_run_name(),
            self.trace_config.get_wandb_run_name(),
            self.scheduler_config.get_wandb_run_name(),
            self.parallel_config.get_wandb_run_name(),
        ]
        return "_".join(substrings)

    def get_human_readable_name(self):
        substrings = [
            self.model_config.get_human_readable_name(),
            self.trace_config.get_human_readable_name(),
            self.scheduler_config.get_human_readable_name(),
            self.parallel_config.get_human_readable_name(),
            f"Hash: {_get_hash(self.get_key())}",
        ]
        return ", ".join(substrings)

    def get_num_gpus(self):
        return self.parallel_config.get_num_gpus()

    def to_config_dict(self):
        return {
            **self.model_config.to_config_dict(),
            **self.trace_config.to_config_dict(),
            **self.parallel_config.to_config_dict(),
            **self.scheduler_config.to_config_dict(),
        }

    @classmethod
    def generate_job_configs(cls, config: dict):
        job_configs = []
        for (
            model_config,
            trace_config,
            scheduler_config,
            parallel_config,
        ) in product(
            config["models"],
            config["traces"],
            config["schedulers"],
            config["parallel_spec"],
        ):
            model_config = ModelConfig(**model_config)
            trace_config = TraceConfig(**trace_config)
            scheduler_config = SchedulerConfig(**scheduler_config)
            parallel_config = ParallelConfig(**parallel_config)

            if (
                not model_config.is_parallel_spec_valid(parallel_config.name)
                or not model_config.is_scheduler_spec_valid(scheduler_config.name)
                or not model_config.is_traces_valid(trace_config.name)
            ):
                continue

            job_config = cls(
                model_config,
                trace_config,
                scheduler_config,
                parallel_config,
            )
            job_configs.append(job_config)

        return job_configs

    def __str__(self) -> str:
        return self.get_human_readable_name()


@dataclass
class BenchmarkConfig:
    output_dir: str
    wandb_project: str
    wandb_group: str
    wandb_sweep_id: str
    qps: float
    time_limit: int
    job_config: JobConfig

    def to_config_dict(self):
        if self.wandb_project:
            wandb_args = {
                "metrics_config_wandb_project": self.wandb_project,
                "metrics_config_wandb_group": self.job_config.get_key(),
                "metrics_config_wandb_sweep_id": self.wandb_sweep_id,
                "metrics_config_wandb_run_id": self.get_run_id(),
                "metrics_config_wandb_run_name": f"qps_{self.qps}",
            }
        else:
            wandb_args = {}

        return {
            **self.job_config.to_config_dict(),
            "output_dir": self.get_run_dir(),
            "poisson_request_interval_generator_config_qps": self.qps,
            "time_limit": self.time_limit * 60,  # to seconds
            "metrics_config_enable_op_level_metrics": False,
            "metrics_config_enable_cpu_op_level_metrics": False,
            "metrics_config_keep_individual_batch_metrics": False,
            "metrics_config_enable_chrome_trace": False,
            **wandb_args,
        }

    def get_run_id(self):
        return _get_hash(self.get_key())

    def get_key(self):
        return f"{self.job_config.get_key()}_qps{self.qps}"

    def to_args(self):
        args = []

        for key, value in self.to_config_dict().items():
            if value is not None:
                args.append(f"--{key} {value}")
            else:
                args.append(f"--{key}")

        return " ".join(args)

    def to_human_readable_name(self):
        return f"{self.job_config.get_human_readable_name()}, QPS: {self.qps}, Run id: {self.get_run_id()}"

    def get_run_dir(self):
        return (
            f"{self.output_dir}/runs/{_get_hash(self.job_config.get_key())}/{self.qps}"
        )
