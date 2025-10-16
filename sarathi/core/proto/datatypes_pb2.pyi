from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "stop", "ignore_eos", "max_tokens")
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_p: float
    top_k: int
    stop: _containers.RepeatedScalarFieldContainer[str]
    ignore_eos: bool
    max_tokens: int
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., stop: _Optional[_Iterable[str]] = ..., ignore_eos: bool = ..., max_tokens: _Optional[int] = ...) -> None: ...

class LogicalTokenBlock(_message.Message):
    __slots__ = ("block_number", "block_size", "token_ids", "num_tokens")
    BLOCK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    NUM_TOKENS_FIELD_NUMBER: _ClassVar[int]
    block_number: int
    block_size: int
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    num_tokens: int
    def __init__(self, block_number: _Optional[int] = ..., block_size: _Optional[int] = ..., token_ids: _Optional[_Iterable[int]] = ..., num_tokens: _Optional[int] = ...) -> None: ...

class SequenceState(_message.Message):
    __slots__ = ("id", "arrived_at", "num_prompt_tokens", "num_output_tokens", "status", "is_scheduled", "is_completed", "scheduled_at", "completed_at", "prompt_processing_completed_at", "last_restart_at", "last_pause_at", "execution_time", "preempted_time", "last_execution_start_at", "num_restarts", "num_pauses", "is_ignore_finished", "last_token_generated_at", "last_token_generation_time")
    ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVED_AT_FIELD_NUMBER: _ClassVar[int]
    NUM_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    IS_SCHEDULED_FIELD_NUMBER: _ClassVar[int]
    IS_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    SCHEDULED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_PROCESSING_COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_RESTART_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_PAUSE_AT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_FIELD_NUMBER: _ClassVar[int]
    PREEMPTED_TIME_FIELD_NUMBER: _ClassVar[int]
    LAST_EXECUTION_START_AT_FIELD_NUMBER: _ClassVar[int]
    NUM_RESTARTS_FIELD_NUMBER: _ClassVar[int]
    NUM_PAUSES_FIELD_NUMBER: _ClassVar[int]
    IS_IGNORE_FINISHED_FIELD_NUMBER: _ClassVar[int]
    LAST_TOKEN_GENERATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_TOKEN_GENERATION_TIME_FIELD_NUMBER: _ClassVar[int]
    id: str
    arrived_at: float
    num_prompt_tokens: int
    num_output_tokens: int
    status: str
    is_scheduled: bool
    is_completed: bool
    scheduled_at: float
    completed_at: float
    prompt_processing_completed_at: float
    last_restart_at: float
    last_pause_at: float
    execution_time: float
    preempted_time: float
    last_execution_start_at: float
    num_restarts: int
    num_pauses: int
    is_ignore_finished: bool
    last_token_generated_at: float
    last_token_generation_time: float
    def __init__(self, id: _Optional[str] = ..., arrived_at: _Optional[float] = ..., num_prompt_tokens: _Optional[int] = ..., num_output_tokens: _Optional[int] = ..., status: _Optional[str] = ..., is_scheduled: bool = ..., is_completed: bool = ..., scheduled_at: _Optional[float] = ..., completed_at: _Optional[float] = ..., prompt_processing_completed_at: _Optional[float] = ..., last_restart_at: _Optional[float] = ..., last_pause_at: _Optional[float] = ..., execution_time: _Optional[float] = ..., preempted_time: _Optional[float] = ..., last_execution_start_at: _Optional[float] = ..., num_restarts: _Optional[int] = ..., num_pauses: _Optional[int] = ..., is_ignore_finished: bool = ..., last_token_generated_at: _Optional[float] = ..., last_token_generation_time: _Optional[float] = ...) -> None: ...

class Sequence(_message.Message):
    __slots__ = ("seq_id", "prompt", "prompt_token_ids", "block_size", "eos_token_id", "arrival_time", "sampling_params", "output_token_ids", "prompt_tokens_processed", "prompt_tokens_stage_processed", "prompt_processing_finished", "prompt_stage_processing_finished", "output_text", "logical_token_blocks", "prefix_offset", "read_offset", "tokens", "state")
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_STAGE_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    PROMPT_PROCESSING_FINISHED_FIELD_NUMBER: _ClassVar[int]
    PROMPT_STAGE_PROCESSING_FINISHED_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_TOKEN_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    PREFIX_OFFSET_FIELD_NUMBER: _ClassVar[int]
    READ_OFFSET_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    seq_id: str
    prompt: str
    prompt_token_ids: _containers.RepeatedScalarFieldContainer[int]
    block_size: int
    eos_token_id: int
    arrival_time: float
    sampling_params: SamplingParams
    output_token_ids: _containers.RepeatedScalarFieldContainer[int]
    prompt_tokens_processed: int
    prompt_tokens_stage_processed: int
    prompt_processing_finished: bool
    prompt_stage_processing_finished: bool
    output_text: str
    logical_token_blocks: _containers.RepeatedCompositeFieldContainer[LogicalTokenBlock]
    prefix_offset: int
    read_offset: int
    tokens: _containers.RepeatedScalarFieldContainer[str]
    state: SequenceState
    def __init__(self, seq_id: _Optional[str] = ..., prompt: _Optional[str] = ..., prompt_token_ids: _Optional[_Iterable[int]] = ..., block_size: _Optional[int] = ..., eos_token_id: _Optional[int] = ..., arrival_time: _Optional[float] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., output_token_ids: _Optional[_Iterable[int]] = ..., prompt_tokens_processed: _Optional[int] = ..., prompt_tokens_stage_processed: _Optional[int] = ..., prompt_processing_finished: bool = ..., prompt_stage_processing_finished: bool = ..., output_text: _Optional[str] = ..., logical_token_blocks: _Optional[_Iterable[_Union[LogicalTokenBlock, _Mapping]]] = ..., prefix_offset: _Optional[int] = ..., read_offset: _Optional[int] = ..., tokens: _Optional[_Iterable[str]] = ..., state: _Optional[_Union[SequenceState, _Mapping]] = ...) -> None: ...

class SequenceScheduleMetadata(_message.Message):
    __slots__ = ("seq_id", "prompt_chunk_len")
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_CHUNK_LEN_FIELD_NUMBER: _ClassVar[int]
    seq_id: str
    prompt_chunk_len: int
    def __init__(self, seq_id: _Optional[str] = ..., prompt_chunk_len: _Optional[int] = ...) -> None: ...

class SamplerOutput(_message.Message):
    __slots__ = ("seq_id", "output_token")
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    seq_id: str
    output_token: int
    def __init__(self, seq_id: _Optional[str] = ..., output_token: _Optional[int] = ...) -> None: ...

class SamplerOutputs(_message.Message):
    __slots__ = ("outputs",)
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedCompositeFieldContainer[SamplerOutput]
    def __init__(self, outputs: _Optional[_Iterable[_Union[SamplerOutput, _Mapping]]] = ...) -> None: ...

class SchedulerOutputs(_message.Message):
    __slots__ = ("id", "ignored_seq_ids", "preempted_seq_ids", "scheduled_seq_metadata_list", "prompt_chunk_lens", "num_batched_prompt_tokens", "num_batched_output_tokens", "num_batched_tokens")
    ID_FIELD_NUMBER: _ClassVar[int]
    IGNORED_SEQ_IDS_FIELD_NUMBER: _ClassVar[int]
    PREEMPTED_SEQ_IDS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULED_SEQ_METADATA_LIST_FIELD_NUMBER: _ClassVar[int]
    PROMPT_CHUNK_LENS_FIELD_NUMBER: _ClassVar[int]
    NUM_BATCHED_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_BATCHED_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_BATCHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    id: int
    ignored_seq_ids: _containers.RepeatedScalarFieldContainer[str]
    preempted_seq_ids: _containers.RepeatedScalarFieldContainer[str]
    scheduled_seq_metadata_list: _containers.RepeatedCompositeFieldContainer[SequenceScheduleMetadata]
    prompt_chunk_lens: _containers.RepeatedScalarFieldContainer[int]
    num_batched_prompt_tokens: int
    num_batched_output_tokens: int
    num_batched_tokens: int
    def __init__(self, id: _Optional[int] = ..., ignored_seq_ids: _Optional[_Iterable[str]] = ..., preempted_seq_ids: _Optional[_Iterable[str]] = ..., scheduled_seq_metadata_list: _Optional[_Iterable[_Union[SequenceScheduleMetadata, _Mapping]]] = ..., prompt_chunk_lens: _Optional[_Iterable[int]] = ..., num_batched_prompt_tokens: _Optional[int] = ..., num_batched_output_tokens: _Optional[int] = ..., num_batched_tokens: _Optional[int] = ...) -> None: ...

class StepInputs(_message.Message):
    __slots__ = ("scheduler_outputs", "new_seqs", "pending_step_outputs")
    SCHEDULER_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    NEW_SEQS_FIELD_NUMBER: _ClassVar[int]
    PENDING_STEP_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    scheduler_outputs: SchedulerOutputs
    new_seqs: _containers.RepeatedCompositeFieldContainer[Sequence]
    pending_step_outputs: _containers.RepeatedCompositeFieldContainer[PendingStepOutput]
    def __init__(self, scheduler_outputs: _Optional[_Union[SchedulerOutputs, _Mapping]] = ..., new_seqs: _Optional[_Iterable[_Union[Sequence, _Mapping]]] = ..., pending_step_outputs: _Optional[_Iterable[_Union[PendingStepOutput, _Mapping]]] = ...) -> None: ...

class PendingStepOutput(_message.Message):
    __slots__ = ("scheduler_outputs", "sampler_outputs")
    SCHEDULER_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SAMPLER_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    scheduler_outputs: SchedulerOutputs
    sampler_outputs: SamplerOutputs
    def __init__(self, scheduler_outputs: _Optional[_Union[SchedulerOutputs, _Mapping]] = ..., sampler_outputs: _Optional[_Union[SamplerOutputs, _Mapping]] = ...) -> None: ...
