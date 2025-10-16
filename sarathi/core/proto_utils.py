"""Utilities for protobuf serialization and deserialization."""

from typing import List, Optional, Tuple

from sarathi.core.datatypes.block import LogicalTokenBlock
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    Sequence,
    SequenceScheduleMetadata,
)
from sarathi.core.datatypes.sequence_state import SequenceState
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.core.datatypes.step_inputs import StepInputs

# Import generated protobuf classes
# Note: These will be generated from the .proto file using protoc
from sarathi.core.proto import datatypes_pb2


def sampling_params_to_proto(params: SamplingParams) -> datatypes_pb2.SamplingParams:
    """Convert SamplingParams to protobuf."""
    proto = datatypes_pb2.SamplingParams()
    proto.temperature = params.temperature
    proto.top_p = params.top_p
    proto.top_k = params.top_k
    proto.stop.extend(params.stop)
    proto.ignore_eos = params.ignore_eos
    proto.max_tokens = params.max_tokens
    return proto


def sampling_params_from_proto(proto: datatypes_pb2.SamplingParams) -> SamplingParams:
    """Convert protobuf to SamplingParams."""
    return SamplingParams(
        temperature=proto.temperature,
        top_p=proto.top_p,
        top_k=proto.top_k,
        stop=list(proto.stop),
        ignore_eos=proto.ignore_eos,
        max_tokens=proto.max_tokens,
    )


def logical_token_block_to_proto(
    block: LogicalTokenBlock,
) -> datatypes_pb2.LogicalTokenBlock:
    """Convert LogicalTokenBlock to protobuf."""
    proto = datatypes_pb2.LogicalTokenBlock()
    proto.block_number = block.block_number
    proto.block_size = block.block_size
    proto.token_ids.extend(block.token_ids)
    proto.num_tokens = block.num_tokens
    return proto


def logical_token_block_from_proto(
    proto: datatypes_pb2.LogicalTokenBlock,
) -> LogicalTokenBlock:
    """Convert protobuf to LogicalTokenBlock."""
    block = LogicalTokenBlock(
        block_number=proto.block_number,
        block_size=proto.block_size,
    )
    block.token_ids = list(proto.token_ids)
    block.num_tokens = proto.num_tokens
    return block


def sequence_state_to_proto(state: SequenceState) -> datatypes_pb2.SequenceState:
    """Convert SequenceState to protobuf."""
    proto = datatypes_pb2.SequenceState()
    proto.id = state._id
    proto.arrived_at = state._arrived_at
    proto.num_prompt_tokens = state._num_prompt_tokens
    proto.num_output_tokens = state._num_output_tokens
    proto.status = state._status.name
    proto.is_scheduled = state._is_scheduled
    proto.is_completed = state._is_completed
    
    if state._scheduled_at is not None:
        proto.scheduled_at = state._scheduled_at
    if state._completed_at is not None:
        proto.completed_at = state._completed_at
    if state._prompt_processing_completed_at is not None:
        proto.prompt_processing_completed_at = state._prompt_processing_completed_at
    if state._last_restart_at is not None:
        proto.last_restart_at = state._last_restart_at
    if state._last_pause_at is not None:
        proto.last_pause_at = state._last_pause_at
    
    proto.execution_time = state._execution_time
    proto.preempted_time = state._preempted_time
    
    if state._last_execution_start_at is not None:
        proto.last_execution_start_at = state._last_execution_start_at
    
    proto.num_restarts = state._num_restarts
    proto.num_pauses = state._num_pauses
    proto.is_ignore_finished = state._is_ignore_finished
    
    if state._last_token_generated_at is not None:
        proto.last_token_generated_at = state._last_token_generated_at
    
    proto.last_token_generation_time = state._last_token_generation_time
    
    return proto


def sequence_state_from_proto(proto: datatypes_pb2.SequenceState) -> SequenceState:
    """Convert protobuf to SequenceState."""
    state = SequenceState(
        id=proto.id,
        arrived_at=proto.arrived_at,
        num_prompt_tokens=proto.num_prompt_tokens,
    )
    
    # Restore internal state
    state._num_output_tokens = proto.num_output_tokens
    state._status = SequenceStatus[proto.status]
    state._is_scheduled = proto.is_scheduled
    state._is_completed = proto.is_completed
    
    if proto.HasField("scheduled_at"):
        state._scheduled_at = proto.scheduled_at
    if proto.HasField("completed_at"):
        state._completed_at = proto.completed_at
    if proto.HasField("prompt_processing_completed_at"):
        state._prompt_processing_completed_at = proto.prompt_processing_completed_at
    if proto.HasField("last_restart_at"):
        state._last_restart_at = proto.last_restart_at
    if proto.HasField("last_pause_at"):
        state._last_pause_at = proto.last_pause_at
    
    state._execution_time = proto.execution_time
    state._preempted_time = proto.preempted_time
    
    if proto.HasField("last_execution_start_at"):
        state._last_execution_start_at = proto.last_execution_start_at
    
    state._num_restarts = proto.num_restarts
    state._num_pauses = proto.num_pauses
    state._is_ignore_finished = proto.is_ignore_finished
    
    if proto.HasField("last_token_generated_at"):
        state._last_token_generated_at = proto.last_token_generated_at
    
    state._last_token_generation_time = proto.last_token_generation_time
    
    return state


def sequence_to_proto(seq: Sequence) -> datatypes_pb2.Sequence:
    """Convert Sequence to protobuf."""
    proto = datatypes_pb2.Sequence()
    proto.seq_id = seq.seq_id
    proto.prompt = seq.prompt
    proto.prompt_token_ids.extend(seq.prompt_token_ids)
    proto.block_size = seq.block_size
    proto.eos_token_id = seq.eos_token_id
    proto.arrival_time = seq.arrival_time
    proto.sampling_params.CopyFrom(sampling_params_to_proto(seq.sampling_params))
    proto.output_token_ids.extend(seq.output_token_ids)
    proto.prompt_tokens_processed = seq.prompt_tokens_processed
    proto.prompt_tokens_stage_processed = seq.prompt_tokens_stage_processed
    proto.prompt_processing_finished = seq.prompt_processing_finished
    proto.prompt_stage_processing_finished = seq.prompt_stage_processing_finished
    proto.output_text = seq.output_text
    
    for block in seq.logical_token_blocks:
        proto.logical_token_blocks.append(logical_token_block_to_proto(block))
    
    proto.prefix_offset = seq.prefix_offset
    proto.read_offset = seq.read_offset
    
    if seq.tokens is not None:
        proto.tokens.extend(seq.tokens)
    
    proto.state.CopyFrom(sequence_state_to_proto(seq.state))
    
    return proto


def sequence_from_proto(proto: datatypes_pb2.Sequence) -> Sequence:
    """Convert protobuf to Sequence."""
    # Create sequence with basic info
    seq = Sequence(
        seq_id=proto.seq_id,
        prompt=proto.prompt,
        prompt_token_ids=list(proto.prompt_token_ids),
        block_size=proto.block_size,
        eos_token_id=proto.eos_token_id,
        arrival_time=proto.arrival_time,
        sampling_params=sampling_params_from_proto(proto.sampling_params),
    )
    
    # Restore other fields
    seq.output_token_ids = list(proto.output_token_ids)
    seq.prompt_tokens_processed = proto.prompt_tokens_processed
    seq.prompt_tokens_stage_processed = proto.prompt_tokens_stage_processed
    seq.prompt_processing_finished = proto.prompt_processing_finished
    seq.prompt_stage_processing_finished = proto.prompt_stage_processing_finished
    seq.output_text = proto.output_text
    
    # Restore logical token blocks
    seq.logical_token_blocks = []
    for block_proto in proto.logical_token_blocks:
        seq.logical_token_blocks.append(logical_token_block_from_proto(block_proto))
    
    seq.prefix_offset = proto.prefix_offset
    seq.read_offset = proto.read_offset
    
    if proto.tokens:
        seq.tokens = list(proto.tokens)
    
    # Restore state
    seq.state = sequence_state_from_proto(proto.state)
    
    return seq


def sequence_schedule_metadata_to_proto(
    metadata: SequenceScheduleMetadata,
) -> datatypes_pb2.SequenceScheduleMetadata:
    """Convert SequenceScheduleMetadata to protobuf."""
    proto = datatypes_pb2.SequenceScheduleMetadata()
    proto.seq_id = metadata.seq_id
    proto.prompt_chunk_len = metadata.prompt_chunk_len
    return proto


def sequence_schedule_metadata_from_proto(
    proto: datatypes_pb2.SequenceScheduleMetadata,
) -> SequenceScheduleMetadata:
    """Convert protobuf to SequenceScheduleMetadata."""
    return SequenceScheduleMetadata(
        seq_id=proto.seq_id,
        prompt_chunk_len=proto.prompt_chunk_len,
    )


def sampler_output_to_proto(output: SamplerOutput) -> datatypes_pb2.SamplerOutput:
    """Convert SamplerOutput to protobuf."""
    proto = datatypes_pb2.SamplerOutput()
    proto.seq_id = output.seq_id
    proto.output_token = output.output_token
    return proto


def sampler_output_from_proto(proto: datatypes_pb2.SamplerOutput) -> SamplerOutput:
    """Convert protobuf to SamplerOutput."""
    return SamplerOutput(
        seq_id=proto.seq_id,
        output_token=proto.output_token,
    )


def sampler_outputs_to_proto(
    outputs: SamplerOutputs,
) -> datatypes_pb2.SamplerOutputs:
    """Convert SamplerOutputs to protobuf."""
    proto = datatypes_pb2.SamplerOutputs()
    for output in outputs:
        proto.outputs.append(sampler_output_to_proto(output))
    return proto


def sampler_outputs_from_proto(
    proto: datatypes_pb2.SamplerOutputs,
) -> SamplerOutputs:
    """Convert protobuf to SamplerOutputs."""
    return [sampler_output_from_proto(output) for output in proto.outputs]


def scheduler_outputs_to_proto(
    outputs: SchedulerOutputs,
) -> datatypes_pb2.SchedulerOutputs:
    """Convert SchedulerOutputs to protobuf."""
    proto = datatypes_pb2.SchedulerOutputs()
    proto.id = outputs.id
    proto.ignored_seq_ids.extend(outputs.ignored_seq_ids)
    proto.preempted_seq_ids.extend(outputs.preempted_seq_ids)
    
    for metadata in outputs.scheduled_seq_metadata_list:
        proto.scheduled_seq_metadata_list.append(
            sequence_schedule_metadata_to_proto(metadata)
        )
    
    proto.prompt_chunk_lens.extend(outputs.prompt_chunk_lens)
    proto.num_batched_prompt_tokens = outputs.num_batched_prompt_tokens
    proto.num_batched_output_tokens = outputs.num_batched_output_tokens
    proto.num_batched_tokens = outputs.num_batched_tokens
    
    return proto


def scheduler_outputs_from_proto(
    proto: datatypes_pb2.SchedulerOutputs,
) -> SchedulerOutputs:
    """Convert protobuf to SchedulerOutputs."""
    metadata_list = [
        sequence_schedule_metadata_from_proto(metadata)
        for metadata in proto.scheduled_seq_metadata_list
    ]
    
    return SchedulerOutputs(
        id=proto.id,
        ignored_seq_ids=list(proto.ignored_seq_ids),
        preempted_seq_ids=list(proto.preempted_seq_ids),
        scheduled_seq_metadata_list=metadata_list,
    )


def step_inputs_to_proto(inputs: StepInputs) -> datatypes_pb2.StepInputs:
    """Convert StepInputs to protobuf."""
    proto = datatypes_pb2.StepInputs()
    proto.scheduler_outputs.CopyFrom(
        scheduler_outputs_to_proto(inputs.scheduler_outputs)
    )
    
    if inputs.new_seqs:
        for seq in inputs.new_seqs:
            proto.new_seqs.append(sequence_to_proto(seq))
    
    if inputs.pending_step_outputs:
        for scheduler_outputs, sampler_outputs in inputs.pending_step_outputs:
            pending = datatypes_pb2.PendingStepOutput()
            pending.scheduler_outputs.CopyFrom(
                scheduler_outputs_to_proto(scheduler_outputs)
            )
            pending.sampler_outputs.CopyFrom(
                sampler_outputs_to_proto(sampler_outputs)
            )
            proto.pending_step_outputs.append(pending)
    
    return proto


def step_inputs_from_proto(proto: datatypes_pb2.StepInputs) -> StepInputs:
    """Convert protobuf to StepInputs."""
    new_seqs = None
    if proto.new_seqs:
        new_seqs = [sequence_from_proto(seq) for seq in proto.new_seqs]
    
    pending_step_outputs = None
    if proto.pending_step_outputs:
        pending_step_outputs = []
        for pending in proto.pending_step_outputs:
            scheduler_outputs = scheduler_outputs_from_proto(pending.scheduler_outputs)
            sampler_outputs = sampler_outputs_from_proto(pending.sampler_outputs)
            pending_step_outputs.append((scheduler_outputs, sampler_outputs))
    
    return StepInputs(
        scheduler_outputs=scheduler_outputs_from_proto(proto.scheduler_outputs),
        new_seqs=new_seqs,
        pending_step_outputs=pending_step_outputs,
    )


def serialize_step_inputs(inputs: StepInputs) -> bytes:
    """Serialize StepInputs to bytes."""
    proto = step_inputs_to_proto(inputs)
    return proto.SerializeToString()


def deserialize_step_inputs(data: bytes) -> StepInputs:
    """Deserialize bytes to StepInputs."""
    proto = datatypes_pb2.StepInputs()
    proto.ParseFromString(data)
    return step_inputs_from_proto(proto)


def serialize_sampler_outputs(outputs: SamplerOutputs) -> bytes:
    """Serialize SamplerOutputs to bytes."""
    proto = sampler_outputs_to_proto(outputs)
    return proto.SerializeToString()


def deserialize_sampler_outputs(data: bytes) -> SamplerOutputs:
    """Deserialize bytes to SamplerOutputs."""
    proto = datatypes_pb2.SamplerOutputs()
    proto.ParseFromString(data)
    return sampler_outputs_from_proto(proto)