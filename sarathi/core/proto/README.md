# Protocol Buffer Support for Sarathi

This directory contains Protocol Buffer (protobuf) definitions for secure data serialization in Sarathi.

## Background

Previously, Sarathi used Python's `pickle` for data exchange over ZMQ sockets via `send_pyobj()`/`recv_pyobj()`. While convenient, pickle has known security vulnerabilities that can allow arbitrary code execution when deserializing untrusted data.

To address this security concern, we've migrated to Protocol Buffers, which provides:
- Type-safe serialization
- Cross-language compatibility
- Better performance for large messages
- Protection against arbitrary code execution

## Files

- `datatypes.proto` - Protocol buffer definitions for all major data structures used in ZMQ communication
- `compile_proto.sh` - Script to compile the .proto files into Python classes
- `README.md` - This file

## Compiling Protocol Buffers

Before using the protobuf-based communication, you need to compile the .proto files:

```bash
cd sarathi/core/proto
./compile_proto.sh
```

This will generate:
- `datatypes_pb2.py` - Python classes for the protobuf messages
- `datatypes_pb2.pyi` - Type stubs for better IDE support

## Usage

The migration replaces all `send_pyobj()`/`recv_pyobj()` calls with protobuf serialization:

### Before (using pickle):
```python
# Sending
socket.send_pyobj(data)

# Receiving
data = socket.recv_pyobj()
```

### After (using protobuf):
```python
from sarathi.core.proto_utils import serialize_step_inputs, deserialize_step_inputs

# Sending
serialized_data = serialize_step_inputs(step_inputs)
socket.send(serialized_data)

# Receiving
data = socket.recv()
step_inputs = deserialize_step_inputs(data)
```

## Key Data Structures

The following data structures are serialized using protobuf:

1. **StepInputs** - Sent from engine to workers containing:
   - Scheduler outputs
   - New sequences to process
   - Pending step outputs (for pipeline parallelism)

2. **SamplerOutputs** - Sent from workers back to engine containing:
   - Generated tokens for each sequence

3. **Supporting structures**:
   - Sequence (full sequence information)
   - SchedulerOutputs (scheduling decisions)
   - SamplingParams (generation parameters)
   - SequenceState (sequence execution state)
   - LogicalTokenBlock (KV cache block information)

## Security Benefits

By using Protocol Buffers instead of pickle:
1. **No arbitrary code execution** - Protobuf only deserializes data, not executable code
2. **Schema validation** - Data must conform to the defined schema
3. **Type safety** - Strong typing prevents type confusion attacks
4. **Version compatibility** - Built-in support for schema evolution

## Performance Considerations

While protobuf adds a small serialization overhead compared to pickle, it provides:
- Better performance for large messages due to efficient binary encoding
- Smaller message sizes for numeric data
- Predictable performance characteristics

The security benefits far outweigh the minimal performance impact.