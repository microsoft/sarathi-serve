#!/bin/bash

# Compile protobuf files
# Run this script from the sarathi/core/proto directory

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to the root of the project (3 levels up from proto dir)
PROJECT_ROOT="$SCRIPT_DIR/../../.."

# Change to project root to ensure correct import paths
cd "$PROJECT_ROOT"

# Compile the proto files
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --pyi_out=. \
    sarathi/core/proto/datatypes.proto

echo "Protobuf compilation complete!"