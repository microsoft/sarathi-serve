#!/bin/bash
set -x

_script_dir=$(dirname "$0")
# root dir is 3 levels up
ROOT_DIR=$(dirname $(dirname $(dirname $_script_dir)))

MODEL_NAME=meta-llama/Meta-Llama-3-8B
TP_DREGREE=8
PREFILL_LENGTH=1024
DECODE_LENGTH=1
MIN_CHUNK_SIZE=1024
MAX_CHUNK_SIZE=1024
ATTENTION_BACKEND="FLASHINFER_UNPAGED"

# read cli args for above defaults

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --model_name)
    MODEL_NAME="$2"
    shift
    shift
    ;;
    --model_tensor_parallel_degree)
    TP_DREGREE="$2"
    shift
    shift
    ;;
    --prefill_length)
    PREFILL_LENGTH="$2"
    shift
    shift
    ;;
    --decode_length)
    DECODE_LENGTH="$2"
    shift
    shift
    ;;
    --min_chunk_size)
    MIN_CHUNK_SIZE="$2"
    shift
    shift
    ;;
    --max_chunk_size)
    MAX_CHUNK_SIZE="$2"
    shift
    shift
    ;;
    --attention_backend)
    ATTENTION_BACKEND="$2"
    shift
    shift
    ;;
    *)
    echo "Unknown option $1"
    exit 1
    ;;
esac
done

_PREFILL_LENGTH=$((1024 * PREFILL_LENGTH))
TOTAL_LENGTH=$((_PREFILL_LENGTH + DECODE_LENGTH))

cd $ROOT_DIR

RUN_NAME=l-${PREFILL_LENGTH}-${DECODE_LENGTH}_tp-${TP_DREGREE}_cs${MIN_CHUNK_SIZE}-${MAX_CHUNK_SIZE}-ab-${ATTENTION_BACKEND}-${MODEL_NAME}
OUTPUT_DIR=$ROOT_DIR/benchmark_output/single_req_prefill_test/$RUN_NAME

mkdir -p $OUTPUT_DIR

python -m sarathi.benchmark.main \
--output_dir $OUTPUT_DIR \
--model_name $MODEL_NAME \
--model_attention_backend $ATTENTION_BACKEND \
--model_max_model_len $TOTAL_LENGTH \
--model_tensor_parallel_degree $TP_DREGREE \
--model_pipeline_parallel_degree 1 \
--request_generator_provider synthetic \
--synthetic_request_generator_interval_provider static \
--synthetic_request_generator_num_requests 1 \
--synthetic_request_generator_length_provider fixed \
--fixed_request_length_generator_prefill_tokens $_PREFILL_LENGTH \
--fixed_request_length_generator_decode_tokens $DECODE_LENGTH \
--replica_scheduler_provider sarathi \
--replica_scheduler_max_batch_size 128 \
--sarathi_scheduler_enable_dynamic_chunking_schedule true \
--sarathi_scheduler_low_chunk_size $MIN_CHUNK_SIZE \
--sarathi_scheduler_high_chunk_size $MAX_CHUNK_SIZE \
--sarathi_scheduler_chunk_schedule_max_tokens $((1024 * 1024)) \
--sarathi_scheduler_chunk_schedule_stages 16 \
--metrics_store_wandb_project mnemosyne \
--metrics_store_wandb_group single_req_prefill_test \
--metrics_store_wandb_run_name $RUN_NAME \
--metrics_store_enable_op_level_metrics false
