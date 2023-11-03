"""Benchmark offline inference throughput."""
import argparse
import datetime
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Sample the requests.
    filtered_dataset = [(x["prompt"], x["prompt_tokens_length"],
                         x["completion_tokens_length"]) for x in dataset]
    sampled_requests = random.sample(filtered_dataset,
                                     min(len(filtered_dataset), num_requests))
    return sampled_requests


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    scheduler_type: str,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    chunk_size: Optional[int],
    enable_rolling_prefills: bool,
    prefill_fitting_tolerance: Optional[float],
    write_metrics: bool,
    output_dir: str,
    enable_op_level_metrics: bool,
    enable_chrome_trace: bool,
    save_table_to_wandb: bool,
    wandb_project: Optional[str],
    wandb_group: Optional[str],
    wandb_run_name: Optional[str],
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        scheduler_type=scheduler_type,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        chunk_size=chunk_size,
        enable_rolling_prefills=enable_rolling_prefills,
        prefill_fitting_tolerance=prefill_fitting_tolerance,
        write_metrics=write_metrics,
        output_dir=output_dir,
        enable_op_level_metrics=enable_op_level_metrics,
        enable_chrome_trace=enable_chrome_trace,
        save_table_to_wandb=save_table_to_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
            logprobs=None,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    llm._run_engine(use_tqdm=True)
    end = time.time()
    llm.log_metrics()
    return end - start


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.time()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.time()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.scheduler_type,
            args.max_num_batched_tokens, args.max_num_seqs,
            args.chunk_size, args.enable_rolling_prefills, args.prefill_fitting_tolerance,
            args.write_metrics, args.output_dir, args.enable_op_level_metrics,
            args.enable_chrome_trace, args.save_table_to_wandb,
            args.wandb_project, args.wandb_group, args.wandb_run_name)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument('--scheduler-type',
                            type=str,
                            default="vllm",
                            help='type of scheduler to use')
    parser.add_argument('--max-num-seqs',
                        type=int,
                        required=True,
                        help='maximum number of sequences per iteration')
    parser.add_argument('--max-num-batched-tokens',
                    type=int,
                    default=None,
                    help='maximum number of batched tokens per '
                    'iteration')
    parser.add_argument('--chunk-size',
                        type=int,
                        default=None,
                        help='size of each prefill chunk used in sarathi')
    parser.add_argument('--enable-rolling-prefills',
                        action="store_true",
                        default=True,
                        help='enable rolling prefill in sarathi')
    parser.add_argument(
        '--prefill-fitting-tolerance',
        type=float,
        default=0.0,
        help=
        'maximum fraction of prefill chunk that can be left empty in sarathi'
    )
    parser.add_argument("--write-metrics",
                        action="store_true",
                        help="'Capture metrics and export them")
    parser.add_argument('--output-dir',
                        type=str,
                        default=f"./outputs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                        help='directory to save captured metrics')
    parser.add_argument('--enable-op-level-metrics',
                        action='store_true',
                        default=True,
                        help='enable op-level metrics')
    parser.add_argument('--enable-chrome-trace',
                        action='store_true',
                        default=True,
                        help='enable chrome trace')
    parser.add_argument("--save-table-to-wandb",
                        action="store_true",
                        help="save captured metrics to wandb")
    parser.add_argument("--wandb-project",
                        type=str,
                        default=None,
                        help="wandb project name")
    parser.add_argument("--wandb-group",
                        type=str,
                        default=None,
                        help="wandb group name")
    parser.add_argument("--wandb-run-name",
                        type=str,
                        default=None,
                        help="wandb run name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)
