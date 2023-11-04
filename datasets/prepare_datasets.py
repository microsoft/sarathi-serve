""" Create a set of datasets with specific prompt lengths and prefill to decode ratios """
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

prompt_length_bins = [
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]
prefill_decode_ratio_bins = [
    0.125,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
]


def get_key(prompt_length: int, prefill_decode_ratio: float):
    return f"prompt_length_{prompt_length}_pd_ratio_{prefill_decode_ratio}"


def plot_scatter(
    data: List[Tuple[int, float]],
    xlabel: str,
    ylabel: str,
    filepath: str,
):
    xitems, yitems = zip(*data)
    plt.scatter(xitems, yitems)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filepath)
    plt.clf()


def plot_cdf(
    data: Union[List[int], List[float]],
    xlabel: str,
    title: str,
    ylabel: str,
    filename: str,
):
    sorted_data = np.sort(data)
    # Calculate the cumulative distribution function
    cdf = np.cumsum(sorted_data) / np.sum(sorted_data)
    plt.plot(sorted_data, cdf)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def plot_cdfs(datapoints, key):
    plot_cdf([x["completion_tokens_length"] for x in datapoints],
             "Completion Length", f"Completion Length CDF for {key}", "CDF",
             f"{key}_completion_length_cdf.svg")
    plot_cdf([x["prefill_to_decode_ratio"]
              for x in datapoints], "Prefill to Decode Ratio",
             f"Prefill to Decode Ratio CDF for {key}", "CDF",
             f"{key}_prefill_to_decode_ratio_cdf.svg")
    plot_cdf([x["prompt_tokens_length"] for x in datapoints], "Prompt Length",
             f"Prompt Length CDF for {key}", "CDF",
             f"{key}_prompt_length_cdf.svg")


def prepare_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
):
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2 and data["conversations"][0]["from"]
        == "human" and data["conversations"][1]["from"] == "gpt"
    ]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
    """
     { "prompt": str, "prompt_token_ids": List[int], "output_len": int, "prefill_to_decode_ratio": float}
    """
    split_datasets = {}
    full_dataset = []
    prompt_length_prefill_to_decode_ratio_pairs = []

    for datapoint in dataset:
        prompt, completion = datapoint
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prefill_decode_ratio = len(prompt_token_ids) / len(
            completion_token_ids)

        if (len(prompt_token_ids) < prompt_length_bins[0]
                or len(prompt_token_ids) > prompt_length_bins[-1]):
            # Prune too short or too long sequences.
            continue

        if (prefill_decode_ratio < prefill_decode_ratio_bins[0]
                or prefill_decode_ratio > prefill_decode_ratio_bins[-1]):
            # Prune too small or large prefill to decode ratios.
            continue

        prompt_length_prefill_to_decode_ratio_pairs.append(
            (len(prompt_token_ids),
             len(prompt_token_ids) / len(completion_token_ids)))
        nearest_prompt_length = min(
            prompt_length_bins, key=lambda x: abs(x - len(prompt_token_ids)))
        nearest_prefill_decode_ratio = min(
            prefill_decode_ratio_bins,
            key=lambda x: abs(x - len(prompt_token_ids) / len(
                completion_token_ids)),
        )
        key = get_key(nearest_prompt_length, nearest_prefill_decode_ratio)
        if split_datasets.get(key) is None:
            split_datasets[key] = []
        datapoint = {
            "prompt": prompt,
            "completion": completion,
            "prompt_tokens_length": len(prompt_token_ids),
            "completion_tokens_length": len(completion_token_ids),
            "prefill_to_decode_ratio": prefill_decode_ratio,
        }
        split_datasets[key].append(datapoint)
        full_dataset.append(datapoint)

    plot_scatter(prompt_length_prefill_to_decode_ratio_pairs,
                 "Sequence Length", "P:D Ratio",
                 "datasets/prefill_to_decode_ratio_vs_sequence_length.svg")
    plot_cdfs(full_dataset, "datasets/all")

    for prompt_length in prompt_length_bins:
        for prefill_decode_ratio in prefill_decode_ratio_bins:
            key = get_key(prompt_length, prefill_decode_ratio)
            datapoints = split_datasets.get(key, [])
            Path(f"datasets/{key}").mkdir(parents=True, exist_ok=True)
            with open(f"datasets/{key}/{key}.json", "w") as f:
                json.dump(datapoints, f, indent=4)
            plot_cdfs(datapoints, f"datasets/{key}/{key}")
    return


def prepare_truncated_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
):
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2 and data["conversations"][0]["from"]
        == "human" and data["conversations"][1]["from"] == "gpt"
    ]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
    truncated_dataset = []
    for datapoint in dataset:
        prompt, completion = datapoint
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prefill_decode_ratio = len(prompt_token_ids) / len(
            completion_token_ids)

        if (len(prompt_token_ids) < 32 or len(prompt_token_ids) > 4096):
            continue
        truncated_dataset.append({
            "prompt":
            prompt,
            "completion":
            completion,
            "prompt_tokens_length":
            len(prompt_token_ids),
            "completion_tokens_length":
            len(completion_token_ids),
            "prefill_to_decode_ratio":
            prefill_decode_ratio,
        })
    with open(f"truncated_dataset_32_4096.json", "w") as f:
        json.dump(truncated_dataset, f, indent=4)
    return


if __name__ == "__main__":
    tokenizer = get_tokenizer("hf-internal-testing/llama-tokenizer")
    prepare_truncated_dataset(
        dataset_path="datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
        tokenizer=tokenizer,
    )
