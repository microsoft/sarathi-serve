from collections.abc import Iterable
from itertools import islice
from typing import Optional

from transformers import AutoTokenizer

from datasets import load_dataset

DATASET_FIELDS = {
    "xsum": ("document", "train"),
    "openai_humaneval": ("prompt", "test"),
    "ccdv/arxiv-summarization": ("article", "train"),
    "lmsys/lmsys-chat-1m": ("conversation", "train"),
    "OpenGVLab/ShareGPT-4o": ("conversations", "image_caption"),
    "Fredithefish/ShareGPT-unfiltered-alpaca-lora-format": ("instruction", "train"),
    "openai/gsm8k": ("question", "main"),
    # 'another_dataset': 'text_field',
    # ... add other datasets and their respective fields
}


def get_data_loader(
    input_string: Optional[str], dataset_str: Optional[str], meta_prompt: Optional[str], max_samples: Optional[int],
        model_for_tokenizer_chat_template: Optional[str]
) -> Iterable[str]:
    if input_string:
        return [input_string]
    if dataset_str not in DATASET_FIELDS:
        # assume bwb path
        return islice(load_bwb(dataset_str, meta_prompt), max_samples)

    field_name = DATASET_FIELDS[dataset_str][0]
    split_name = DATASET_FIELDS[dataset_str][1]
    if dataset_str != "OpenGVLab/ShareGPT-4o":
        dataset = load_dataset(dataset_str, split=split_name)
    else:
        dataset = load_dataset(dataset_str, 'image_caption')

    if "conversation" not in field_name:
        sample_iterator = map(
            lambda x: f"{meta_prompt} {x}", map(lambda x: x[field_name], dataset)
        )
    else:
        # reformat sharegpt-4
        if "lmsys" not in dataset_str:
            dataset['images'] = dataset['images'].map(format_conversations)
            dataset = dataset['images']
        # don't use meta prompt, this is chat case
        tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer_chat_template)
        conversations = dataset[field_name]
        sample_iterator = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True,)

    if not max_samples:
        return sample_iterator

    return islice(sample_iterator, max_samples)


def format_conversations(sample):
    sample['conversations'] = [
        {
            "content": turn["value"],
            "role": "user" if turn["from"] == "human" else "assistant"
        }
        for turn in sample['conversations']
    ]
    return sample


def load_bwb(file_path, meta_prompt, chunk_size=1000):
    translation_chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                chunk = meta_prompt + chunk
                translation_chunks.append(chunk)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    return translation_chunks