from collections.abc import Iterable
from itertools import islice
from typing import Optional

from transformers import AutoTokenizer

from datasets import load_dataset

DATASET_FIELDS = {
    "xsum": "document",
    "openai_humaneval": "prompt",
    "ccdv/arxiv-summarization": "article",
    "lmsys/lmsys-chat-1m": "conversation",
    "Fredithefish/ShareGPT-unfiltered-alpaca-lora-format": "instruction"
    # 'another_dataset': 'text_field',
    # ... add other datasets and their respective fields
}

def get_data_loader(
    dataset_str: Optional[str], max_samples: Optional[int]
) -> Iterable[str]:

    dataset = load_dataset(dataset_str, split='train')
    field_name = DATASET_FIELDS[dataset_str]

    if "conversation" not in field_name:
        sample_iterator = map(
            lambda x: x, map(lambda x: x[field_name], dataset)
        )
    else:
        # assume llama3
        # don't use meta prompt, this is chat case
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        conversations = dataset[field_name]
        sample_iterator = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True,)

    if not max_samples:
        return sample_iterator

    return islice(sample_iterator, max_samples)
