import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

from sarathi.benchmark.utils import data_loader
from sarathi.benchmark.config import DatasetRequestLengthGeneratorConfig
from sarathi.benchmark.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)

logger = logging.getLogger(__name__)


class DatasetRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(self, config: DatasetRequestLengthGeneratorConfig):
        super().__init__(config)
        self.next_request_idx = 0
        prompts = data_loader.get_data_loader(None, config.dataset, config.meta_prompt, None, config.tokenizer_model)
        self.requests = [prompt for prompt in prompts if len(prompt) <= config.max_prompt_len][:config.max_num_prompts]
        self.decode_tokens = config.max_decode_tokens

    def get_next_num_tokens(self) -> Tuple[Union[str|float], float]:
        if self.next_request_idx >= len(self.requests):
            return None, None

        row = self.requests[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row,
            self.decode_tokens,
        )