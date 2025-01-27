from abc import ABC, abstractmethod
from typing import Tuple, Union

from sarathi.benchmark.config import BaseRequestLengthGeneratorConfig


class BaseRequestLengthGenerator(ABC):

    def __init__(self, config: BaseRequestLengthGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_num_tokens(self) -> Tuple[Union[str|float], float]:
        pass
