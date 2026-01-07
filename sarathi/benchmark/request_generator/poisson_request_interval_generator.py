import math
import random

from sarathi.benchmark.config import PoissonRequestIntervalGeneratorConfig
from sarathi.benchmark.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRequestIntervalGenerator(BaseRequestIntervalGenerator):

    def __init__(self, config: PoissonRequestIntervalGeneratorConfig):
        super().__init__(config)

        self.qps = self.config.qps
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 8.0
        self.count = 0

    def get_next_inter_request_time(self) -> float:
        next_interval = -math.log(1.0 - random.random()) / self.qps
        next_interval = min(next_interval, self.max_interval)

        return next_interval

    
    def get_next_inter_request_time_fix(self) -> float:
        if self.count == 0:
            self.count = 1
            return 0
        return 1 / self.qps
        