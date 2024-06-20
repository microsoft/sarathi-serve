import logging
import os

import yaml

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from sarathi.benchmark.utils.random import set_seeds
from sarathi.logger import init_logger

logger = init_logger(__name__)


def main() -> None:
    config = BenchmarkConfig.create_from_cli_args()

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    logger.info(f"Starting benchmark with config: {config}")

    set_seeds(config.seed)

    log_level = getattr(logging, config.log_level.upper())
    logging.basicConfig(
        format=LOGGER_FORMAT, level=log_level, datefmt=LOGGER_TIME_FORMAT
    )

    runner = BenchmarkRunnerLauncher(config)
    runner.run()


if __name__ == "__main__":
    main()
