import logging

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.config import ConfigParser
from sarathi.benchmark.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from sarathi.benchmark.utils.random import set_seeds


def main():
    config = ConfigParser().get_config()

    set_seeds(config.seed)

    log_level = getattr(logging, config.log_level.upper())
    logging.basicConfig(
        format=LOGGER_FORMAT, level=log_level, datefmt=LOGGER_TIME_FORMAT
    )

    runner = BenchmarkRunnerLauncher(config)
    runner.run()


if __name__ == "__main__":
    main()
