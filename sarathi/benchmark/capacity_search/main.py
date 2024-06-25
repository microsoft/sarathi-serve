"""
    Automated search for capacity for different systems via latency vs qps data.
    A system is characterized by:
    1. trace
    2. model
    3. parallel spec
    4. scheduler
"""

import argparse
import json
import os
import time

import wandb
import yaml

from sarathi.benchmark.capacity_search.search_manager import SearchManager
from sarathi.logger import init_logger

logger = init_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-search-granularity",
        type=float,
        default=2.5,
        help="Minimum search granularity for capacity (%)",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--scheduling-delay-slo-value", type=float, default=2.0)
    parser.add_argument("--scheduling-delay-slo-quantile", type=float, default=0.50)
    parser.add_argument("--tbt-slo-value", type=float, default=0.2)
    parser.add_argument("--tbt-slo-quantile", type=float, default=0.99)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--time-limit", type=int, default=30, help="Time limit in minutes"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug logs and commands"
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-sweep-name", type=str, default=None)
    parser.add_argument("--wandb-sweep-id", type=str, default=None)

    args = parser.parse_args()

    if args.wandb_project:
        assert (
            args.wandb_sweep_name or args.wandb_sweep_id
        ), "wandb-sweep-name/id is required with wandb-project"

    return args


if __name__ == "__main__":
    args = get_args()

    config = yaml.safe_load(open(args.config_path))

    assert (
        args.scheduling_delay_slo_quantile >= 0
        and args.scheduling_delay_slo_quantile <= 1
        and args.tbt_slo_quantile >= 0
        and args.tbt_slo_quantile <= 1
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting capacity search")

    # merge the config with the args
    config.update(vars(args))
    logger.info(f"Config: {config}")

    # store the config and args
    json.dump(config, open(f"{args.output_dir}/config.json", "w"))

    if args.wandb_project and not args.wandb_sweep_id:
        config["name"] = args.wandb_sweep_name
        config["method"] = "custom"

        sweep_id = wandb.sweep(config, project=args.wandb_project)
        args.wandb_sweep_id = sweep_id
        # required so that wandb doesn't delay flush of child logs
        wandb.finish(quiet=True)

    search_manager = SearchManager(args, config)

    start_time = time.time()

    all_results = search_manager.run()

    end_time = time.time()

    logger.info(f"Benchmarking took time: {end_time - start_time}")
