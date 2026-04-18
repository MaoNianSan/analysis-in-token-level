#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path

from config import (
    ROBUSTNESS_SEEDS,
    DATASET_PATHS,
    build_script_dirs,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    pipeline_dirs = build_script_dirs("main_pipeline")
    ensure_dirs(pipeline_dirs, keys=["logs"])
    pipeline_log_path = Path(pipeline_dirs["logs"]) / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(pipeline_log_path, mode="w"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full AD TF-IDF attribution pipeline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run downstream analysis with one seed only.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 7,42,123.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets to run, chosen from: test, ad, control.",
    )
    return parser.parse_args()


def resolve_seeds(args):
    if args.seed is not None:
        return [args.seed]
    if args.seeds:
        resolved = []
        for item in args.seeds.split(","):
            item = item.strip()
            if not item:
                continue
            resolved.append(int(item))
        if resolved:
            return resolved
    return list(ROBUSTNESS_SEEDS)


def resolve_datasets(args):
    if not args.datasets:
        return dict(DATASET_PATHS)

    names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    resolved = {}
    for name in names:
        if name not in DATASET_PATHS:
            raise ValueError(
                f"Unknown dataset '{name}'. Available datasets: {list(DATASET_PATHS.keys())}"
            )
        resolved[name] = DATASET_PATHS[name]
    return resolved


def build_subprocess_env(seed: int, seeds: list[int]) -> dict:
    env = os.environ.copy()
    env["RANDOM_SEED"] = str(seed)
    env["ROBUSTNESS_SEEDS"] = ",".join(str(s) for s in seeds)
    return env


def run_script_direct(
    script_name: str, description: str, env: dict | None = None
) -> bool:
    logger.info(f"Starting stage: {description}")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    return result.returncode == 0


def run_seeded_stage(
    script_name: str,
    description: str,
    seed: int,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    output_with_seed: bool = True,
    seeds: list[int] | None = None,
) -> bool:
    module_name = Path(script_name).stem
    env = build_subprocess_env(seed, seeds or [seed])

    if dataset_name is None or dataset_path is None:
        inline_code = (
            f"import {module_name}; "
            f"{module_name}.main("
            f"seed={seed}, "
            f"output_with_seed={output_with_seed}"
            f")"
        )
    else:
        inline_code = (
            f"import {module_name}; "
            f"{module_name}.main("
            f"seed={seed}, "
            f"output_with_seed={output_with_seed}, "
            f"dataset_name='{dataset_name}', "
            f"dataset_path=r'{dataset_path}'"
            f")"
        )

    logger.info(f"Stage: {description} | seed={seed} | dataset={dataset_name}")

    result = subprocess.run(
        [sys.executable, "-c", inline_code],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    return result.returncode == 0


def run_seed_pipeline(
    seed: int, dataset_name: str, dataset_path: str, seeds: list[int]
) -> bool:
    logger.info("=" * 60)
    logger.info(f"Seed={seed} | Dataset={dataset_name}")
    logger.info("=" * 60)

    stages = [
        ("window_extraction.py", "Window extraction"),
        ("token_aggregation.py", "Token aggregation"),
        ("debug_stability.py", "Stability diagnostics"),
        ("visualization.py", "Visualization"),
    ]

    success = True
    for script_name, description in stages:
        ok = run_seeded_stage(
            script_name=script_name,
            description=description,
            seed=seed,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            output_with_seed=True,
            seeds=seeds,
        )
        if not ok:
            logger.warning(
                f"Failed stage: {description} | seed={seed} | dataset={dataset_name}"
            )
            success = False

    return success


def main():
    setup_logging()
    args = parse_args()
    seeds = resolve_seeds(args)
    datasets = resolve_datasets(args)

    logger.info("Pipeline start")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Datasets: {list(datasets.keys())}")

    all_success = True

    preprocess_env = build_subprocess_env(seeds[0], seeds)

    if not run_script_direct("preprocess_data.py", "Preprocess", env=preprocess_env):
        logger.warning("Preprocess failed")
        all_success = False

    if not run_seeded_stage(
        script_name="train_classifier.py",
        description="Classifier",
        seed=seeds[0],
        output_with_seed=False,
        seeds=seeds,
    ):
        logger.warning("Classifier failed")
        all_success = False

    for seed in seeds:
        for dataset_name, dataset_path in datasets.items():
            success = run_seed_pipeline(seed, dataset_name, dataset_path, seeds)
            if not success:
                all_success = False

    logger.info("=" * 60)
    logger.info("Pipeline finished")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
