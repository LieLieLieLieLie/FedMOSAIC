from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable, List

from config import DATASETS, METHODS, ExperimentConfig
from federated import run_experiment


def main_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(dataset=dataset, method=method, seed=seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in (0, 1, 2)
    ]


def stress_configs() -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for method in METHODS:
            for lag in (0, 1, 3, 4):
                configs.append(
                    ExperimentConfig(dataset=dataset, method=method, seed=0, max_staleness=lag)
                )
            for alpha in (0.1, 0.6, 1.0):
                configs.append(
                    ExperimentConfig(dataset=dataset, method=method, seed=0, dirichlet_alpha=alpha)
                )
            for participation in (0.3, 0.8, 1.0):
                configs.append(
                    ExperimentConfig(
                        dataset=dataset,
                        method=method,
                        seed=0,
                        participation=participation,
                    )
                )
    return configs


def ablation_configs() -> List[ExperimentConfig]:
    variants = ("no_stats", "no_simplex", "no_subspace", "uniform_fusion", "no_calibration")
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for seed in (0, 1, 2):
            for variant in variants:
                kwargs = {"calibration_weight": 0.0} if variant == "no_calibration" else {}
                configs.append(
                    ExperimentConfig(
                        dataset=dataset,
                        method="fedmosaic",
                        seed=seed,
                        variant=variant,
                        **kwargs,
                    )
                )
    return configs


def sensitivity_configs() -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for dataset in DATASETS:
        for weight in (0.3, 0.6, 1.2, 1.5):
            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    method="fedmosaic",
                    seed=0,
                    variant=f"sens_anchor_{weight:g}",
                    anchor_weight=weight,
                )
            )
        for weight in (0.0, 0.1, 0.4, 0.6):
            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    method="fedmosaic",
                    seed=0,
                    variant=f"sens_stability_{weight:g}",
                    stability_weight=weight,
                )
            )
    return configs


def execute(configs: Iterable[ExperimentConfig], overwrite: bool) -> None:
    configs = list(configs)
    for index, config in enumerate(configs, start=1):
        print(f"\n=== Run {index}/{len(configs)}: {config.run_id} ===")
        run_experiment(config, overwrite=overwrite, verbose=False)
        print(f"[complete] {config.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("main", "stress", "ablation", "sensitivity", "all"),
        default="main",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    suites = {
        "main": main_configs,
        "stress": stress_configs,
        "ablation": ablation_configs,
        "sensitivity": sensitivity_configs,
    }
    selected = tuple(suites) if args.suite == "all" else (args.suite,)
    for suite in selected:
        print(f"\n##### SUITE: {suite} #####")
        execute(suites[suite](), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
