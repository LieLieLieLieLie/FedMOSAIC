from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from config import METHOD_LABELS, METHODS, MODELS_DIR, TABLES_DIR


DATASET_LABELS = {"edgeiiot": "Edge-IIoTset", "ciciot": "CICIoT2023"}
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_auroc",
    "ece",
    "average_forgetting",
    "backward_transfer",
    "communication_mb",
    "peak_auxiliary_memory_mb",
    "wall_time_seconds",
)


def load_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for path in sorted(MODELS_DIR.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "final" in item and "config" in item:
            item["_path"] = str(path)
            runs.append(item)
    return runs


def is_main(run: Dict[str, object]) -> bool:
    cfg = run["config"]
    return (
        cfg["variant"] == "full"
        and cfg["dirichlet_alpha"] == 0.3
        and cfg["max_staleness"] == 2
        and cfg["participation"] == 0.6
        and cfg["seed"] in (0, 1, 2)
    )


def rows_from(runs: Iterable[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        cfg, final = run["config"], run["final"]
        row = {
            "dataset": cfg["dataset"],
            "dataset_label": DATASET_LABELS[cfg["dataset"]],
            "method": cfg["method"],
            "method_label": METHOD_LABELS[cfg["method"]],
            "seed": cfg["seed"],
            "variant": cfg.get("variant", "full"),
            "alpha": cfg["dirichlet_alpha"],
            "staleness": cfg["max_staleness"],
            "participation": cfg["participation"],
        }
        for metric in METRICS:
            row[metric] = final.get(metric, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(value: pd.Series, digits: int = 3) -> str:
    return f"{value.mean():.{digits}f} ± {value.std(ddof=1):.{digits}f}"


def export_main(frame: pd.DataFrame) -> None:
    main = frame[
        (frame.variant == "full")
        & (frame.alpha == 0.3)
        & (frame.staleness == 2)
        & (frame.participation == 0.6)
        & (frame.seed.isin([0, 1, 2]))
    ].copy()
    main.to_csv(TABLES_DIR / "main_runs.csv", index=False)
    grouped = main.groupby(["dataset", "dataset_label", "method", "method_label"], sort=False)
    records = []
    for keys, group in grouped:
        record = dict(zip(("dataset", "dataset_label", "method", "method_label"), keys))
        for metric in METRICS:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1)
            record[metric] = mean_std(group[metric])
        records.append(record)
    summary = pd.DataFrame(records)
    order = {method: idx for idx, method in enumerate(METHODS)}
    summary["_order"] = summary.method.map(order)
    summary = summary.sort_values(["dataset", "_order"]).drop(columns="_order")
    summary.to_csv(TABLES_DIR / "main_results.csv", index=False)

    # Compact manuscript-ready table. Lower is better for ECE/forgetting/resources.
    compact = summary[[
        "dataset_label", "method_label", "macro_f1", "balanced_accuracy",
        "macro_auroc", "ece", "average_forgetting", "communication_mb",
        "peak_auxiliary_memory_mb",
    ]].copy()
    compact.columns = [
        "Dataset", "Method", "Macro-F1", "Balanced accuracy", "AUROC",
        "ECE", "Forgetting", "Comm. (MB)", "Aux. memory (MB)",
    ]
    compact.to_csv(TABLES_DIR / "main_results_manuscript.csv", index=False)


def export_ablation(frame: pd.DataFrame) -> None:
    variants = ["full", "no_stats", "no_simplex", "no_subspace", "uniform_fusion", "no_calibration"]
    labels = {
        "full": "FedMOSAIC",
        "no_stats": "w/o statistical consolidation",
        "no_simplex": "w/o simplex anchoring",
        "no_subspace": "w/o stable-plastic projection",
        "uniform_fusion": "w/o reliability-staleness fusion",
        "no_calibration": "w/o incremental calibration",
    }
    ab = frame[
        (frame.method == "fedmosaic")
        & (frame.variant.isin(variants))
        & (frame.alpha == 0.3)
        & (frame.staleness == 2)
        & (frame.participation == 0.6)
        & (frame.seed.isin([0, 1, 2]))
    ].copy()
    ab["variant_label"] = ab.variant.map(labels)
    ab.to_csv(TABLES_DIR / "ablation_runs.csv", index=False)
    out = (
        ab.groupby(["dataset_label", "variant", "variant_label"], sort=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"), ece_std=("ece", "std"),
            forgetting_mean=("average_forgetting", "mean"),
            forgetting_std=("average_forgetting", "std"),
            bwt_mean=("backward_transfer", "mean"), bwt_std=("backward_transfer", "std"),
        )
        .reset_index()
    )
    out["_order"] = out.variant.map({v: i for i, v in enumerate(variants)})
    out = out.sort_values(["dataset_label", "_order"]).drop(columns="_order")
    out.to_csv(TABLES_DIR / "ablation_results.csv", index=False)


def export_stress(frame: pd.DataFrame) -> None:
    stress = frame[
        (frame.seed == 0)
        & (frame.variant == "full")
        & (
            ((frame.alpha.isin([0.1, 0.3, 0.6, 1.0])) & (frame.staleness == 2) & (frame.participation == 0.6))
            | ((frame.alpha == 0.3) & (frame.staleness.isin([0, 1, 2, 3, 4])) & (frame.participation == 0.6))
            | ((frame.alpha == 0.3) & (frame.staleness == 2) & (frame.participation.isin([0.3, 0.6, 0.8, 1.0])))
        )
    ].drop_duplicates(subset=["dataset", "method", "seed", "variant", "alpha", "staleness", "participation"])
    stress.to_csv(TABLES_DIR / "stress_results.csv", index=False)


def export_sensitivity(frame: pd.DataFrame) -> None:
    sens = frame[(frame.method == "fedmosaic") & frame.variant.str.startswith("sens_")].copy()
    sens.to_csv(TABLES_DIR / "sensitivity_results.csv", index=False)


def export_significance(frame: pd.DataFrame) -> None:
    main = frame[
        (frame.variant == "full") & (frame.alpha == 0.3) & (frame.staleness == 2)
        & (frame.participation == 0.6) & (frame.seed.isin([0, 1, 2]))
    ]
    rows = []
    for dataset in DATASET_LABELS:
        proposal = main[(main.dataset == dataset) & (main.method == "fedmosaic")].sort_values("seed")
        for baseline in METHODS[:-1]:
            other = main[(main.dataset == dataset) & (main.method == baseline)].sort_values("seed")
            merged = proposal[["seed", "macro_f1"]].merge(
                other[["seed", "macro_f1"]], on="seed", suffixes=("_ours", "_base")
            )
            diff = merged.macro_f1_ours - merged.macro_f1_base
            try:
                w_p = float(wilcoxon(merged.macro_f1_ours, merged.macro_f1_base, alternative="greater").pvalue)
            except ValueError:
                w_p = 1.0
            rows.append({
                "dataset": DATASET_LABELS[dataset],
                "comparison": f"FedMOSAIC vs. {METHOD_LABELS[baseline]}",
                "mean_f1_gain": diff.mean(),
                "cohen_dz": diff.mean() / (diff.std(ddof=1) + 1e-12),
                "paired_t_p": float(ttest_rel(merged.macro_f1_ours, merged.macro_f1_base, alternative="greater").pvalue),
                "wilcoxon_p": w_p,
            })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "significance_results.csv", index=False)


def main() -> None:
    frame = rows_from(load_runs())
    if frame.empty:
        raise RuntimeError("No completed result JSON files were found.")
    export_main(frame)
    export_ablation(frame)
    export_stress(frame)
    export_sensitivity(frame)
    export_significance(frame)
    print(f"Exported result tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
