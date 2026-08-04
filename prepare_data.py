from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from config import DATASETS, DATA_DIR


def _read_numeric_csv(path: Path, label_column: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtypes: Dict[str, object] = {
        column: (np.int64 if column == label_column else np.float32)
        for column in header.columns
    }
    frame = pd.read_csv(path, dtype=dtypes)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return frame


def prepare_dataset(dataset: str, per_class: int = 6000, seed: int = 2026) -> Path:
    spec = DATASETS[dataset]
    csv_path = Path(spec["csv"])
    label_column = str(spec["label"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {csv_path}")

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"{dataset}_pc{per_class}_seed{seed}.npz"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        print(f"[cached] {output_path}")
        return output_path

    frame = _read_numeric_csv(csv_path, label_column)
    sampled = (
        frame.groupby(label_column, group_keys=False)
        .apply(lambda group: group.sample(n=min(per_class, len(group)), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    feature_names = [column for column in sampled.columns if column != label_column]
    x = sampled[feature_names].to_numpy(dtype=np.float32, copy=True)
    y_raw = sampled[label_column].to_numpy()
    classes = np.unique(y_raw)
    mapping = {int(value): index for index, value in enumerate(classes.tolist())}
    y = np.asarray([mapping[int(value)] for value in y_raw], dtype=np.int64)

    x_train, x_hold, y_train, y_hold = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_hold, y_hold, test_size=0.50, random_state=seed + 1, stratify=y_hold
    )

    scaler = RobustScaler(quantile_range=(5.0, 95.0), unit_variance=True)
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    x_train = np.clip(x_train, -12.0, 12.0)
    x_val = np.clip(x_val, -12.0, 12.0)
    x_test = np.clip(x_test, -12.0, 12.0)

    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        center=np.asarray(scaler.center_, dtype=np.float32),
        scale=np.asarray(scaler.scale_, dtype=np.float32),
        classes=classes,
        feature_names=np.asarray(feature_names),
    )

    metadata = {
        "dataset": dataset,
        "source_name": spec["name"],
        "source_csv": str(csv_path.resolve()),
        "label_column": label_column,
        "feature_names": feature_names,
        "class_mapping": {str(key): value for key, value in mapping.items()},
        "samples_per_class_requested": per_class,
        "split_sizes": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "seed": seed,
        "preprocessing": "RobustScaler(5th,95th), training split only; clipped to [-12,12]",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prepared] {output_path} | train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="edgeiiot")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--per-class", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    targets = tuple(DATASETS) if args.all else (args.dataset,)
    for dataset in targets:
        prepare_dataset(dataset, per_class=args.per_class, seed=args.seed)


if __name__ == "__main__":
    main()
