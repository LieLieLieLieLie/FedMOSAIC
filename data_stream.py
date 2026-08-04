from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from config import DATA_DIR, ExperimentConfig


def load_processed(dataset: str, per_class: int = 6000, seed: int = 2026) -> Dict[str, np.ndarray]:
    path = DATA_DIR / "processed" / f"{dataset}_pc{per_class}_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first; missing {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 64,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    for _ in range(200):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            indices = np.flatnonzero(labels == class_id)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            # Damp already-large clients to avoid empty edge sites.
            capacity = np.asarray([len(bucket) < len(labels) / num_clients for bucket in buckets])
            proportions = proportions * capacity
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client_id, split in enumerate(np.split(indices, cuts)):
                buckets[client_id].extend(split.tolist())
        if min(map(len, buckets)) >= min_size:
            return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]
    raise RuntimeError("Could not construct a non-empty Dirichlet partition")


@dataclass
class ClientSchedule:
    stages: List[np.ndarray]
    class_order: List[int]
    start_delay: int


class FederatedContinualStream:
    """Deterministic task-free simulator with hidden client-specific stages."""

    def __init__(self, config: ExperimentConfig, per_class: int = 6000) -> None:
        arrays = load_processed(config.dataset, per_class=per_class)
        self.x_train = arrays["x_train"].astype(np.float32, copy=False)
        self.y_train = arrays["y_train"].astype(np.int64, copy=False)
        self.x_val = arrays["x_val"].astype(np.float32, copy=False)
        self.y_val = arrays["y_val"].astype(np.int64, copy=False)
        self.x_test = arrays["x_test"].astype(np.float32, copy=False)
        self.y_test = arrays["y_test"].astype(np.int64, copy=False)
        self.num_features = int(self.x_train.shape[1])
        self.num_classes = int(self.y_train.max()) + 1
        self.num_clients = config.num_clients
        self.rounds_per_stage = config.rounds_per_stage
        self.max_staleness = config.max_staleness
        self.feature_shift = config.feature_shift
        self.rng = np.random.default_rng(config.seed + 91)

        partitions = dirichlet_partition(
            self.y_train,
            config.num_clients,
            config.dirichlet_alpha,
            config.seed + 17,
        )
        self.schedules = self._build_schedules(partitions, config.seed + 31)
        self.client_shift = self.rng.normal(
            0.0, config.feature_shift, size=(config.num_clients, self.num_features)
        ).astype(np.float32)
        self.client_scale = np.exp(
            self.rng.normal(0.0, config.feature_shift / 2, size=(config.num_clients, self.num_features))
        ).astype(np.float32)

        self.test_partitions = dirichlet_partition(
            self.y_test,
            config.num_clients,
            max(config.dirichlet_alpha, 0.4),
            config.seed + 53,
            min_size=16,
        )

    def _build_schedules(self, partitions: Sequence[np.ndarray], seed: int) -> List[ClientSchedule]:
        rng = np.random.default_rng(seed)
        schedules: List[ClientSchedule] = []
        common_class = 0
        attack_classes = list(range(1, self.num_classes))
        for client_id, indices in enumerate(partitions):
            order = attack_classes.copy()
            rng.shuffle(order)
            common_indices = indices[self.y_train[indices] == common_class]
            common_chunks = np.array_split(common_indices, len(order))
            stages: List[np.ndarray] = []
            for stage_id, class_id in enumerate(order):
                class_indices = indices[self.y_train[indices] == class_id]
                stage = np.concatenate([common_chunks[stage_id], class_indices])
                if len(stage) == 0:
                    # Extreme Dirichlet tails can miss both scheduled classes. The learner is
                    # task-free, so the site's available local stream remains a valid fallback.
                    stage = indices
                rng.shuffle(stage)
                stages.append(stage.astype(np.int64, copy=False))
            schedules.append(
                ClientSchedule(
                    stages=stages,
                    class_order=order,
                    start_delay=int(rng.integers(0, self.max_staleness + 1)),
                )
            )
        return schedules

    @property
    def num_stages(self) -> int:
        return self.num_classes - 1

    def stage_at_round(self, client_id: int, round_id: int) -> int:
        delay = self.schedules[client_id].start_delay
        effective = max(0, round_id - delay)
        return min(effective // self.rounds_per_stage, self.num_stages - 1)

    def active_classes(self, client_id: int, round_id: int) -> Tuple[int, ...]:
        stage_id = self.stage_at_round(client_id, round_id)
        labels = np.unique(self.y_train[self.schedules[client_id].stages[stage_id]])
        return tuple(int(value) for value in labels)

    def sample_batch(
        self,
        client_id: int,
        round_id: int,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        stage_id = self.stage_at_round(client_id, round_id)
        candidates = self.schedules[client_id].stages[stage_id]
        if len(candidates) == 0:
            raise RuntimeError(f"Client {client_id} has an empty stream stage")
        chosen = rng.choice(candidates, size=batch_size, replace=len(candidates) < batch_size)
        x = self.x_train[chosen] * self.client_scale[client_id] + self.client_shift[client_id]
        y = self.y_train[chosen]
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device),
            chosen,
        )

    def tensors(self, split: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        x = getattr(self, f"x_{split}")
        y = getattr(self, f"y_{split}")
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device),
        )
