from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import LOGS_DIR, MODELS_DIR, ExperimentConfig
from data_stream import FederatedContinualStream
from metrics import classification_metrics
from methods import LocalResult, make_client_states, train_client
from models import (
    SimplexStatistics,
    clone_model,
    flatten_delta,
    make_model,
    trainable_numel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _predict(
    model: nn.Module,
    x: torch.Tensor,
    server_stats: SimplexStatistics,
    method: str,
    calibration_weight: float,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    probabilities: List[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        logits, features = model(x[start : start + batch_size], return_features=True)
        if method == "fedmosaic":
            logits = server_stats.calibrated_logits(
                logits,
                features,
                anchor_scale=5.5,
                calibration_weight=calibration_weight,
            )
        elif method == "fedta" and server_stats.active.any():
            logits = server_stats.calibrated_logits(
                logits,
                features,
                anchor_scale=4.0,
                calibration_weight=0.10,
            )
        probabilities.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probabilities, dim=0).numpy()


def _stable_basis(update_history: Sequence[torch.Tensor], rank: int = 4) -> Optional[torch.Tensor]:
    if not update_history:
        return None
    matrix = torch.stack(list(update_history), dim=1)
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-8)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, : min(rank, q.shape[1])]


def _server_consolidate(
    model: nn.Module,
    server_stats: SimplexStatistics,
    config: ExperimentConfig,
    round_id: int,
) -> None:
    if config.server_consolidation_steps <= 0 or config.variant == "no_stats":
        return
    generator = torch.Generator(device=server_stats.anchors.device)
    generator.manual_seed(config.seed * 10000 + round_id + 811)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.server_learning_rate, weight_decay=config.weight_decay
    )
    model.train()
    for _ in range(config.server_consolidation_steps):
        synthetic = server_stats.sample_inputs(24, generator)
        if synthetic is None:
            return
        x_synthetic, y_synthetic = synthetic
        optimizer.zero_grad(set_to_none=True)
        logits, features = model(x_synthetic, return_features=True)
        normalized = F.normalize(features, dim=1)
        anchor_logits = normalized @ server_stats.anchors.T / config.temperature
        loss = F.cross_entropy(logits + 0.25 * anchor_logits, y_synthetic)
        if config.variant != "no_simplex":
            loss = loss + 0.35 * F.cross_entropy(anchor_logits, y_synthetic)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        optimizer.step()


@torch.no_grad()
def _aggregate(
    model: nn.Module,
    results: Sequence[LocalResult],
    method: str,
    velocity: Optional[Dict[str, torch.Tensor]],
    momentum: float,
    reliability_weighted: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if not results:
        raise ValueError("At least one client result is required")
    if method == "fedmosaic" and reliability_weighted:
        raw_weights = torch.tensor(
            [result.samples * result.reliability for result in results],
            device=next(model.parameters()).device,
        )
    else:
        raw_weights = torch.tensor(
            [result.samples for result in results],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    weights = raw_weights / raw_weights.sum().clamp_min(1e-8)
    state = model.state_dict()
    aggregated: Dict[str, torch.Tensor] = {}
    new_velocity: Dict[str, torch.Tensor] = {}
    new_state: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        update = sum(weight * result.delta[name] for weight, result in zip(weights, results))
        if not torch.is_floating_point(tensor):
            new_state[name] = tensor
            new_velocity[name] = torch.zeros_like(tensor, dtype=torch.float32)
            continue
        previous = torch.zeros_like(update) if velocity is None else velocity[name]
        current_velocity = momentum * previous + (1.0 - momentum) * update
        aggregated[name] = current_velocity
        new_velocity[name] = current_velocity
        new_state[name] = tensor + current_velocity
    model.load_state_dict(new_state)
    return aggregated, new_velocity


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def run_experiment(config: ExperimentConfig, overwrite: bool = False, verbose: bool = True) -> Path:
    history_path = MODELS_DIR / f"{config.run_id}.json"
    checkpoint_path = MODELS_DIR / f"{config.run_id}.pt"
    if history_path.exists() and checkpoint_path.exists() and not overwrite:
        if verbose:
            print(f"[cached] {config.run_id}")
        return history_path

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    stream = FederatedContinualStream(config)
    model = make_model(
        stream.num_features,
        stream.num_classes,
        config.hidden_dim,
        config.embed_dim,
        config.dropout,
        device,
    )
    server_stats = SimplexStatistics(
        stream.num_classes, config.embed_dim, device, input_dim=stream.num_features
    )
    client_states = make_client_states(config)
    model_history: List[nn.Module] = [clone_model(model).to(device).eval()]
    update_history: deque[torch.Tensor] = deque(maxlen=6)
    velocity: Optional[Dict[str, torch.Tensor]] = None
    x_test, y_test = stream.tensors("test", device)
    y_test_np = y_test.cpu().numpy()
    rng = np.random.default_rng(config.seed + 707)
    records: List[Dict[str, object]] = []
    stage_matrix: List[List[float]] = []
    cumulative_communication = 0
    peak_auxiliary_memory = 0
    start_time = time.perf_counter()

    for round_id in range(config.rounds):
        participants = max(1, int(math.ceil(config.num_clients * config.participation)))
        selected = np.sort(rng.choice(config.num_clients, size=participants, replace=False))
        basis = _stable_basis(tuple(update_history)) if config.method == "fedmosaic" else None
        results: List[LocalResult] = []
        stalenesses: List[int] = []
        for client_id_np in selected:
            client_id = int(client_id_np)
            staleness = int(rng.integers(0, min(config.max_staleness, round_id) + 1))
            stale_index = max(0, round_id - staleness)
            stale_model = model_history[stale_index]
            result = train_client(
                config.method,
                model,
                stale_model,
                stream,
                client_states[client_id],
                config,
                server_stats,
                basis,
                client_id,
                round_id,
                staleness,
            )
            results.append(result)
            stalenesses.append(staleness)

        momentum = config.server_momentum if config.method == "fedmosaic" else 0.0
        aggregate_delta, velocity = _aggregate(
            model,
            results,
            config.method,
            velocity,
            momentum,
            reliability_weighted=config.variant != "uniform_fusion",
        )
        flat_update = flatten_delta(aggregate_delta).detach()
        if flat_update.norm() > 1e-10:
            update_history.append(flat_update)

        if config.method in {"fedta", "fedmosaic"}:
            for result in results:
                server_stats.update(
                    result.class_ids,
                    result.feature_means,
                    result.feature_variances,
                    result.feature_counts,
                    result.input_means,
                    result.input_variances,
                    result.input_counts,
                )
        if config.method == "fedmosaic":
            _server_consolidate(model, server_stats, config, round_id)

        model_history.append(clone_model(model).to(device).eval())
        round_communication = sum(result.communication_bytes for result in results)
        cumulative_communication += round_communication
        peak_auxiliary_memory = max(
            peak_auxiliary_memory,
            sum(result.auxiliary_memory_bytes for result in results),
        )

        if (round_id + 1) % config.eval_every == 0 or round_id == config.rounds - 1:
            probabilities = _predict(
                model,
                x_test,
                server_stats,
                config.method,
                config.calibration_weight,
            )
            metrics = classification_metrics(y_test_np, probabilities)
            client_scores: List[float] = []
            for partition in stream.test_partitions:
                if len(partition) == 0:
                    continue
                client_scores.append(
                    float(
                        classification_metrics(y_test_np[partition], probabilities[partition])["macro_f1"]
                    )
                )
            record: Dict[str, object] = {
                "round": round_id + 1,
                **metrics,
                "train_loss": float(np.mean([result.train_loss for result in results])),
                "mean_reliability": float(np.mean([result.reliability for result in results])),
                "mean_staleness": float(np.mean(stalenesses)),
                "selected_clients": selected.tolist(),
                "client_macro_f1": client_scores,
                "communication_mb": cumulative_communication / (1024**2),
                "peak_auxiliary_memory_mb": peak_auxiliary_memory / (1024**2),
                "drift_events": int(sum(state.drift_events for state in client_states)),
            }
            records.append(record)
            if (round_id + 1) % config.rounds_per_stage == 0:
                stage_matrix.append([float(value) for value in metrics["per_class_recall"]])
            if verbose:
                print(
                    f"[{config.dataset}/{config.method}/s{config.seed}] "
                    f"r={round_id + 1:02d} F1={metrics['macro_f1']:.4f} "
                    f"BAcc={metrics['balanced_accuracy']:.4f} ECE={metrics['ece']:.4f}"
                )

    elapsed = time.perf_counter() - start_time
    recall_trajectory = np.asarray([record["per_class_recall"] for record in records], dtype=float)
    final_recall = recall_trajectory[-1]
    forgetting_per_class = np.nanmax(recall_trajectory, axis=0) - final_recall
    bwt_per_class = final_recall - recall_trajectory[0]
    final_metrics = dict(records[-1])
    final_metrics.update(
        {
            "average_forgetting": float(np.mean(forgetting_per_class)),
            "backward_transfer": float(np.mean(bwt_per_class)),
            "wall_time_seconds": float(elapsed),
            "model_parameters": trainable_numel(model),
            "communication_mb": float(cumulative_communication / (1024**2)),
            "peak_auxiliary_memory_mb": float(peak_auxiliary_memory / (1024**2)),
        }
    )
    payload = {
        "run_id": config.run_id,
        "config": config.to_dict(),
        "dataset": config.dataset,
        "method": config.method,
        "seed": config.seed,
        "records": records,
        "stage_matrix": stage_matrix,
        "forgetting_per_class": forgetting_per_class.tolist(),
        "bwt_per_class": bwt_per_class.tolist(),
        "final": final_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
    }
    history_path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=True), encoding="utf-8"
    )
    torch.save(
        {
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "anchors": server_stats.anchors.detach().cpu(),
            "anchor_means": server_stats.means.detach().cpu(),
            "anchor_variances": server_stats.variances.detach().cpu(),
            "anchor_counts": server_stats.counts.detach().cpu(),
            "anchor_active": server_stats.active.detach().cpu(),
            "input_means": server_stats.input_means.detach().cpu(),
            "input_variances": server_stats.input_variances.detach().cpu(),
            "input_counts": server_stats.input_counts.detach().cpu(),
            "records": records,
        },
        checkpoint_path,
    )
    if verbose:
        print(f"[saved] {history_path.name} ({elapsed:.1f} s)")
    return history_path
