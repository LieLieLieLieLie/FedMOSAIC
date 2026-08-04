from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import MODELS_DIR
from models import make_model


def is_main(meta: dict) -> bool:
    cfg = meta["config"]
    return (
        cfg.get("variant", "full") == "full"
        and cfg["dirichlet_alpha"] == 0.3
        and cfg["max_staleness"] == 2
        and cfg["participation"] == 0.6
        and cfg["seed"] in (0, 1, 2)
    )


@torch.no_grad()
def infer(checkpoint_path: Path, device: torch.device, overwrite: bool = False) -> None:
    output_path = checkpoint_path.with_suffix(".predictions.npz")
    if output_path.exists() and not overwrite:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    data_path = Path(__file__).resolve().parent / "data" / "processed" / f"{cfg['dataset']}_pc6000_seed2026.npz"
    data = np.load(data_path, allow_pickle=True)
    x = data["x_test"].astype(np.float32)
    y = data["y_test"].astype(np.int64)
    model = make_model(
        input_dim=x.shape[1],
        num_classes=len(np.unique(y)),
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probs, embeddings = [], []
    for start in range(0, len(x), 1024):
        xb = torch.from_numpy(x[start : start + 1024]).to(device)
        logits, features = model(xb, return_features=True)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        embeddings.append(features.cpu().numpy())
    np.savez_compressed(
        output_path,
        y_true=y,
        probabilities=np.concatenate(probs).astype(np.float32),
        embeddings=np.concatenate(embeddings).astype(np.float32),
    )
    print(f"[saved] {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in sorted(MODELS_DIR.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if is_main(meta):
            checkpoint_path = Path(meta["checkpoint"])
            if not checkpoint_path.exists():
                checkpoint_path = MODELS_DIR / f"{path.stem}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint referenced by {path.name} was not found at either "
                    f"{meta['checkpoint']} or {checkpoint_path}"
                )
            infer(checkpoint_path, device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
