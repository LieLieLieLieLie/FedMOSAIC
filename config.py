from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"
PAPER_FIGURES_DIR = ROOT.parent / "paper" / "EAAI" / "figures"

# Keep paper-figure synchronization for the full local research workspace while
# making a standalone GitHub checkout self-contained.
if not PAPER_FIGURES_DIR.parent.exists():
    PAPER_FIGURES_DIR = FIGURES_DIR

for _path in (DATA_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR, LOGS_DIR, PAPER_FIGURES_DIR):
    _path.mkdir(parents=True, exist_ok=True)


DATASETS: Dict[str, Dict[str, object]] = {
    "edgeiiot": {
        "name": "Edge-IIoTset",
        "csv": DATA_DIR / "EdgeIIoTset" / "df_FL_Edge-IIoTset_6_classes.csv",
        "label": "mapped_attack_type",
    },
    "ciciot": {
        "name": "CICIoT2023",
        "csv": DATA_DIR / "CICIoT2023" / "df_FL_CICIoT2023_6_classes.csv",
        "label": "sub_label",
    },
}


METHODS: Tuple[str, ...] = (
    "fedavg_er",
    "glfc",
    "evofedids",
    "fedta",
    "fedagc",
    "fedmosaic",
)

METHOD_LABELS = {
    "fedavg_er": "FedAvg-ER",
    "glfc": "GLFC",
    "evofedids": "EvoFedIDS",
    "fedta": "FedTA",
    "fedagc": "FedAGC",
    "fedmosaic": "FedMOSAIC",
}

METHOD_COLORS = {
    "fedavg_er": "#FFAA53",
    "glfc": "#50CC55",
    "evofedids": "#3399FF",
    "fedta": "#6666FF",
    "fedagc": "#9933FF",
    "fedmosaic": "#FF6666",
}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "edgeiiot"
    method: str = "fedmosaic"
    variant: str = "full"
    seed: int = 0
    num_clients: int = 10
    participation: float = 0.6
    rounds: int = 25
    rounds_per_stage: int = 5
    max_staleness: int = 2
    dirichlet_alpha: float = 0.3
    feature_shift: float = 0.08
    local_steps: int = 5
    batch_size: int = 128
    learning_rate: float = 2.5e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 96
    embed_dim: int = 32
    dropout: float = 0.10
    replay_per_class: int = 24
    synthetic_per_class: int = 12
    anchor_weight: float = 0.9
    stability_weight: float = 0.20
    calibration_weight: float = 0.15
    proximal_weight: float = 0.01
    distill_weight: float = 0.7
    temperature: float = 0.20
    staleness_tau: float = 2.0
    server_momentum: float = 0.0
    server_consolidation_steps: int = 2
    server_learning_rate: float = 8e-4
    eval_every: int = 1
    device: str = "cuda"

    @property
    def run_id(self) -> str:
        suffix = "" if self.variant == "full" else f"__v{self.variant}"
        return (
            f"{self.dataset}__{self.method}__s{self.seed}__a{self.dirichlet_alpha:g}"
            f"__lag{self.max_staleness}__p{self.participation:g}"
            f"{suffix}"
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
