from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # ── data ─────────────────────────────────────────────────────────────────
    data_dir:   Path = Path("tokenization_outputs/ver1")
    output_dir: Path = Path("model_outputs/run1")

    # ── training loop ─────────────────────────────────────────────────────────
    epochs:       int   = 20
    batch_size:   int   = 32
    lr:           float = 1e-4
    weight_decay: float = 1e-2

    # ── model architecture ────────────────────────────────────────────────────
    d_model:    int   = 128
    num_heads:  int   = 4
    num_layers: int   = 4
    ff_dim:     int   = 512
    dropout:    float = 0.1

    # ── runtime ───────────────────────────────────────────────────────────────
    num_workers: int = 0
    device:      str = "auto"   # "auto" | "cpu" | "cuda" | "mps"
    seed:        int = 42

    # ── experiment tracking ───────────────────────────────────────────────────
    use_wandb:     bool       = False
    wandb_project: str        = "mimic-cardio-oncology"
    run_name:      str | None = None   # defaults to wandb auto-generated name

    # ── ablations ─────────────────────────────────────────────────────────────
    # CEHR-BERT sinusoidal time embedding: sin((days_since_2000 / 365.25) * w + φ)
    # where w and φ are learned per embedding dimension.
    # Requires dates.pt in data_dir — re-run tokenization to generate it.
    use_time_embedding: bool = False

    # ── serialization ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"]   = str(d["data_dir"])
        d["output_dir"] = str(d["output_dir"])
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        with open(path) as f:
            d = json.load(f)
        d["data_dir"]   = Path(d["data_dir"])
        d["output_dir"] = Path(d["output_dir"])
        return cls(**d)
