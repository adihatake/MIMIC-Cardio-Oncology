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
    epochs:          int   = 20
    batch_size:      int   = 32
    lr:              float = 1e-4
    weight_decay:    float = 1e-2
    label_smoothing: float = 0.0

    # ── model selection ───────────────────────────────────────────────────────
    # "transformer" → EHR_Encoder   (BERT-style, O(L²) attention)
    # "mamba"       → EHR_Mamba     (bidirectional SSM, O(L) recurrence)
    model_type: str = "transformer"

    # ── transformer architecture ──────────────────────────────────────────────
    d_model:    int   = 128
    num_heads:  int   = 4
    num_layers: int   = 4
    ff_dim:     int   = 512
    dropout:    float = 0.1

    # ── mamba-specific architecture ───────────────────────────────────────────
    # d_state:       SSM latent state size N (typical: 16)
    # d_conv:        depthwise Conv1d kernel width (typical: 4)
    # d_expand:      d_inner = d_expand * d_model (typical: 2)
    # bidirectional: True → BiMambaBlock (full context); False → causal scan
    # Requires CUDA + mamba-ssm: pip install causal-conv1d mamba-ssm
    d_state:       int  = 16
    d_conv:        int  = 4
    d_expand:      int  = 2
    bidirectional: bool = True

    # ── runtime ───────────────────────────────────────────────────────────────
    num_workers: int = 0
    device:      str = "auto"   # "auto" | "cpu" | "cuda" | "mps"
    seed:        int = 42

    # ── experiment tracking ───────────────────────────────────────────────────
    use_wandb:     bool       = False
    wandb_project: str        = "mimic-cardio-oncology"
    run_name:      str | None = None   # defaults to wandb auto-generated name

    # ── embedding ablation flags ──────────────────────────────────────────────
    # fusion    "add"    BEHRT-style: element-wise sum of all embedding tables.
    #           "concat" CEHR-BERT style: cat([concept, time*, age*, position]) →
    #                    Linear(4d→d) → GELU, then + type + visit + segment residuals.
    #                    Missing components (use_time=False / use_age=False) are
    #                    zeroed in the concat before projection.
    # use_time  add sinusoidal time-gap embedding per token (requires dates.pt)
    # use_age   add continuous-age sinusoidal embedding per token (requires age_years.pt)
    #
    # Ablation grid:
    #   A0  fusion="add",    use_time=False, use_age=False  — baseline
    #   A1  fusion="add",    use_time=True,  use_age=False  — + time
    #   A2  fusion="add",    use_time=False, use_age=True   — + age
    #   A3  fusion="add",    use_time=True,  use_age=True   — + time + age
    #   B0  fusion="concat", use_time=False, use_age=False  — concat only
    #   B1  fusion="concat", use_time=True,  use_age=False  — concat + time
    #   B2  fusion="concat", use_time=True,  use_age=True   — CEHR-BERT
    #   C1/C2 — same flags but data_dir must point to an insert_att=True tokenization
    fusion:   str  = "add"
    use_time: bool = False
    use_age:  bool = False

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
