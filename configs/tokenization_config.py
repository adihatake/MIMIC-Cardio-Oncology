from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TokenizationConfig:
    # required
    data_dir: Path

    # optional
    cohort_name:             str  = "cycle_modeling_ver2"
    output_name:             str  = "ver1"
    max_seq_len:             int  = 600
    run_split:               bool = False
    run_summarize:           bool = True
    insert_att:              bool = False  # CEHR-BERT ATT tokens between visits
    insert_visit_delimiters: bool = False  # [V_START]/[V_END] around each visit

    # ── derived (read-only) ───────────────────────────────────────────────────
    @property
    def cohort_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "cohort_outputs" / self.cohort_name

    @property
    def output_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "tokenization_outputs" / self.output_name

    # ── serialization ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(d["data_dir"])
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TokenizationConfig:
        with open(path) as f:
            d = json.load(f)
        d["data_dir"] = Path(d["data_dir"])
        return cls(**d)
