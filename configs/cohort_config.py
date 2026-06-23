from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CohortConfig:
    # required
    data_dir: Path

    # optional
    output_name: str = "cycle_modeling_ver2"

    # ── derived (read-only) ───────────────────────────────────────────────────
    @property
    def output_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "cohort_outputs" / self.output_name

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
    def load(cls, path: Path) -> CohortConfig:
        with open(path) as f:
            d = json.load(f)
        d["data_dir"] = Path(d["data_dir"])
        return cls(**d)
