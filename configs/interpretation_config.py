from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InterpretationConfig:
    # ── required ──────────────────────────────────────────────────────────────
    model_dir: Path

    # ── sample selection: provide sample_idx OR (subject_id + cycle_number) ──
    sample_idx:   int | None = None
    subject_id:   int | None = None
    cycle_number: int | None = None

    # ── paths ─────────────────────────────────────────────────────────────────
    data_dir:   Path | None = None   # falls back to value in model config.json
    output_dir: Path | None = None   # default: interpretation/outputs/<label>/

    # ── IG settings ───────────────────────────────────────────────────────────
    ig_steps: int  = 100    # integration steps (more = more accurate, slower)
    skip_ig:  bool = False  # set True if captum is not installed

    # ── visualization ─────────────────────────────────────────────────────────
    top_k:        int  = 30    # top-K tokens shown in bar charts
    run_visualize: bool = True  # run visualize_attributions after explain

    # ── compute ───────────────────────────────────────────────────────────────
    device: str = "auto"

    # ── derived ───────────────────────────────────────────────────────────────
    def validate(self) -> None:
        has_idx    = self.sample_idx is not None
        has_patient = self.subject_id is not None and self.cycle_number is not None
        if not has_idx and not has_patient:
            raise ValueError(
                "InterpretationConfig: provide sample_idx or "
                "both subject_id and cycle_number"
            )
        if has_idx and has_patient:
            raise ValueError(
                "InterpretationConfig: provide either sample_idx or "
                "(subject_id + cycle_number), not both"
            )

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        repo_root = Path(__file__).resolve().parent.parent
        if self.subject_id is not None and self.cycle_number is not None:
            label = f"{self.subject_id}_cycle{self.cycle_number}"
        else:
            label = f"sample_{self.sample_idx}"
        return repo_root / "interpretation" / "outputs" / label

    @property
    def label(self) -> str:
        if self.subject_id is not None and self.cycle_number is not None:
            return f"subject {self.subject_id} cycle {self.cycle_number}"
        return f"sample_idx {self.sample_idx}"
