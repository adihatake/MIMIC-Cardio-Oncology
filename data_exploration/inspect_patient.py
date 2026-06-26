"""
inspect_patient.py

Visualize a tokenized patient EHR sequence and optionally run a model prediction.

Usage:
    # Random patient from test split
    python data_exploration/inspect_patient.py

    # Specific patient index (0-based within the split)
    python data_exploration/inspect_patient.py --patient-idx 5

    # By MIMIC subject_id
    python data_exploration/inspect_patient.py --subject-id 13595646

    # Subject with multiple cycles — pick the second one (0-based)
    python data_exploration/inspect_patient.py --subject-id 13595646 --cycle-idx 1

    # With model prediction
    python data_exploration/inspect_patient.py --patient-idx 5 --model-dir experiment_outputs/test1

    # From a different split
    python data_exploration/inspect_patient.py --split val --patient-idx 0 --model-dir experiment_outputs/test1

    # Show more events per visit
    python data_exploration/inspect_patient.py --patient-idx 5 --max-per-visit 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.ehr_encoder import EHR_Encoder

# ── lookup tables ─────────────────────────────────────────────────────────────

TYPE_COLORS = {0: "dim", 1: "cyan", 2: "green", 3: "yellow", 4: "magenta"}
TYPE_NAMES  = {0: "Special", 1: "Diagnosis", 2: "Procedure", 3: "Lab", 4: "Medication"}
AGE_LABELS  = ["0–9", "10–19", "20–29", "30–39", "40–49",
                "50–59", "60–69", "70–79", "80–89", "90+"]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_resources(data_dir: Path):
    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    vocab_inv = {v: k for k, v in vocab["concept_vocab"].items()}

    samples_meta = []
    with open(data_dir / "samples.csv") as f:
        for row in csv.DictReader(f):
            samples_meta.append({
                "subject_id":      int(row["subject_id"]),
                "cycle_number":    int(row["cycle_number"]),
                "prediction_time": row["prediction_time"],
                "binary_label":    int(row["binary_label"]),
                "seq_len":         int(row["seq_len"]),
            })

    tensors = {
        "concept_ids":  torch.load(data_dir / "concept_ids.pt",  weights_only=True),
        "type_ids":     torch.load(data_dir / "type_ids.pt",     weights_only=True),
        "visit_ids":    torch.load(data_dir / "visit_ids.pt",    weights_only=True),
        "position_ids": torch.load(data_dir / "position_ids.pt", weights_only=True),
        "age_ids":      torch.load(data_dir / "age_ids.pt",      weights_only=True),
        "labels":       torch.load(data_dir / "labels.pt",       weights_only=True),
    }
    return vocab_inv, splits["row_indices"], samples_meta, tensors


def _get_sample(tensors: dict, row_idx: int) -> dict:
    return {k: v[row_idx] for k, v in tensors.items()}


def _find_subject_cycles(subject_id: int, samples_meta: list, row_indices: dict):
    """Return [(split_name, split_pos, row_idx, meta)] for all cycles of a subject."""
    row_to_split: dict[int, tuple[str, int]] = {}
    for split_name, rows in row_indices.items():
        for pos, row_idx in enumerate(rows):
            row_to_split[row_idx] = (split_name, pos)

    results = []
    for global_row, meta in enumerate(samples_meta):
        if meta["subject_id"] == subject_id and global_row in row_to_split:
            split_name, split_pos = row_to_split[global_row]
            results.append((split_name, split_pos, global_row, meta))
    return results


# ── sequence display ──────────────────────────────────────────────────────────

def show_sequence(
    sample:        dict,
    row_meta:      dict,
    vocab_inv:     dict,
    split_name:    str,
    split_pos:     int,
    row_idx:       int,
    max_seq_len:   int,
    max_per_visit: int,
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    concept_ids = sample["concept_ids"].tolist()
    type_ids    = sample["type_ids"].tolist()
    visit_ids   = sample["visit_ids"].tolist()
    age_id      = int(sample["age_ids"].item())
    label       = int(sample["labels"].item())

    real_tokens = [
        (pos, cid, tid, vid)
        for pos, (cid, tid, vid) in enumerate(zip(concept_ids, type_ids, visit_ids))
        if cid != 0
    ]
    real_len = len(real_tokens)

    visits: dict[int, list] = {}
    for pos, cid, tid, vid in real_tokens:
        visits.setdefault(vid, []).append((pos, cid, tid))

    type_counts: dict[str, int] = {}
    for _, cid, tid, _ in real_tokens:
        name = TYPE_NAMES.get(tid, f"type_{tid}")
        if name != "Special":
            type_counts[name] = type_counts.get(name, 0) + 1

    label_str = "[bold red]POSITIVE[/]"   if label  == 1 else "[bold green]NEGATIVE[/]"
    age_str   = AGE_LABELS[age_id]        if age_id < len(AGE_LABELS) else str(age_id)
    trunc_str = (f" [yellow](truncated from raw len {row_meta['seq_len']})[/]"
                 if row_meta["seq_len"] >= max_seq_len else "")

    header = (
        f"[bold]Subject ID:[/] {row_meta['subject_id']}    "
        f"[bold]Cycle:[/] {row_meta['cycle_number']}    "
        f"[bold]Prediction time:[/] {row_meta['prediction_time']}\n"
        f"[bold]Split:[/] {split_name}  "
        f"[bold]Index:[/] {split_pos} (global row {row_idx})\n"
        f"[bold]Label:[/] {label_str}    "
        f"[bold]Age bucket:[/] {age_id} ({age_str} yrs)\n"
        f"[bold]Tokens:[/] {real_len} / {len(concept_ids)}{trunc_str}    "
        f"[bold]Visits:[/] {len(visits)}"
    )
    console.print(Panel(header, title="[bold blue]Patient Sequence[/]", expand=False))

    if type_counts:
        breakdown = "  ".join(
            f"[{TYPE_COLORS.get(next((k for k, v in TYPE_NAMES.items() if v == name), 0), 'white')}]"
            f"{name}[/]: {cnt}"
            for name, cnt in sorted(type_counts.items(), key=lambda x: -x[1])
        )
        console.print(f"\n[bold]Event breakdown:[/] {breakdown}\n")

    for vid in sorted(visits.keys()):
        events    = visits[vid]
        truncated = len(events) > max_per_visit
        shown     = events[:max_per_visit]

        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
            title=f"[bold]Visit {vid}[/]  ({len(events)} events)",
            title_justify="left",
            expand=False,
        )
        table.add_column("Pos",     style="dim", width=5, justify="right")
        table.add_column("Type",    width=11)
        table.add_column("Concept", overflow="fold")

        for pos, cid, tid in shown:
            type_name = TYPE_NAMES.get(tid, f"type_{tid}")
            color     = TYPE_COLORS.get(tid, "white")
            concept   = vocab_inv.get(cid, f"<id={cid}>")
            table.add_row(str(pos), f"[{color}]{type_name}[/]", concept)

        if truncated:
            table.add_row(
                "...", "...",
                f"… {len(events) - max_per_visit} more events (use --max-per-visit to show all)"
            )

        console.print(table)


# ── model prediction ──────────────────────────────────────────────────────────

def run_prediction(sample: dict, model_dir: Path) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    config_path = model_dir / "config.json"
    ckpt_path   = model_dir / "best_model.pt"

    if not config_path.exists():
        console.print(f"[yellow]No config.json in {model_dir} — skipping prediction.[/]")
        return
    if not ckpt_path.exists():
        console.print(f"[yellow]No best_model.pt in {model_dir} — skipping prediction.[/]")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = EHR_Encoder(
        num_concepts   = cfg["num_concepts"],
        max_num_visits = cfg["max_num_visits"],
        d_model        = cfg["d_model"],
        num_heads      = cfg["num_heads"],
        num_layers     = cfg["num_layers"],
        ff_dim         = cfg["ff_dim"],
        dropout        = cfg.get("dropout", 0.1),
        max_seq_len    = cfg["max_seq_len"],
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        concept_ids  = sample["concept_ids"].unsqueeze(0).to(device)
        type_ids     = sample["type_ids"].unsqueeze(0).to(device)
        visit_ids    = sample["visit_ids"].unsqueeze(0).to(device)
        position_ids = sample["position_ids"].unsqueeze(0).to(device)
        age_ids      = sample["age_ids"].unsqueeze(0).to(device)

        logits = model(concept_ids, type_ids, visit_ids, position_ids, age_ids)
        probs  = F.softmax(logits, dim=-1).squeeze(0)

    pred       = int(probs.argmax().item())
    true_label = int(sample["labels"].item())
    p_pos      = probs[1].item()
    p_neg      = probs[0].item()
    correct    = pred == true_label

    pred_str = "[bold red]POSITIVE[/]"  if pred       == 1 else "[bold green]NEGATIVE[/]"
    true_str = "[bold red]POSITIVE[/]"  if true_label == 1 else "[bold green]NEGATIVE[/]"
    check    = "[bold green]CORRECT[/]" if correct else "[bold red]WRONG[/]"
    bar_pos  = "█" * int(p_pos * 20) + "░" * (20 - int(p_pos * 20))
    bar_neg  = "█" * int(p_neg * 20) + "░" * (20 - int(p_neg * 20))

    result = (
        f"[bold]Model:[/]      {ckpt_path}\n"
        f"[bold]True label:[/] {true_str}\n"
        f"[bold]Predicted:[/]  {pred_str}  →  {check}\n\n"
        f"[bold]P(cardiotoxic):[/]     [red]{bar_pos}[/] {p_pos:.4f}\n"
        f"[bold]P(non-cardiotoxic):[/] [green]{bar_neg}[/] {p_neg:.4f}"
    )
    console.print(Panel(result, title="[bold blue]Model Prediction[/]", expand=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a tokenized patient sequence and optionally run a model prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",      default="tokenization_outputs/ver1",
                   help="Path to tokenization_outputs/<name>/")
    p.add_argument("--split",         default="test", choices=["train", "val", "test"],
                   help="Which split to sample from (ignored when --subject-id is used).")
    p.add_argument("--patient-idx",   type=int, default=None,
                   help="0-based index within the split. Random if omitted.")
    p.add_argument("--subject-id",    type=int, default=None, dest="subject_id",
                   help="MIMIC subject_id to look up directly (across all splits).")
    p.add_argument("--cycle-idx",     type=int, default=0, dest="cycle_idx",
                   help="0-based cycle to display when a subject has multiple (default: 0).")
    p.add_argument("--model-dir",     default=None,
                   help="Experiment dir containing config.json + best_model.pt.")
    p.add_argument("--max-per-visit", type=int, default=20,
                   help="Max events to display per visit (0 = all).")
    return p.parse_args()


def main() -> None:
    try:
        import rich  # noqa: F401
    except ImportError:
        print("This tool requires the 'rich' library.  pip install rich")
        sys.exit(1)

    args     = parse_args()
    data_dir = Path(args.data_dir)
    max_pv   = args.max_per_visit if args.max_per_visit > 0 else 10_000

    vocab_inv, row_indices, samples_meta, tensors = _load_resources(data_dir)

    with open(data_dir / "metadata.json") as f:
        max_seq_len = json.load(f)["max_seq_len"]

    # ── resolve which row to display ─────────────────────────────────────────
    if args.subject_id is not None:
        cycles = _find_subject_cycles(args.subject_id, samples_meta, row_indices)
        if not cycles:
            print(f"Subject ID {args.subject_id} not found in any split.")
            sys.exit(1)

        if len(cycles) > 1:
            print(f"Subject {args.subject_id} has {len(cycles)} cycle(s):")
            for i, (sname, spos, ridx, smeta) in enumerate(cycles):
                label_tag = "POS" if smeta["binary_label"] == 1 else "NEG"
                print(f"  [{i}]  split={sname:<5}  idx={spos:<4}  "
                      f"cycle={smeta['cycle_number']}  "
                      f"time={smeta['prediction_time']}  label={label_tag}")
            print(f"\nShowing --cycle-idx {args.cycle_idx} "
                  f"(pass --cycle-idx N to pick another).\n")

        if args.cycle_idx >= len(cycles):
            print(f"--cycle-idx {args.cycle_idx} out of range (0–{len(cycles) - 1}).")
            sys.exit(1)

        split_name, split_pos, row_idx, _ = cycles[args.cycle_idx]

    else:
        split_rows = row_indices[args.split]

        if args.patient_idx is None:
            split_pos = random.randrange(len(split_rows))
            print(f"No --patient-idx given. Randomly selected index {split_pos} "
                  f"from {args.split} split.")
        else:
            split_pos = args.patient_idx
            if split_pos < 0 or split_pos >= len(split_rows):
                print(f"--patient-idx {split_pos} out of range for {args.split} split "
                      f"(0–{len(split_rows) - 1}).")
                sys.exit(1)

        row_idx    = split_rows[split_pos]
        split_name = args.split

    sample   = _get_sample(tensors, row_idx)
    row_meta = samples_meta[row_idx]

    show_sequence(sample, row_meta, vocab_inv, split_name, split_pos, row_idx, max_seq_len, max_pv)

    if args.model_dir:
        run_prediction(sample, Path(args.model_dir))


if __name__ == "__main__":
    main()
