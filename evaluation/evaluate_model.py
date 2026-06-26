"""
evaluate_model.py

Evaluate a trained EHR_Encoder checkpoint on a data split and report
aggregate metrics plus a per-sample result table.

Usage:
    # Evaluate on test split (default)
    python evaluation/evaluate_model.py --model-dir experiment_outputs/test1

    # Evaluate on val split
    python evaluation/evaluate_model.py --model-dir experiment_outputs/test1 --split val

    # Save per-sample CSV
    python evaluation/evaluate_model.py --model-dir experiment_outputs/test1 \\
        --output-csv experiment_outputs/test1/test_results.csv

    # Smaller batch size to save memory
    python evaluation/evaluate_model.py --model-dir experiment_outputs/test1 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.ehr_encoder import EHR_Encoder


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_data(data_dir: Path):
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    samples_meta = []
    with open(data_dir / "samples.csv") as f:
        for row in csv.DictReader(f):
            samples_meta.append({
                "subject_id":      int(row["subject_id"]),
                "cycle_number":    int(row["cycle_number"]),
                "prediction_time": row["prediction_time"],
            })

    tensors = {
        "concept_ids":  torch.load(data_dir / "concept_ids.pt",  weights_only=True),
        "type_ids":     torch.load(data_dir / "type_ids.pt",     weights_only=True),
        "visit_ids":    torch.load(data_dir / "visit_ids.pt",    weights_only=True),
        "position_ids": torch.load(data_dir / "position_ids.pt", weights_only=True),
        "age_ids":      torch.load(data_dir / "age_ids.pt",      weights_only=True),
        "labels":       torch.load(data_dir / "labels.pt",       weights_only=True),
    }
    return splits["row_indices"], samples_meta, tensors


def _load_model(model_dir: Path, device: torch.device) -> tuple[EHR_Encoder, dict]:
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

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

    ckpt = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model, cfg


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(
    model:       EHR_Encoder,
    tensors:     dict,
    row_indices: list[int],
    device:      torch.device,
    batch_size:  int,
) -> tuple[list[float], list[int]]:
    all_probs, all_labels = [], []

    for start in range(0, len(row_indices), batch_size):
        rows = row_indices[start : start + batch_size]
        concept_ids  = tensors["concept_ids"][rows].to(device)
        type_ids     = tensors["type_ids"][rows].to(device)
        visit_ids    = tensors["visit_ids"][rows].to(device)
        position_ids = tensors["position_ids"][rows].to(device)
        age_ids      = tensors["age_ids"][rows].to(device)
        labels       = tensors["labels"][rows]

        logits = model(concept_ids, type_ids, visit_ids, position_ids, age_ids)
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()

        all_probs.extend(probs)
        all_labels.extend(labels.tolist())

    return all_probs, all_labels


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score,
        precision_recall_fscore_support, confusion_matrix,
    )

    preds = [1 if p >= threshold else 0 for p in probs]

    auroc    = roc_auc_score(labels, probs)
    accuracy = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return {
        "auroc":     auroc,
        "accuracy":  accuracy,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "cm":        cm.tolist(),
        "threshold": threshold,
        "n_samples": len(labels),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
    }


# ── display ───────────────────────────────────────────────────────────────────

def _display_results(
    metrics:     dict,
    split:       str,
    row_indices: list[int],
    samples_meta: list[dict],
    labels:      list[int],
    probs:       list[float],
    model_dir:   Path,
    max_rows:    int,
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    # ── metrics panel ─────────────────────────────────────────────────────────
    cm     = metrics["cm"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    summary = (
        f"[bold]Model:[/]     {model_dir}\n"
        f"[bold]Split:[/]     {split}  "
        f"({metrics['n_samples']} samples: "
        f"{metrics['n_positive']} pos / {metrics['n_negative']} neg)\n\n"
        f"[bold]AUROC:[/]     [cyan]{metrics['auroc']:.4f}[/]\n"
        f"[bold]Accuracy:[/]  {metrics['accuracy']:.4f}  "
        f"(threshold = {metrics['threshold']})\n"
        f"[bold]Precision:[/] {metrics['precision']:.4f}   "
        f"[bold]Recall:[/] {metrics['recall']:.4f}   "
        f"[bold]F1:[/] {metrics['f1']:.4f}\n\n"
        f"[bold]Confusion matrix[/] (rows = true, cols = pred):\n"
        f"  TN={tn}  FP={fp}\n"
        f"  FN={fn}  TP={tp}"
    )
    console.print(Panel(summary, title="[bold blue]Evaluation Results[/]", expand=False))

    # ── per-sample table ──────────────────────────────────────────────────────
    threshold = metrics["threshold"]
    preds     = [1 if p >= threshold else 0 for p in probs]

    console.print(f"\n[bold]Per-sample results[/] (showing first {max_rows} of {len(row_indices)}):\n")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=False,
    )
    table.add_column("#",              style="dim", width=5,  justify="right")
    table.add_column("Subject ID",     width=12)
    table.add_column("Cycle",          width=6,  justify="center")
    table.add_column("Pred. Time",     width=22)
    table.add_column("True",           width=8,  justify="center")
    table.add_column("Pred",           width=8,  justify="center")
    table.add_column("P(cardiotoxic)", width=16, justify="right")
    table.add_column("",               width=7)

    for i, (row_idx, true, pred, prob) in enumerate(
        zip(row_indices, labels, preds, probs)
    ):
        if i >= max_rows:
            break
        meta     = samples_meta[row_idx]
        true_str = "[red]POS[/]"   if true == 1 else "[green]NEG[/]"
        pred_str = "[red]POS[/]"   if pred == 1 else "[green]NEG[/]"
        ok_str   = "[green]✓[/]"   if true == pred else "[red]✗[/]"
        prob_str = f"[red]{prob:.4f}[/]" if prob >= threshold else f"[green]{prob:.4f}[/]"

        table.add_row(
            str(i),
            str(meta["subject_id"]),
            str(meta["cycle_number"]),
            meta["prediction_time"],
            true_str,
            pred_str,
            prob_str,
            ok_str,
        )

    if len(row_indices) > max_rows:
        table.add_row(
            "...", "...", "...", "...", "...", "...", "...",
            f"[dim]+{len(row_indices) - max_rows} more[/]"
        )

    console.print(table)


# ── CSV export ────────────────────────────────────────────────────────────────

def _save_csv(
    output_path: Path,
    split:       str,
    row_indices: list[int],
    samples_meta: list[dict],
    labels:      list[int],
    probs:       list[float],
    threshold:   float,
) -> None:
    preds = [1 if p >= threshold else 0 for p in probs]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "split_pos", "row_idx",
            "subject_id", "cycle_number", "prediction_time",
            "true_label", "pred_label", "prob_cardiotoxic", "correct",
        ])
        for i, (row_idx, true, pred, prob) in enumerate(
            zip(row_indices, labels, preds, probs)
        ):
            meta = samples_meta[row_idx]
            writer.writerow([
                split, i, row_idx,
                meta["subject_id"], meta["cycle_number"], meta["prediction_time"],
                true, pred, f"{prob:.6f}", int(true == pred),
            ])

    print(f"Per-sample results saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on a data split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir",   required=True,
                   help="Experiment dir with config.json + best_model.pt.")
    p.add_argument("--data-dir",    default=None,
                   help="Data dir (default: read from model config.json).")
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Decision threshold for binary predictions.")
    p.add_argument("--max-rows",    type=int, default=50,
                   help="Max rows to show in the per-sample table.")
    p.add_argument("--output-csv",  default=None,
                   help="Optional path to save per-sample results as CSV.")
    p.add_argument("--device",      default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'.")
    return p.parse_args()


def main() -> None:
    try:
        from sklearn.metrics import roc_auc_score  # noqa: F401
        import rich                                 # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with:  pip install scikit-learn rich")
        sys.exit(1)

    args      = parse_args()
    model_dir = Path(args.model_dir)

    for required in ("config.json", "best_model.pt"):
        if not (model_dir / required).exists():
            print(f"Missing {required} in {model_dir}")
            sys.exit(1)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    model, cfg = _load_model(model_dir, device)
    print(f"Loaded checkpoint from {model_dir / 'best_model.pt'}")

    data_dir_str = args.data_dir or cfg.get("data_dir")
    if not data_dir_str:
        print("Cannot determine data_dir. Pass --data-dir explicitly.")
        sys.exit(1)
    data_dir = Path(data_dir_str)

    row_indices_all, samples_meta, tensors = _load_data(data_dir)
    split_rows = row_indices_all[args.split]
    print(f"Running on {args.split} split ({len(split_rows)} samples) …")

    probs, labels = _run_inference(model, tensors, split_rows, device, args.batch_size)
    metrics       = _compute_metrics(labels, probs, threshold=args.threshold)

    _display_results(
        metrics, args.split, split_rows, samples_meta,
        labels, probs, model_dir, args.max_rows,
    )

    if args.output_csv:
        _save_csv(
            Path(args.output_csv), args.split, split_rows,
            samples_meta, labels, probs, args.threshold,
        )


if __name__ == "__main__":
    main()
