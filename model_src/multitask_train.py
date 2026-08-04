"""
multitask_train.py

Training loop for multi-task cardiotoxicity prediction at 90 / 180 / 365 days.

Requires a tokenization produced with --multitask (emits task_ids.pt). Each
(patient, cycle) row is expanded to three rows — one per prediction window —
each prepended with a learned task token that tells the shared Transformer which
window to predict.  A single classifier head is shared across all tasks.

Usage:
    python model_src/multitask_train.py --data-dir tokenization_outputs/mt_ver1

    # smaller config for debugging:
    python model_src/multitask_train.py --data-dir tokenization_outputs/mt_ver1 \\
        --d-model 64 --num-heads 4 --num-layers 2 --epochs 3 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
)
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.dataset import get_dataloaders
from model_src.ehr_encoder import EHR_Encoder

TASK_WINDOWS = {0: "90d", 1: "180d", 2: "365d"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _load_meta(data_dir: Path) -> tuple[dict, dict]:
    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    return vocab, meta


def _compute_metrics(labels: list, probs: list, threshold: float) -> dict:
    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs, pos_label=1)
        preds = [1 if p >= threshold else 0 for p in probs]
        _, sensitivity, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        auroc, auprc, f1, sensitivity, specificity = (float("nan"),) * 5
    return {
        "auroc": auroc, "auprc": auprc, "f1": f1,
        "sensitivity": sensitivity, "specificity": specificity,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    all_labels: list   = []
    all_probs:  list   = []
    all_task_ids: list = []

    for batch in tqdm(loader, desc="  val", unit="batch", leave=False):
        concept_ids  = batch["concept_ids"].to(device)
        type_ids     = batch["type_ids"].to(device)
        visit_ids    = batch["visit_ids"].to(device)
        position_ids = batch["position_ids"].to(device)
        age_ids      = batch["age_ids"].to(device)
        labels       = batch["label"].to(device)
        dates        = batch["dates"].to(device)     if "dates"    in batch else None
        age_years    = batch["age_years"].to(device) if "age_years" in batch else None
        task_ids     = batch["task_id"].to(device)   if "task_id"  in batch else None

        logits = model(concept_ids, type_ids, visit_ids, position_ids,
                       age_ids, dates, age_years, task_ids)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        n          += len(labels)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        if task_ids is not None:
            all_task_ids.extend(task_ids.cpu().numpy().tolist())

    avg_loss = total_loss / n
    overall  = _compute_metrics(all_labels, all_probs, threshold)

    per_task: dict[str, dict] = {}
    if all_task_ids:
        arr_labels   = np.array(all_labels)
        arr_probs    = np.array(all_probs)
        arr_task_ids = np.array(all_task_ids)
        for tid, tname in TASK_WINDOWS.items():
            mask = arr_task_ids == tid
            if mask.sum() > 0:
                per_task[tname] = _compute_metrics(
                    arr_labels[mask].tolist(), arr_probs[mask].tolist(), threshold
                )

    return {"loss": avg_loss, **overall, "per_task": per_task, "eval_threshold": threshold}


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace | object) -> None:
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    device = _device(args.device)
    print(f"Device: {device}  |  Seed: {args.seed}")

    vocab, meta = _load_meta(data_dir)
    num_concepts  = len(vocab["concept_vocab"])
    max_seq_len   = meta["max_seq_len"]
    positive_rate = meta["positive_rate"]
    n_tasks       = meta.get("n_tasks", 3)

    if not meta.get("multitask", False):
        raise ValueError(
            f"data_dir {data_dir} was not tokenized with --multitask. "
            "Re-run tokenize_cli.py --multitask and point --data-dir here."
        )

    class_weights = torch.tensor(
        [1.0 / (1.0 - positive_rate), 1.0 / positive_rate],
        dtype=torch.float32,
        device=device,
    )
    print(f"Vocab size   : {num_concepts:,}")
    print(f"Max seq len  : {max_seq_len}")
    print(f"Tasks        : {n_tasks}  (windows: {meta.get('task_windows', [90,180,365])})")
    print(f"Positive rate: {positive_rate:.1%}  →  class weights {class_weights.tolist()}")

    visit_ids_all  = torch.load(data_dir / "visit_ids.pt", weights_only=True)
    max_num_visits = int(visit_ids_all.max().item()) + 1
    print(f"Max visit id : {max_num_visits - 1}  →  visit embedding size {max_num_visits}")
    del visit_ids_all

    train_dl, val_dl, test_dl = get_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_dl)}  |  Val: {len(val_dl)}  |  Test: {len(test_dl)}")

    model = EHR_Encoder(
        num_concepts=num_concepts,
        max_num_visits=max_num_visits,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
        fusion=getattr(args, "fusion", "add"),
        use_time=getattr(args, "use_time", False),
        use_age=getattr(args, "use_age", False),
        num_tasks=n_tasks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}")

    criterion     = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=getattr(args, "label_smoothing", 0.0),
    )
    optimizer     = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(0, round(getattr(args, "warmup_frac", 0.1) * args.epochs))
    if warmup_epochs == 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    else:
        warmup  = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine  = CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr / 10,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        gpu_name  = torch.cuda.get_device_name(device)
        gpu_count = torch.cuda.device_count()
    else:
        gpu_name, gpu_count = None, 0

    config = vars(args) | {
        "num_concepts":   num_concepts,
        "max_seq_len":    max_seq_len,
        "max_num_visits": max_num_visits,
        "n_params":       n_params,
        "n_tasks":        n_tasks,
        "run_date":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "compute_host":   socket.gethostname(),
        "platform":       platform.platform(),
        "cpu":            platform.processor() or platform.machine(),
        "gpu_name":       gpu_name,
        "gpu_count":      gpu_count,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config,
            dir=str(output_dir),
        )
        wandb.watch(model, log="gradients", log_freq=100)

    CKPT_METRICS = ["auroc", "auprc", "f1", "sensitivity", "specificity"]
    best_scores  = {m: -1.0 for m in CKPT_METRICS}
    best_epochs  = {m: -1   for m in CKPT_METRICS}
    history      = []

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0

        batch_bar = tqdm(train_dl, desc="  train", unit="batch", leave=False)
        for batch in batch_bar:
            concept_ids  = batch["concept_ids"].to(device)
            type_ids     = batch["type_ids"].to(device)
            visit_ids    = batch["visit_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            age_ids      = batch["age_ids"].to(device)
            labels       = batch["label"].to(device)
            dates        = batch["dates"].to(device)     if "dates"    in batch else None
            age_years    = batch["age_years"].to(device) if "age_years" in batch else None
            task_ids     = batch["task_id"].to(device)   if "task_id"  in batch else None

            optimizer.zero_grad()
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(concept_ids, type_ids, visit_ids, position_ids,
                               age_ids, dates, age_years, task_ids)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            n          += len(labels)
            batch_bar.set_postfix(loss=f"{train_loss / n:.4f}")

        scheduler.step()
        avg_train_loss = train_loss / n
        val_metrics    = evaluate(model, val_dl, criterion, device,
                                  threshold=getattr(args, "eval_threshold", 0.5))
        elapsed        = time.time() - t0

        updated = []
        for m in CKPT_METRICS:
            if val_metrics[m] > best_scores[m]:
                best_scores[m] = val_metrics[m]
                best_epochs[m] = epoch
                torch.save(model.state_dict(), output_dir / f"best_model_{m}.pt")
                updated.append(m)

        epoch_bar.set_postfix(
            loss    = f"{avg_train_loss:.4f}",
            val_loss= f"{val_metrics['loss']:.4f}",
            auroc   = f"{val_metrics['auroc']:.4f}",
            auprc   = f"{val_metrics['auprc']:.4f}",
            new     = ",".join(updated) if updated else "—",
        )

        row = {"epoch": epoch, "train_loss": avg_train_loss, "elapsed": elapsed,
               **{k: v for k, v in val_metrics.items() if k != "per_task"}}
        for tname, tm in val_metrics.get("per_task", {}).items():
            for metric, val in tm.items():
                row[f"val_{tname}_{metric}"] = val
        history.append(row)

        if args.use_wandb:
            log_dict = {
                "train/loss":      avg_train_loss,
                "val/loss":        val_metrics["loss"],
                "val/auroc":       val_metrics["auroc"],
                "val/auprc":       val_metrics["auprc"],
                "val/f1":          val_metrics["f1"],
                "val/sensitivity": val_metrics["sensitivity"],
                "val/specificity": val_metrics["specificity"],
                "lr":              scheduler.get_last_lr()[0],
                "epoch":           epoch,
            }
            for tname, tm in val_metrics.get("per_task", {}).items():
                for metric, val in tm.items():
                    log_dict[f"val_{tname}/{metric}"] = val
            wandb.log(log_dict)

    print("\nBest validation scores (overall):")
    col_w = max(len(m) for m in CKPT_METRICS)
    for m in CKPT_METRICS:
        print(f"  {m:<{col_w}}  {best_scores[m]:.4f}  (epoch {best_epochs[m]})")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate each per-metric checkpoint on held-out test set
    print("\nTest results by checkpoint:")
    header_metrics = ["auroc", "auprc", "f1", "sensitivity", "specificity", "loss"]
    print(f"  {'ckpt':<12}" + "".join(f"  {h:>8}" for h in header_metrics))
    print("  " + "─" * (12 + len(header_metrics) * 10))

    all_test_metrics: dict[str, dict] = {}
    for m in CKPT_METRICS:
        model.load_state_dict(
            torch.load(output_dir / f"best_model_{m}.pt", weights_only=True)
        )
        tm = evaluate(model, test_dl, criterion, device,
                      threshold=getattr(args, "eval_threshold", 0.5))
        all_test_metrics[m] = {k: v for k, v in tm.items() if k != "per_task"}
        all_test_metrics[m]["per_task"] = tm.get("per_task", {})

        with open(output_dir / f"test_metrics_{m}.json", "w") as f:
            json.dump(all_test_metrics[m], f, indent=2)
        row_str = "".join(f"  {tm[h]:>8.4f}" for h in header_metrics)
        print(f"  {m:<12}{row_str}")

        # Print per-task breakdown for the AUROC checkpoint
        if m == "auroc" and tm.get("per_task"):
            print("\n  Per-task breakdown (AUROC checkpoint):")
            for tname, task_m in tm["per_task"].items():
                task_str = "  ".join(f"{k}={task_m[k]:.4f}" for k in ["auroc", "auprc", "f1"])
                print(f"    {tname}: {task_str}")

        if args.use_wandb:
            wandb.log({f"test_{m}_ckpt/{k}": v for k, v in tm.items()
                       if isinstance(v, (int, float))})

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(all_test_metrics["auroc"], f, indent=2)

    if args.use_wandb:
        artifact = wandb.Artifact("best_models", type="model")
        for m in CKPT_METRICS:
            artifact.add_file(str(output_dir / f"best_model_{m}.pt"))
        wandb.log_artifact(artifact)
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-task EHR_Encoder training (90d / 180d / 365d cardiotoxicity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",     required=True,
                   help="Path to a multitask tokenization_outputs/<name>/ (must have task_ids.pt).")
    p.add_argument("--output-dir",   default="model_outputs/multitask_run1",
                   help="Where to save checkpoints and logs.")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--d-model",      type=int,   default=96)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--num-layers",   type=int,   default=1)
    p.add_argument("--ff-dim",       type=int,   default=192)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--num-workers",  type=int,   default=0)
    p.add_argument("--device",       default="auto")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--use-wandb",          action="store_true", dest="use_wandb")
    p.add_argument("--wandb-project",      default="mimic-cardio-oncology", dest="wandb_project")
    p.add_argument("--run-name",           default=None, dest="run_name")
    p.add_argument("--label-smoothing",    type=float, default=0.1, dest="label_smoothing")
    p.add_argument("--fusion",    default="add", choices=["add", "concat"])
    p.add_argument("--use-time", action="store_true", dest="use_time")
    p.add_argument("--use-age",  action="store_true", dest="use_age")
    p.add_argument("--warmup-frac",     type=float, default=0.1, dest="warmup_frac")
    p.add_argument("--eval-threshold",  type=float, default=0.5, dest="eval_threshold")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
