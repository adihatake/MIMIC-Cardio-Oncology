"""
train.py

One training run for the EHR_Encoder cardiotoxicity classifier.

Reads tokenized tensors and splits from a tokenization_outputs/<name>/ directory,
trains with cross-entropy loss (class-weighted for imbalance), evaluates on the
validation set each epoch, and saves the best checkpoint by validation AUROC.

Usage:
    python model_src/train.py --data-dir tokenization_outputs/ver1

    # smaller/faster config for debugging:
    python model_src/train.py --data-dir tokenization_outputs/ver1 \\
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
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_src.dataset import get_dataloaders
from model_src.ehr_encoder import EHR_Encoder


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# pick the best compute hardware
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


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc="  val", unit="batch", leave=False):
        concept_ids  = batch["concept_ids"].to(device)
        type_ids     = batch["type_ids"].to(device)
        visit_ids    = batch["visit_ids"].to(device)
        position_ids = batch["position_ids"].to(device)
        age_ids      = batch["age_ids"].to(device)
        labels       = batch["label"].to(device)
        dates        = batch["dates"].to(device)     if "dates"     in batch else None
        age_years    = batch["age_years"].to(device) if "age_years" in batch else None

        logits = model(concept_ids, type_ids, visit_ids, position_ids, age_ids, dates, age_years)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        n          += len(labels)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / n
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    return {"loss": avg_loss, "auroc": auroc}


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace | object) -> None:
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    device = _device(args.device)
    print(f"Device: {device}")
    print(f"Seed         : {args.seed}")

    vocab, meta = _load_meta(data_dir)
    num_concepts  = len(vocab["concept_vocab"])
    max_seq_len   = meta["max_seq_len"]
    positive_rate = meta["positive_rate"]

    # Class weights to handle label imbalance: w_pos = 1/rate, w_neg = 1/(1-rate)
    #************ Might need to review this, just in case. 
    class_weights = torch.tensor(
        [1.0 / (1.0 - positive_rate), 1.0 / positive_rate],
        dtype=torch.float32,
        device=device,
    )

    print(f"Vocab size   : {num_concepts:,}")
    print(f"Max seq len  : {max_seq_len}")
    print(f"Positive rate: {positive_rate:.1%}  →  class weights {class_weights.tolist()}")

    # Determine safe max_num_visits from the saved tensor
    visit_ids_all = torch.load(data_dir / "visit_ids.pt", weights_only=True)
    max_num_visits = int(visit_ids_all.max().item()) + 1
    print(f"Max visit id : {max_num_visits - 1}  →  visit embedding size {max_num_visits}")
    del visit_ids_all

    train_dl, val_dl, test_dl = get_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}  |  Test batches: {len(test_dl)}")

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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=getattr(args, "label_smoothing", 0.0),
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) ## Need to adjust/experiment with
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10) ## Need to adjust/experiment with
    scaler    = GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_count = torch.cuda.device_count()
    else:
        gpu_name = None
        gpu_count = 0

    config = vars(args) | {
        "num_concepts":   num_concepts,
        "max_seq_len":    max_seq_len,
        "max_num_visits": max_num_visits,
        "n_params":       n_params,
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

    best_auroc  = -1.0
    best_epoch  = -1
    history     = []

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0

        batch_bar = tqdm(train_dl, desc=f"  train", unit="batch", leave=False)
        for batch in batch_bar:
            concept_ids  = batch["concept_ids"].to(device)
            type_ids     = batch["type_ids"].to(device)
            visit_ids    = batch["visit_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            age_ids      = batch["age_ids"].to(device)
            labels       = batch["label"].to(device)
            dates        = batch["dates"].to(device)     if "dates"     in batch else None
            age_years    = batch["age_years"].to(device) if "age_years" in batch else None

            optimizer.zero_grad()
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(concept_ids, type_ids, visit_ids, position_ids, age_ids, dates, age_years)
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
        val_metrics    = evaluate(model, val_dl, criterion, device)
        elapsed        = time.time() - t0
        is_best        = val_metrics["auroc"] > best_auroc

        epoch_bar.set_postfix(
            train_loss = f"{avg_train_loss:.4f}",
            val_loss   = f"{val_metrics['loss']:.4f}",
            auroc      = f"{val_metrics['auroc']:.4f}",
            best       = f"epoch {epoch} ✓" if is_best else f"epoch {best_epoch}",
        )

        row = {"epoch": epoch, "train_loss": avg_train_loss, **val_metrics, "elapsed": elapsed}
        history.append(row)

        if args.use_wandb:
            wandb.log({
                "train/loss": avg_train_loss,
                "val/loss":   val_metrics["loss"],
                "val/auroc":  val_metrics["auroc"],
                "lr":         scheduler.get_last_lr()[0],
                "epoch":      epoch,
            })

        if is_best:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    print(f"\nBest val AUROC {best_auroc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate best checkpoint on held-out test set
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_dl, criterion, device)
    print(f"Test  AUROC  {test_metrics['auroc']:.4f}  |  loss {test_metrics['loss']:.4f}")
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    if args.use_wandb:
        wandb.log({"test/auroc": test_metrics["auroc"], "test/loss": test_metrics["loss"]})

    if args.use_wandb:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(str(output_dir / "best_model.pt"))
        wandb.log_artifact(artifact)
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EHR_Encoder for cardiotoxicity prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",     default="tokenization_outputs/ver1",
                   help="Path to tokenization_outputs/<name>/")
    p.add_argument("--output-dir",   default="model_outputs/run1",
                   help="Where to save checkpoints and logs.")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--d-model",      type=int,   default=128)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--num-layers",   type=int,   default=4)
    p.add_argument("--ff-dim",       type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--num-workers",  type=int,   default=0)
    p.add_argument("--device",       default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--use-wandb",          action="store_true", dest="use_wandb",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project",      default="mimic-cardio-oncology", dest="wandb_project")
    p.add_argument("--run-name",           default=None, dest="run_name",
                   help="W&B run name (defaults to auto-generated).")
    p.add_argument("--label-smoothing", type=float, default=0.0, dest="label_smoothing",
                   help="Label smoothing for CrossEntropyLoss (0 = off, 0.1 recommended for small datasets).")
    p.add_argument("--fusion",    default="add", choices=["add", "concat"],
                   help="'add': BEHRT-style element-wise sum. 'concat': CEHR-BERT concat→Linear→GELU.")
    p.add_argument("--use-time", action="store_true", dest="use_time",
                   help="Add sinusoidal time-gap embedding per token (requires dates.pt).")
    p.add_argument("--use-age",  action="store_true", dest="use_age",
                   help="Add continuous-age sinusoidal embedding (requires age_years.pt).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
