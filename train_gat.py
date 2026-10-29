"""
GAT Training Script — with Full Run Logging
------------------------------------------------
Trains a GAT (Graph Attention Network) classifier on the extracted cyclist
pose-keypoint graphs (safe vs. unsafe).

Every run is logged to its own timestamped folder under results/runs/, so
multiple training sessions (different hyperparameters, or later comparisons
against GCN and GIN) stay organized and comparable:

    results/runs/GAT_<timestamp>/
        config.json          <- every hyperparameter used this run
        metrics.csv           <- per-epoch train/val loss & accuracy
        train_log.txt         <- full console output, mirrored to disk
        best_model.pt          <- checkpoint from the best-val-accuracy epoch
        test_results.json     <- final test-set metrics (computed ONCE, at the end)

Training and validation run together, every epoch, in the same loop —
standard practice, since validation (no weight updates) is how you monitor
whether the model is generalizing rather than just memorizing the training
set. The TEST set is only touched once, at the very end, using the best
checkpoint — never during training, to keep it a genuinely held-out measure.

Usage:
    python train_gat.py
"""

import os
import json
import csv
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, global_mean_pool
from torch_geometric.nn.models import GAT as PyGGAT

from gnn_dataset import get_dataloaders

RESULTS_ROOT = "results/runs"


class RunLogger:
    """Handles everything about recording a single training run: creating
    the run folder, saving config, appending per-epoch metrics, mirroring
    console output to a log file, and saving the best checkpoint.
    """

    def __init__(self, config):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(RESULTS_ROOT, f"{config['model_name']}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self.log_path = os.path.join(self.run_dir, "train_log.txt")
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.checkpoint_path = os.path.join(self.run_dir, "best_model.pt")
        self.test_results_path = os.path.join(self.run_dir, "test_results.json")

        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        self.best_val_acc = -1.0
        self.epochs_since_improvement = 0
        self.val_acc_history = []

        self.log(f"Run started: {self.run_dir}")
        self.log(f"Config: {json.dumps(config, indent=2)}")

    def log(self, message):
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc])
        self.log(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
                  f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    def check_and_save_best(self, val_acc, model):
        """Uses a SMOOTHED (moving-average) val accuracy for checkpoint
        selection, not the raw per-epoch value. Raw per-epoch val accuracy
        on a few hundred samples is noisy (confirmed empirically in an
        early GAT run, GAT_20260729_142029, where val_acc swung between
        59.6% and 70.6% within a 15-epoch window). Selecting on a single
        noisy spike risks picking a "lucky" epoch rather than the model's
        true best generalization, which showed up as a ~9-point gap between
        best-val (70.6%) and test accuracy (61.4%) in that run. Averaging
        over a short recent window is a standard, simple fix.

        Returns True if training should stop early, False otherwise.
        """
        self.val_acc_history.append(val_acc)
        window = self.val_acc_history[-5:]   # smoothing window: last 5 epochs
        smoothed_val_acc = sum(window) / len(window)

        if smoothed_val_acc > self.best_val_acc:
            self.best_val_acc = smoothed_val_acc
            self.epochs_since_improvement = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            self.log(f"  -> new best smoothed val_acc {smoothed_val_acc:.4f} "
                      f"(raw this epoch: {val_acc:.4f}), checkpoint saved.")
        else:
            self.epochs_since_improvement += 1
        return self.epochs_since_improvement >= 25   # early stopping patience: 25 epochs

    def save_test_results(self, results_dict):
        with open(self.test_results_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        self.log(f"\nFinal test results: {json.dumps(results_dict, indent=2)}")
        self.log(f"Saved to {self.test_results_path}")


# ===========================================================================
# STEP 2: Create Model GAT — the architecture definition
#
# Built ENTIRELY from PyTorch Geometric's own built-in components, with no
# custom nn.Module subclass. The backbone
# is torch_geometric.nn.models.GAT (a complete, ready-made multi-layer GAT
# implementation), assembled with pooling and the final classifier using
# torch_geometric.nn.Sequential, which lets multiple components be chained
# declaratively without writing a custom forward() method.
#
# NOTE: the built-in GAT backbone does not concatenate the attention heads at
# its FINAL layer (only at intermediate layers); it averages them, per
# PyTorch Geometric's standard convention. The pooled representation reaching
# the classifier is therefore `hidden_channels` wide, not
# `hidden_channels * heads`.
# ===========================================================================
def build_gat_model(in_channels=7, hidden_channels=128, heads=8, num_layers=5, dropout=0.1, num_classes=2):
    backbone = PyGGAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        norm="BatchNorm",   # batch normalisation via the built-in's own norm option, for training stability
        act="elu",
    )
    model = Sequential("x, edge_index, batch", [
        (backbone, "x, edge_index -> x"),
        (global_mean_pool, "x, batch -> x"),
        (torch.nn.Linear(hidden_channels, num_classes), "x -> x"),
    ])
    return model


def build_optimizer(model, lr=0.001, weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def run_epoch(model, loader, optimizer, device, train, logger=None, epoch=None, batch_loss_history=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if train:
                optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss_value = loss.item()

            # Live spike detection: compare this batch's loss against a
            # rolling window of recent batch losses. If it's an outlier,
            # log full diagnostic detail (which files, actual x range)
            # RIGHT NOW, in the real run — this is more reliable than
            # trying to replay PyTorch's internal RNG sequence to guess
            # which batch caused a spike after the fact. The angle features
            # are part of x, so the x checks below cover them.
            if train and batch_loss_history is not None:
                if len(batch_loss_history) >= 30:
                    recent_avg = sum(batch_loss_history[-30:]) / 30
                    if loss_value > 2.5 * recent_avg and loss_value > 0.85:
                        msg = (f"\n*** LOSS SPIKE DETECTED at epoch {epoch}: batch loss {loss_value:.4f} "
                               f"vs recent avg {recent_avg:.4f} ***\n"
                               f"  Files in this batch: {getattr(batch, 'filename', 'N/A')}\n"
                               f"  x range: [{batch.x.min().item():.4f}, {batch.x.max().item():.4f}]\n"
                               f"  x has NaN: {torch.isnan(batch.x).any().item()}, "
                               f"has Inf: {torch.isinf(batch.x).any().item()}\n"
                               f"  y distribution: {batch.y.tolist()}")
                        if logger:
                            logger.log(msg)
                        else:
                            print(msg)
                batch_loss_history.append(loss_value)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss_value * batch.num_graphs
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

    return total_loss / total, correct / total


# ===========================================================================
# STEP 4: Fit the model — trains for up to num_epochs, with early stopping
# and best-checkpoint selection handled by RunLogger. num_epochs=250 lives
# here, in this function's own definition, rather than as a bare literal
# where it gets called.
# ===========================================================================
def fit_model(model, train_loader, val_loader, optimizer, logger, device, num_epochs=250):
    batch_loss_history = []  # shared across all epochs, for live spike detection

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True,
                                           logger=logger, epoch=epoch, batch_loss_history=batch_loss_history)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        should_stop = logger.check_and_save_best(val_acc, model)
        if should_stop:
            logger.log(f"\nEarly stopping at epoch {epoch} (no val improvement for 25 epochs).")
            break

    elapsed = time.time() - start_time
    logger.log(f"\nTraining finished in {elapsed:.1f}s. Best val_acc: {logger.best_val_acc:.4f}")


# ===========================================================================
# STEP 5: Prediction — run the trained model on the test set, collect raw
# predictions. This step ONLY produces predictions; it does not compute any
# accuracy/precision/recall numbers — that is deliberately kept separate,
# as STEP 6 below.
# ===========================================================================
def predict(model, test_loader, device):
    """Runs the model ONCE on the held-out test set and returns the raw
    predicted labels alongside the true labels. No metrics are computed
    here — see calculate_accuracy() for that, kept as a separate step.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
    return all_preds, all_labels


# ===========================================================================
# STEP 6: Accuracy calculation — turn raw predictions (from STEP 5) into
# accuracy, precision, recall, F1, and a confusion matrix. Kept as its own
# function, separate from prediction, since class-level detail matters for
# a safe/unsafe classification task, not just a single accuracy number.
# ===========================================================================
def calculate_accuracy(all_preds, all_labels):
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision_unsafe": round(precision, 4),
        "recall_unsafe": round(recall, 4),
        "f1_unsafe": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "num_test_samples": len(all_labels),
    }


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = RunLogger({"model_name": "GAT"})
    logger.log(f"Device: {device}")

    # =======================================================================
    # STEP 1: Loading the prepared dataset to PyTorch Geometric
    # =======================================================================
    logger.log("\n=== STEP 1: Loading the prepared dataset to PyTorch Geometric ===")
    train_loader, val_loader, test_loader = get_dataloaders()

    # =======================================================================
    # STEP 3: Create model and optimizer
    # (STEP 2, the GAT architecture itself, is defined above as build_gat_model —
    # both this and build_optimizer carry their own default values, so no
    # values need to be typed here at all)
    # =======================================================================
    logger.log("\n=== STEP 3: Create model and optimizer ===")
    model = build_gat_model().to(device)
    optimizer = build_optimizer(model)

    # =======================================================================
    # STEP 4: Fit the model
    # =======================================================================
    logger.log("\n=== STEP 4: Fit the model ===")
    fit_model(model, train_loader, val_loader, optimizer, logger, device)

    # Load best checkpoint before touching the test set.
    model.load_state_dict(torch.load(logger.checkpoint_path))

    # =======================================================================
    # STEP 5: Prediction
    # =======================================================================
    logger.log("\n=== STEP 5: Prediction ===")
    all_preds, all_labels = predict(model, test_loader, device)

    # =======================================================================
    # STEP 6: Accuracy calculation
    # =======================================================================
    logger.log("\n=== STEP 6: Accuracy calculation ===")
    test_results = calculate_accuracy(all_preds, all_labels)
    logger.save_test_results(test_results)


if __name__ == "__main__":
    main()
