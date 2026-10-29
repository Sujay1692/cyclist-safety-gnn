"""
GIN Training Script — with Full Run Logging
------------------------------------------------
Trains a GIN (Graph Isomorphism Network) classifier on the extracted
cyclist pose-keypoint graphs (safe vs. unsafe).

Built ENTIRELY from PyTorch Geometric's own built-in components, with no
custom nn.Module subclass and no manually defined MLP:
torch_geometric.nn.models.GIN builds
its own internal per-layer MLPs automatically. Assembled with pooling and
the final classifier via torch_geometric.nn.Sequential, same approach as
train_gat.py and train_gcn.py.

Input: (17, 7) node features per graph — x, y, visibility, plus 4 joint-
angle values (with paired confidences) placed directly at their
anatomically relevant nodes (nose, shoulders, hips, elbows, knees). No
separate graph-level feature vector.

Every run is logged to its own timestamped folder under results/runs/:
    results/runs/GIN_<timestamp>/
        config.json, metrics.csv, train_log.txt, best_model.pt, test_results.json

Configuration values are NOT centralized in one CONFIG dict — each value
lives as a default parameter directly in the function that uses it
(build_gin_model, build_optimizer, fit_model), so main() just calls each
function plainly.

Usage:
    python train_gin.py
"""

import os
import json
import csv
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, global_mean_pool
from torch_geometric.nn.models import GIN as PyGGIN

from gnn_dataset import get_dataloaders

RESULTS_ROOT = "results/runs"


class RunLogger:
    """Identical to train_gat.py/train_gcn.py's RunLogger — kept consistent
    so results across GAT/GCN/GIN are directly comparable in structure."""

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
# STEP 2: Create Model GIN — the architecture definition
#
# Built entirely from PyTorch Geometric built-ins: torch_geometric.nn.models.GIN
# as the backbone, assembled with pooling and a Linear classifier via
# torch_geometric.nn.Sequential. No custom nn.Module subclass, and no
# manually defined MLP — the built-in GIN constructs and manages its own
# internal per-layer MLPs automatically.
#
# Defaults here (32 channels, 4 layers, mean pooling, no Jumping Knowledge)
# are the adopted GIN configuration reported in the paper (Appendix A,
# Table A3).
# ===========================================================================
def build_gin_model(in_channels=7, hidden_channels=32, num_layers=4, dropout=0.1, num_classes=2,):
    backbone = PyGGIN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        norm="BatchNorm",
        act="elu",
    )
    model = Sequential("x, edge_index, batch", [
        (backbone, "x, edge_index -> x"),
        (global_mean_pool, "x, batch -> x"),   # mean pooling, as in GAT and GCN
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

            # Same live spike-detection instrumentation as in train_gat.py.
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
# STEP 4: Fit the model
# ===========================================================================
def fit_model(model, train_loader, val_loader, optimizer, logger, device, num_epochs=150):
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
# STEP 5: Prediction
# ===========================================================================
def predict(model, test_loader, device):
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
# STEP 6: Accuracy calculation
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

    logger = RunLogger({"model_name": "GIN"})
    logger.log(f"Device: {device}")

    # =======================================================================
    # STEP 1: Loading the prepared dataset to PyTorch Geometric
    # =======================================================================
    logger.log("\n=== STEP 1: Loading the prepared dataset to PyTorch Geometric ===")
    train_loader, val_loader, test_loader = get_dataloaders()

    # =======================================================================
    # STEP 3: Create model and optimizer
    # (STEP 2, the GIN architecture itself, is defined above as build_gin_model)
    # =======================================================================
    logger.log("\n=== STEP 3: Create model and optimizer ===")
    model = build_gin_model().to(device)
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
