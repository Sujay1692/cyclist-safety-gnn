"""
Full Metric Suite — Accuracy, Macro-F1, AUC-ROC, Latency, FLOPs
------------------------------------------------------------------------------
One combined script covering every comparison metric across the three
trained GNN models. Reuses each model's own confirmed-final
build_*_model() function to reconstruct the architecture before loading
its real, already-trained weights — no retraining happens here.

Metrics, and how each is genuinely computed (not estimated):
  - Accuracy, Macro-F1: scikit-learn's own accuracy_score / f1_score,
    computed from real predictions on the real held-out test set.
  - AUC-ROC: scikit-learn's roc_auc_score, using real per-sample
    probability scores (softmax output), not just hard predictions.
  - FLOPs: PyTorch's own built-in torch.utils.flop_counter.FlopCounterMode,
    which traces actual tensor operations rather than relying on
    layer-type-specific hooks — verified directly to work correctly on
    PyTorch Geometric's message-passing layers before being used here.
  - Latency: real wall-clock time (Python's time.perf_counter()),
    averaged over many repeated single-graph forward passes, with a
    warmup period first to exclude one-time startup overhead.

Usage:
    python evaluate_full_metrics.py --model gat --checkpoint results/runs/GAT_.../best_model.pt
    python evaluate_full_metrics.py --model gcn --checkpoint results/runs/GCN_.../best_model.pt
    python evaluate_full_metrics.py --model gin --checkpoint results/runs/GIN_.../best_model.pt
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from gnn_dataset import CyclistPoseDataset, load_edge_index

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["gat", "gcn", "gin"])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--csv_path", default="dataset_split.csv")
parser.add_argument("--num_runs", type=int, default=200, help="Timed forward passes for latency, averaged")
parser.add_argument("--warmup_runs", type=int, default=20)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Build the correct model (each script's own confirmed-final defaults) and
# load the real, already-trained weights.
# ---------------------------------------------------------------------------
if args.model == "gat":
    from train_gat import build_gat_model as build_model
elif args.model == "gcn":
    from train_gcn import build_gcn_model as build_model
else:
    from train_gin import build_gin_model as build_model

model = build_model().to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

edge_index = load_edge_index()

# ===========================================================================
# PART 1: Accuracy / Macro-F1 / AUC-ROC — real predictions on the real,
# held-out test set.
# ===========================================================================
test_ds = CyclistPoseDataset(args.csv_path, "test", edge_index)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

all_probs_unsafe, all_preds, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)
        all_probs_unsafe.extend(probs[:, 1].cpu().tolist())
        all_preds.extend(out.argmax(dim=1).cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
auc_roc = roc_auc_score(all_labels, all_probs_unsafe)

# ===========================================================================
# PART 2: FLOPs and Latency — measured on a single graph (17 nodes),
# matching real single-image deployment, independent of the test set.
# ===========================================================================
edge_index_device = edge_index.to(device)
x = torch.randn(17, 7, device=device)
batch_single = torch.zeros(17, dtype=torch.long, device=device)

with FlopCounterMode(display=False) as fc:
    with torch.no_grad():
        _ = model(x, edge_index_device, batch_single)
total_flops = fc.get_total_flops()

with torch.no_grad():
    for _ in range(args.warmup_runs):
        _ = model(x, edge_index_device, batch_single)
    start = time.perf_counter()
    for _ in range(args.num_runs):
        _ = model(x, edge_index_device, batch_single)
    elapsed = time.perf_counter() - start
avg_latency_ms = (elapsed / args.num_runs) * 1000

# ===========================================================================
# Report
# ===========================================================================
print(f"\nModel: {args.model.upper()}")
print(f"Checkpoint: {args.checkpoint}")
print(f"Device: {device}")
print(f"Test samples: {len(all_labels)}")
print("-" * 50)
print(f"Accuracy:   {accuracy*100:.2f}%")
print(f"Macro-F1:   {macro_f1:.4f}")
print(f"AUC-ROC:    {auc_roc:.4f}")
print("-" * 50)
print(f"FLOPs (single graph, 1 forward pass): {total_flops:,}")
print(f"Latency (per graph, {args.num_runs} runs, {args.warmup_runs} warmup): {avg_latency_ms:.3f} ms")
