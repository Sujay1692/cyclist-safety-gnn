"""
Per-Joint Ablation Study — GAT Only
------------------------------------------------
Answers: which of the 17 joints does the trained GAT model actually rely
on for its safe/unsafe decision?

Method: for each joint, zero out that joint's entire (x, y, visibility,
angle1, angle1_conf, angle2, angle2_conf) feature vector across every
graph in the test set, keeping the graph structure and every other joint
untouched, then re-run the ALREADY-TRAINED model on this ablated test set.
The drop in accuracy relative to the real, un-ablated baseline indicates
how much the model relies on that specific joint — a bigger drop means
the model leans on that joint more heavily.

Reuses build_gat_model(), predict(), and calculate_accuracy() directly
from train_gat.py, rather than reimplementing any of that logic — the
only genuinely new code here is the ablation loop itself, since no
existing function in the project does this.

Requires a trained checkpoint (best_model.pt) from a GAT run matching the
confirmed-final configuration (128 hidden channels, 8 heads, 5 layers).

Usage:
    python ablate_gat_joints.py --checkpoint results/runs/GAT_<timestamp>/best_model.pt
"""

import argparse
import copy

import torch
from torch_geometric.loader import DataLoader

from gnn_dataset import CyclistPoseDataset, load_edge_index
from train_gat import build_gat_model, predict, calculate_accuracy

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def build_ablated_loader(test_ds, joint_idx, batch_size=32):
    """Returns a DataLoader identical to the real test set, except every
    graph's `joint_idx`-th node has its 7-dimensional feature vector
    zeroed out. Since every graph in this project always has exactly 17
    nodes in the same fixed anatomical order, this is a clean, direct
    per-graph reshape — no manual offset tracking needed.
    """
    ablated_list = []
    for data in test_ds:
        ablated = copy.deepcopy(data)
        ablated.x[joint_idx, :] = 0.0
        ablated_list.append(ablated)
    return DataLoader(ablated_list, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the trained GAT best_model.pt")
    parser.add_argument("--csv_path", default="dataset_split.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================================================================
    # Load the real, held-out test set (unchanged, same as every other
    # evaluation in this project) and the already-trained model.
    # =======================================================================
    edge_index = load_edge_index()
    test_ds = CyclistPoseDataset(args.csv_path, "test", edge_index)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_gat_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # =======================================================================
    # Baseline — should reproduce the confirmed 80.45% result exactly.
    # =======================================================================
    baseline_preds, baseline_labels = predict(model, test_loader, device)
    baseline = calculate_accuracy(baseline_preds, baseline_labels)
    print(f"Baseline (no ablation) test accuracy: {baseline['accuracy']*100:.2f}%\n")

    # =======================================================================
    # Ablate one joint at a time, re-evaluate, record the accuracy drop.
    # =======================================================================
    results = []
    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        ablated_loader = build_ablated_loader(test_ds, joint_idx)
        preds, labels = predict(model, ablated_loader, device)
        metrics = calculate_accuracy(preds, labels)
        drop = baseline["accuracy"] - metrics["accuracy"]
        results.append((joint_name, metrics["accuracy"], drop))
        print(f"  {joint_name:16s} ablated -> accuracy {metrics['accuracy']*100:5.2f}%  "
              f"(drop: {drop*100:+.2f} pts)")

    # =======================================================================
    # Rank joints by importance (biggest accuracy drop = most important)
    # =======================================================================
    results.sort(key=lambda r: r[2], reverse=True)
    print("\n=== Joints ranked by importance (biggest accuracy drop first) ===")
    for rank, (joint_name, acc, drop) in enumerate(results, start=1):
        print(f"{rank:2d}. {joint_name:16s} drop: {drop*100:+.2f} pts  (ablated accuracy: {acc*100:.2f}%)")


if __name__ == "__main__":
    main()
