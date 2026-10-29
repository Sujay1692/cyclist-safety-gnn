"""
PyTorch Geometric Dataset — Cyclist Pose Keypoints (v2: node-level angles)
------------------------------------------------------------------------------
Loads the extracted (17, 3) keypoint arrays and expands them into (17, 7)
node feature matrices, with joint angles computed and placed directly at
the anatomically relevant NODE — not appended as a separate graph-level
vector. This lets the angle information flow through the GAT/GCN/GIN
message-passing layers themselves, not just the final classifier.

DESIGN (finalized after discussion — see project log):
    Every node:     (x, y, visibility, angle1, angle1_conf, angle2, angle2_conf)
    nose:           angle1 = angle(nose->left_shoulder, nose->right_shoulder)
    left/right shoulder: angle1 = angle(shoulder->elbow, shoulder->hip)
                          angle2 = angle(shoulder->hip, shoulder->shoulder)
    left/right hip:  angle1 = angle(hip->knee, hip->hip)
                      angle2 = angle(hip->hip, hip->shoulder)
    left/right elbow: angle1 = elbow bend angle(shoulder->elbow, elbow->wrist)
    left/right knee:  angle1 = knee bend angle(hip->knee, knee->ankle)
    all other nodes (eyes, ears, wrists, ankles): angle1=angle2=0, confidences=0
    (torso_lean and head_turn_offset from the earlier design are RETIRED —
    replaced by the nose angle and the shoulder/hip angles above, and no
    graph-level feature vector remains at all.)

Implementation note (per project decision): every geometric operation uses
PyTorch's own built-in tensor functions (torch.linalg.norm, torch.clamp,
torch.dot, torch.acos, torch.min) rather than NumPy equivalents. The ONE
function defined from scratch is `_angle_between()`, since no built-in
computes "angle between two arbitrary vectors" directly — everything
inside it is still built from torch primitives.

Usage as a script (smoke test):
    python gnn_dataset.py

Usage as a module:
    from gnn_dataset import CyclistPoseDataset, get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_SPLIT_CSV = "dataset_split.csv"
EDGES_FILE = "skeleton_edges.npy"

CLASS_TO_LABEL = {"safe": 0, "unsafe": 1}

# Node indices matching KEY_LANDMARKS order in extract_keypoints_gnn.py:
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

NUM_NODES = 17
NODE_FEATURE_DIM = 7   # x, y, visibility, angle1, angle1_conf, angle2, angle2_conf


def normalize_keypoints(features):
    """Centers the skeleton on the hip midpoint and scales by torso length
    (shoulder-midpoint-to-hip-midpoint distance), so the network learns POSE
    SHAPE rather than where-in-the-frame or how-large the rider happens to
    be. `features` is a torch.Tensor of shape (17, 3): [x, y, visibility].

    Built entirely from torch built-ins: torch.linalg.norm, torch.clamp.
    """
    features = features.clone()
    hip_mid = (features[LEFT_HIP, :2] + features[RIGHT_HIP, :2]) / 2.0
    shoulder_mid = (features[LEFT_SHOULDER, :2] + features[RIGHT_SHOULDER, :2]) / 2.0

    scale = torch.linalg.norm(shoulder_mid - hip_mid)
    scale = torch.clamp(scale, min=1e-3)   # guard against a literal division-by-zero crash

    features[:, :2] = (features[:, :2] - hip_mid) / scale
    features[:, :2] = torch.clamp(features[:, :2], -10.0, 10.0)   # guard against amplification
    return features


def _angle_between(v1, v2):
    """The one function defined from scratch, since PyTorch has no built-in
    for 'angle between two arbitrary vectors'. Every operation INSIDE it is
    still a torch built-in: torch.linalg.norm, torch.dot, torch.clamp,
    torch.acos. Returns a 0-dim torch.Tensor (radians, 0 to pi).
    """
    n1, n2 = torch.linalg.norm(v1), torch.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return torch.tensor(0.0)
    cos_angle = torch.clamp(torch.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return torch.acos(cos_angle)


def build_node_features(normalized):
    """Expands the (17, 3) normalized [x, y, visibility] array into the
    final (17, 7) node feature matrix: [x, y, visibility, angle1,
    angle1_conf, angle2, angle2_conf] — with angles computed and placed at
    their anatomically relevant node, per the design above. Nodes with no
    applicable angle keep 0 in those four slots (torch.zeros default).

    Built from torch built-ins throughout: torch.zeros, tensor indexing,
    torch.min for the confidence (minimum-visibility) rule. `_angle_between`
    is the only custom-defined piece, per project decision.
    """
    def pt(i):
        return normalized[i, :2]

    def vis(i):
        return normalized[i, 2]

    out = torch.zeros((NUM_NODES, NODE_FEATURE_DIM), dtype=torch.float32)
    out[:, :3] = normalized   # x, y, visibility copied through unchanged for every node

    # nose: angle between (nose->left_shoulder) and (nose->right_shoulder)
    v1 = pt(LEFT_SHOULDER) - pt(NOSE)
    v2 = pt(RIGHT_SHOULDER) - pt(NOSE)
    out[NOSE, 3] = _angle_between(v1, v2) / torch.pi
    out[NOSE, 4] = torch.min(vis(LEFT_SHOULDER), vis(RIGHT_SHOULDER))

    # shoulders: angle1 = (shoulder->elbow, shoulder->hip), angle2 = (shoulder->hip, shoulder->shoulder)
    for shoulder, elbow, hip, other_shoulder in [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_HIP, RIGHT_SHOULDER),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HIP, LEFT_SHOULDER),
    ]:
        to_elbow = pt(elbow) - pt(shoulder)
        to_hip = pt(hip) - pt(shoulder)
        to_other_shoulder = pt(other_shoulder) - pt(shoulder)

        out[shoulder, 3] = _angle_between(to_elbow, to_hip) / torch.pi
        out[shoulder, 4] = torch.min(torch.min(vis(shoulder), vis(elbow)), vis(hip))
        out[shoulder, 5] = _angle_between(to_hip, to_other_shoulder) / torch.pi
        out[shoulder, 6] = torch.min(torch.min(vis(shoulder), vis(hip)), vis(other_shoulder))

    # hips: angle1 = (hip->knee, hip->hip), angle2 = (hip->hip, hip->shoulder)
    for hip, knee, other_hip, shoulder in [
        (LEFT_HIP, LEFT_KNEE, RIGHT_HIP, LEFT_SHOULDER),
        (RIGHT_HIP, RIGHT_KNEE, LEFT_HIP, RIGHT_SHOULDER),
    ]:
        to_knee = pt(knee) - pt(hip)
        to_other_hip = pt(other_hip) - pt(hip)
        to_shoulder = pt(shoulder) - pt(hip)

        out[hip, 3] = _angle_between(to_knee, to_other_hip) / torch.pi
        out[hip, 4] = torch.min(torch.min(vis(hip), vis(knee)), vis(other_hip))
        out[hip, 5] = _angle_between(to_other_hip, to_shoulder) / torch.pi
        out[hip, 6] = torch.min(torch.min(vis(hip), vis(other_hip)), vis(shoulder))

    # elbows: single bend angle (shoulder->elbow, elbow->wrist), angle2 slot unused (stays 0)
    for shoulder, elbow, wrist in [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ]:
        upper_arm = pt(shoulder) - pt(elbow)
        forearm = pt(wrist) - pt(elbow)
        out[elbow, 3] = _angle_between(upper_arm, forearm) / torch.pi
        out[elbow, 4] = torch.min(torch.min(vis(shoulder), vis(elbow)), vis(wrist))

    # knees: single bend angle (hip->knee, knee->ankle), angle2 slot unused (stays 0)
    for hip, knee, ankle in [
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ]:
        thigh = pt(hip) - pt(knee)
        shin = pt(ankle) - pt(knee)
        out[knee, 3] = _angle_between(thigh, shin) / torch.pi
        out[knee, 4] = torch.min(torch.min(vis(hip), vis(knee)), vis(ankle))

    return out


def load_edge_index():
    """Loads the shared (18, 2) skeleton adjacency and converts it to
    PyTorch Geometric's expected (2, num_edges) format, made undirected by
    storing each edge in both directions. Built from torch built-ins:
    torch.from_numpy, tensor indexing (equivalent to torch.flip), torch.cat.
    """
    edges = torch.from_numpy(np.load(EDGES_FILE)).long()   # shape (18, 2)
    edges_reversed = edges[:, [1, 0]]                        # swap columns for reverse direction
    all_edges = torch.cat([edges, edges_reversed], dim=0)     # shape (36, 2)
    edge_index = all_edges.t().contiguous()                    # shape (2, 36)
    return edge_index


class CyclistPoseDataset(InMemoryDataset):
    """A PyTorch Geometric InMemoryDataset built from a dataset_split.csv
    manifest. Filters to a single split ('train', 'val', or 'test') at
    construction time. Node features are now (17, 7) — see module docstring.
    """

    def __init__(self, csv_path, split_name, edge_index):
        super().__init__(root=None)
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split_name].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split_name}' in {csv_path}. "
                              f"Check that create_split.py has been run and the split name is correct.")

        data_list = []
        missing_files = 0
        for _, row in df.iterrows():
            keypoint_path = row["keypoint_path"]
            if not os.path.exists(keypoint_path):
                missing_files += 1
                continue

            raw = torch.from_numpy(np.load(keypoint_path)).float()   # shape (17, 3)
            normalized = normalize_keypoints(raw)
            x = build_node_features(normalized)                        # shape (17, 7)
            y = torch.tensor([CLASS_TO_LABEL[row["class"]]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.filename = row["filename"]                 # kept for later error-analysis / stratified eval
            data_list.append(data)

        if missing_files:
            print(f"WARNING: {missing_files} keypoint files listed in the CSV were not found on disk and were skipped.")

        self.data, self.slices = self.collate(data_list)
        print(f"[{split_name}] loaded {len(data_list)} samples.")


def get_dataloaders(csv_path=DATASET_SPLIT_CSV, batch_size=32, shuffle_train=True):
    """Convenience function: builds train/val/test datasets and DataLoaders
    in one call, sharing the same skeleton edge_index across all three.
    """
    edge_index = load_edge_index()

    train_ds = CyclistPoseDataset(csv_path, "train", edge_index)
    val_ds = CyclistPoseDataset(csv_path, "val", edge_index)
    test_ds = CyclistPoseDataset(csv_path, "test", edge_index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Smoke test: build the loaders and print the shape of one batch, so you
    # can visually confirm everything is wired correctly before training.
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  x shape:          {batch.x.shape}       (nodes_in_batch, 7)")
    print(f"  edge_index shape: {batch.edge_index.shape}  (2, edges_in_batch)")
    print(f"  y shape:          {batch.y.shape}       (graphs_in_batch,)")
    print(f"  num_graphs:       {batch.num_graphs}")
