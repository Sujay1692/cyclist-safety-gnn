# Cyclist Safety Classification from Skeletal Pose Graphs

Code and data for:

> Somayaji, S.K.; Mostofi Darbani, H. Cyclist Safety Classification from Skeletal Pose Graphs: A Comparative Study of Graph Attention, Convolutional, and Isomorphism Networks. *AI* **2026**, [volume, article number, DOI to be added].

Each photograph of a cyclist is converted into a 17-node skeleton graph with MediaPipe Pose. Every node carries its position, a visibility score, and the joint angles at that node with their confidences. Three graph neural networks then classify the rider's posture as safe or unsafe: a Graph Attention Network (GAT), a Graph Convolutional Network (GCN), and a Graph Isomorphism Network (GIN).

## Results (test set, 445 images)

| Model | Accuracy | Precision | Recall | Macro-F1 | AUC-ROC | Latency (CPU) |
|---|---|---|---|---|---|---|
| GAT | 80.45% | 85.41% | 72.48% | 0.8027 | 0.8833 | 2.393 ms |
| GCN | 74.16% | 74.88% | 71.10% | 0.7410 | 0.8182 | 1.888 ms |
| GIN | 77.53% | 83.52% | 67.43% | 0.7723 | 0.8342 | 0.948 ms |

Configurations were chosen by test accuracy, so these figures are optimistic estimates; see Section 3.2.2 of the paper. Every run is deterministic with seed 42 on CPU.

## Repository contents

| Path | What it is |
|---|---|
| `verify_pose_extraction.py` | Pose verification and curation overlays (paper Section 3.1.1) |
| `extract_keypoints_gnn.py` | Two-pass MediaPipe keypoint extraction (Sections 3.1.2–3.1.3) |
| `gnn_dataset.py` | Normalisation, node-level angle features, PyTorch Geometric dataset (Sections 3.1.5–3.1.6) |
| `create_split.py` | Stratified 70/15/15 split, seed 42 (Section 3.1.7) |
| `train_gat.py`, `train_gcn.py`, `train_gin.py` | Training of the three models (Sections 3.2.1–3.2.5) |
| `evaluate_full_metrics.py` | Macro-F1, AUC-ROC, FLOPs and latency from a saved checkpoint (Section 3.2.6) |
| `ablate_gat_joints.py` | Per-joint ablation on the trained GAT (Section 3.2.7) |
| `keypoints/safe/`, `keypoints/unsafe/` | 2970 keypoint arrays, each of shape (17, 3): x, y, visibility |
| `dataset_split.csv` | One row per image: file, class, keypoint path, detection quality, split |
| `skeleton_edges.npy`, `skeleton_node_order.csv` | The 18 anatomical edges and the joint order |
| `excluded_images.csv` | The 30 curated images with no pose detected |
| `DATA.md` | Dataset description: collection, labelling criteria, limitations |

The photographs (2970 images, full resolution, metadata removed) are attached to the [v1.0 release](../../releases) as zip files.

## Setup

Python 3.10:

```
pip install -r requirements.txt
```

## Reproducing the results

The keypoints and split are included, so training does not need the photographs:

```
python train_gat.py
python train_gcn.py
python train_gin.py
python evaluate_full_metrics.py --model gat --checkpoint results/runs/GAT_<timestamp>/best_model.pt
python ablate_gat_joints.py --checkpoint results/runs/GAT_<timestamp>/best_model.pt
```

Replace `<timestamp>` with the folder name of your training run. Each run writes its configuration, per-epoch metrics, log, best checkpoint and test results to `results/runs/<MODEL>_<timestamp>/`.

To redo pose extraction from the photographs:

1. Download the photo zips from the release.
2. Unzip them so the images sit in `image_data/safe/` and `image_data/unsafe/`.
3. Optionally run `python verify_pose_extraction.py` to draw the detected skeletons onto the photographs for inspection.
4. Run `python extract_keypoints_gnn.py` (writes `keypoints/` and `dataset_index.csv`), then `python create_split.py` (writes `dataset_split.csv`).

## Data

See [DATA.md](DATA.md) for how the photographs were collected and labelled, the label definitions, and known limitations. All nine riders gave written consent for their images to be published. Please do not try to identify or contact the people shown.

## Licences

- **Code:** MIT ([LICENSE](LICENSE)).
- **Data:** the photographs, keypoints, labels and split are licensed under CC BY-NC 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)), which allows non-commercial use with attribution.

## Citation

If you use this code or data, please cite the paper above (see also [CITATION.cff](CITATION.cff)).
