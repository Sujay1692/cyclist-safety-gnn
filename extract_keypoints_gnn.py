"""
GNN Keypoint Extraction Script
------------------------------------------
Converts the final, verified image_data/ (3,000 images) into the actual
input format for GAT / GCN / GIN: a per-image node feature matrix, plus
one shared skeleton adjacency structure used by all images.

Approach: confidence-weighted pass-through. Every key joint's
(x, y, visibility) from MediaPipe is kept as-is, including low-confidence
ones. No coordinates are fabricated.

Detection pipeline: PASS 1 (normal) + PASS 2 (lenient retry), same as
verify_pose_extraction.py. Images that fail BOTH passes are automatically
excluded — logged in excluded_images.csv, not silently dropped.

FLIP AUGMENTATION: DISABLED. Horizontal-flip augmentation was evaluated on
GIN and not adopted (see the paper). The flip code is kept so that experiment
can be repeated; with ENABLE_FLIP_AUGMENTATION = False, only ONE .npy per
source image is produced, so no pair_id bookkeeping is needed for splitting.

Output structure:
    keypoints/
        safe/<filename_stem>.npy      <- shape (17, 3): [x, y, visibility] per joint
        unsafe/<filename_stem>.npy
    skeleton_edges.npy                 <- shape (18, 2): shared adjacency (same for every image)
    skeleton_node_order.csv            <- which node index = which joint name (reference)
    dataset_index.csv                  <- one row per successfully extracted image
    excluded_images.csv                <- images with no pose detected in either pass

Usage:
    python extract_keypoints_gnn.py
"""

import os
import csv
import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_BASE_FOLDER = "image_data"
CLASS_FOLDERS = ["safe", "unsafe"]
KEYPOINTS_OUTPUT_FOLDER = "keypoints"
DATASET_INDEX_CSV = "dataset_index.csv"
EXCLUDED_CSV = "excluded_images.csv"
EDGES_FILE = "skeleton_edges.npy"
NODE_ORDER_CSV = "skeleton_node_order.csv"

USE_CONTRAST_ENHANCEMENT = True

ENABLE_RETRY_PASS = True
RETRY_MIN_DETECTION_CONFIDENCE = 0.15
RETRY_FORCE_CONTRAST_ENHANCEMENT = True

# Flip augmentation is not used for any reported result.
ENABLE_FLIP_AUGMENTATION = False

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")

# 17-point set: 12 body joints + 5 head joints (nose, eyes, ears).
# Order here defines the node order in every output array.
KEY_LANDMARKS = [
    (0, "nose"),
    (2, "left_eye"), (5, "right_eye"),
    (7, "left_ear"), (8, "right_ear"),
    (11, "left_shoulder"), (12, "right_shoulder"),
    (13, "left_elbow"), (14, "right_elbow"),
    (15, "left_wrist"), (16, "right_wrist"),
    (23, "left_hip"), (24, "right_hip"),
    (25, "left_knee"), (26, "right_knee"),
    (27, "left_ankle"), (28, "right_ankle"),
]
MEDIAPIPE_IDX_TO_NODE_IDX = {mp_idx: node_idx for node_idx, (mp_idx, _) in enumerate(KEY_LANDMARKS)}

# Shared skeleton adjacency (node indices 0-16, matching KEY_LANDMARKS order)
SKELETON_EDGES = [
    (0, 1), (0, 2),         # nose - left/right eye
    (1, 3), (2, 4),         # left eye - left ear, right eye - right ear
    (0, 5), (0, 6),         # nose - left/right shoulder
    (5, 6),                 # left shoulder - right shoulder
    (5, 7), (7, 9),         # left shoulder - left elbow - left wrist
    (6, 8), (8, 10),        # right shoulder - right elbow - right wrist
    (5, 11), (6, 12),       # left/right shoulder - left/right hip
    (11, 12),               # left hip - right hip
    (11, 13), (13, 15),     # left hip - left knee - left ankle
    (12, 14), (14, 16),     # right hip - right knee - right ankle
]

# Left/right NODE-index swap pairs, used only if flip augmentation is re-enabled later.
FLIP_SWAP_PAIRS = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]

mp_pose = mp.solutions.pose


def enhance_contrast(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)


def run_detection(image_bgr, pose_estimator, use_contrast):
    detection_input = enhance_contrast(image_bgr) if use_contrast else image_bgr
    image_rgb = cv2.cvtColor(detection_input, cv2.COLOR_BGR2RGB)
    return pose_estimator.process(image_rgb)


def extract_node_features(result):
    """Returns a (17, 3) array: [x, y, visibility] per key joint, in
    KEY_LANDMARKS order. Confidence-weighted pass-through — kept as-is
    regardless of confidence, no fabrication, no dropping.
    """
    landmarks = result.pose_landmarks.landmark
    features = np.zeros((len(KEY_LANDMARKS), 3), dtype=np.float32)
    for node_idx, (mp_idx, _) in enumerate(KEY_LANDMARKS):
        lm = landmarks[mp_idx]
        features[node_idx] = [lm.x, lm.y, lm.visibility]
    return features


def flip_keypoints(features):
    """Only used if ENABLE_FLIP_AUGMENTATION is re-enabled. Mirrors x and
    swaps left/right node values (position + visibility)."""
    flipped = features.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    for a, b in FLIP_SWAP_PAIRS:
        flipped[[a, b]] = flipped[[b, a]]
    return flipped


def save_skeleton_reference():
    edges_array = np.array(SKELETON_EDGES, dtype=np.int64)
    np.save(EDGES_FILE, edges_array)
    with open(NODE_ORDER_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "joint_name", "mediapipe_landmark_index"])
        for node_idx, (mp_idx, name) in enumerate(KEY_LANDMARKS):
            writer.writerow([node_idx, name, mp_idx])
    print(f"Skeleton adjacency saved to {EDGES_FILE} (shape {edges_array.shape})")
    print(f"Node order reference saved to {NODE_ORDER_CSV}")


def save_keypoint_entry(features, class_label, filename, is_flipped, pair_id, rescued):
    stem = os.path.splitext(filename)[0]
    suffix = "_flip" if is_flipped else ""
    out_filename = f"{stem}{suffix}.npy"
    npy_path = os.path.join(KEYPOINTS_OUTPUT_FOLDER, class_label, out_filename)
    np.save(npy_path, features)

    num_detected = int((features[:, 2] >= 0.5).sum())
    avg_vis = float(features[:, 2].mean())

    return {
        "filename": out_filename, "class": class_label, "keypoint_path": npy_path,
        "pair_id": pair_id, "is_flipped": is_flipped,
        "num_key_landmarks_detected": num_detected, "avg_visibility": round(avg_vis, 3),
        "rescued_by_retry": rescued,
    }


def process_folder():
    all_images = []
    for class_label in CLASS_FOLDERS:
        class_dir = os.path.join(INPUT_BASE_FOLDER, class_label)
        if not os.path.isdir(class_dir):
            print(f"WARNING: '{class_dir}' not found — skipping.")
            continue
        for f in sorted(os.listdir(class_dir)):
            if f.lower().endswith(VALID_EXTENSIONS):
                all_images.append((class_label, f, os.path.join(class_dir, f)))
        os.makedirs(os.path.join(KEYPOINTS_OUTPUT_FOLDER, class_label), exist_ok=True)

    if not all_images:
        print(f"No images found under '{INPUT_BASE_FOLDER}'. Check the path.")
        return

    print(f"Found {len(all_images)} images. Flip augmentation: {'ENABLED' if ENABLE_FLIP_AUGMENTATION else 'DISABLED'}.")
    print("Extracting keypoints (PASS 1)...\n")

    dataset_index = []
    excluded = []
    retry_queue = []

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                       min_detection_confidence=0.5) as pose:

        for idx, (class_label, filename, filepath) in enumerate(all_images, 1):
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                print(f"[{idx}/{len(all_images)}] {class_label}/{filename}: COULD NOT READ FILE")
                excluded.append({"filename": filename, "class": class_label, "reason": "FILE_UNREADABLE"})
                continue

            result = run_detection(image_bgr, pose, USE_CONTRAST_ENHANCEMENT)
            if result.pose_landmarks is None:
                retry_queue.append((class_label, filename, filepath, image_bgr))
                continue

            features = extract_node_features(result)
            pair_id = filename
            entry = save_keypoint_entry(features, class_label, filename, is_flipped=False, pair_id=pair_id, rescued=False)
            dataset_index.append(entry)
            print(f"[{idx}/{len(all_images)}] {class_label}/{filename}: extracted | "
                  f"{entry['num_key_landmarks_detected']}/{len(KEY_LANDMARKS)} confident | avg vis {entry['avg_visibility']:.2f}")

            if ENABLE_FLIP_AUGMENTATION:
                flipped_features = flip_keypoints(features)
                flip_entry = save_keypoint_entry(flipped_features, class_label, filename, is_flipped=True, pair_id=pair_id, rescued=False)
                dataset_index.append(flip_entry)

    # PASS 2 retry
    if retry_queue and ENABLE_RETRY_PASS:
        print(f"\n{'=' * 60}\nPASS 2 (retry): {len(retry_queue)} images had no pose detected in PASS 1.\n{'=' * 60}\n")
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                           min_detection_confidence=RETRY_MIN_DETECTION_CONFIDENCE) as retry_pose:

            for idx, (class_label, filename, filepath, image_bgr) in enumerate(retry_queue, 1):
                result = run_detection(image_bgr, retry_pose, RETRY_FORCE_CONTRAST_ENHANCEMENT)
                if result.pose_landmarks is None:
                    print(f"[retry {idx}/{len(retry_queue)}] {class_label}/{filename}: STILL NO POSE — excluded")
                    excluded.append({"filename": filename, "class": class_label, "reason": "NO_POSE_DETECTED_EITHER_PASS"})
                    continue

                features = extract_node_features(result)
                pair_id = filename
                entry = save_keypoint_entry(features, class_label, filename, is_flipped=False, pair_id=pair_id, rescued=True)
                dataset_index.append(entry)
                print(f"[retry {idx}/{len(retry_queue)}] {class_label}/{filename}: RESCUED & extracted | "
                      f"{entry['num_key_landmarks_detected']}/{len(KEY_LANDMARKS)} confident | avg vis {entry['avg_visibility']:.2f}")

                if ENABLE_FLIP_AUGMENTATION:
                    flipped_features = flip_keypoints(features)
                    flip_entry = save_keypoint_entry(flipped_features, class_label, filename, is_flipped=True, pair_id=pair_id, rescued=True)
                    dataset_index.append(flip_entry)
    elif retry_queue:
        for class_label, filename, filepath, image_bgr in retry_queue:
            excluded.append({"filename": filename, "class": class_label, "reason": "NO_POSE_DETECTED_RETRY_DISABLED"})

    # Write outputs
    save_skeleton_reference()

    with open(DATASET_INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "class", "keypoint_path", "pair_id", "is_flipped",
            "num_key_landmarks_detected", "avg_visibility", "rescued_by_retry"
        ])
        writer.writeheader()
        writer.writerows(dataset_index)

    with open(EXCLUDED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    total_in = len(all_images)
    total_extracted = sum(1 for r in dataset_index if not r["is_flipped"])
    total_excluded = len(excluded)
    rescued_count = sum(1 for r in dataset_index if r["rescued_by_retry"] and not r["is_flipped"])
    total_dataset_entries = len(dataset_index)

    print("\n" + "=" * 60)
    print("GNN KEYPOINT EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total images found:            {total_in}")
    print(f"Successfully extracted:        {total_extracted} ({total_extracted/total_in*100:.1f}%) unique source images")
    print(f"  of which rescued by PASS 2 retry: {rescued_count}")
    print(f"Excluded (no pose found):      {total_excluded} ({total_excluded/total_in*100:.1f}%)")
    print(f"Flip augmentation: {'ENABLED — total dataset entries incl. flips: ' + str(total_dataset_entries) if ENABLE_FLIP_AUGMENTATION else 'DISABLED — one entry per source image'}")
    for c in CLASS_FOLDERS:
        c_extracted = sum(1 for r in dataset_index if r["class"] == c and not r["is_flipped"])
        c_excluded = sum(1 for r in excluded if r["class"] == c)
        print(f"  {c}: extracted={c_extracted}, excluded={c_excluded}")
    print(f"\nKeypoint arrays saved to:  {KEYPOINTS_OUTPUT_FOLDER}/<class>/<filename>.npy  (shape 17x3: x,y,visibility)")
    print(f"Dataset index saved to:    {DATASET_INDEX_CSV}")
    print(f"Excluded images logged to: {EXCLUDED_CSV}")
    print(f"Shared skeleton adjacency: {EDGES_FILE}, {NODE_ORDER_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    process_folder()
