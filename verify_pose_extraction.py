"""
Pose Extraction Verification Script
----------------------------------------------
Features:
  - Two-class folder walk (safe/ and unsafe/)
  - Custom confidence-graded connection-line drawing (fixes MediaPipe's
    internal, unconfigurable dot/line threshold mismatch)
  - PASS 1 (normal) + PASS 2 (lenient retry) for zero-detection images
  - 17-point keypoint set: 12 body joints + 5 head joints (nose, eyes, ears)
  - BEST-SIDE quality flagging (judges by whichever side is better captured,
    NOT total joints across both sides — a clean side-profile shot is
    expected geometry, not a defect)
  - Face-detection diagnostic (informational only — NOT used to auto-correct
    head landmarks, since it has known false negatives on extreme profile
    shots that carry the strongest head-orientation signal)

Usage:
    pip install mediapipe==0.10.14 opencv-python pandas
    python verify_pose_extraction.py
"""

import os
import csv
import shutil
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_BASE_FOLDER = "image_data"
CLASS_FOLDERS = ["safe", "unsafe"]
OUTPUT_BASE_FOLDER = "pose_verification_output"
SUMMARY_CSV = "pose_verification_summary.csv"

QUALITY_FLAG_THRESHOLD = 5   # a side is considered "well-captured" if this many of its 6 joints are detected
FLAGGED_REVIEW_FOLDER = "flagged_for_review"
FLAGGED_CSV = "flagged_low_quality.csv"

VISIBILITY_THRESHOLD = 0.5
BORDERLINE_THRESHOLD = 0.3

KEY_LANDMARKS = {
    0: "nose",
    2: "left_eye", 5: "right_eye",
    7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}
HEAD_LANDMARK_NAMES = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
BODY_LANDMARK_NAMES = set(KEY_LANDMARKS.values()) - HEAD_LANDMARK_NAMES
LEFT_BODY_NAMES = {n for n in BODY_LANDMARK_NAMES if n.startswith("left")}
RIGHT_BODY_NAMES = {n for n in BODY_LANDMARK_NAMES if n.startswith("right")}

USE_CONTRAST_ENHANCEMENT = True
ENABLE_RETRY_PASS = True
RETRY_MIN_DETECTION_CONFIDENCE = 0.15
RETRY_FORCE_CONTRAST_ENHANCEMENT = True
FACE_DETECTION_MIN_CONFIDENCE = 0.5

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")


def enhance_contrast(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)


def draw_confidence_graded_connections(image, landmarks, w, h):
    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
        v1, v2 = landmarks[start_idx].visibility, landmarks[end_idx].visibility
        if v1 < BORDERLINE_THRESHOLD or v2 < BORDERLINE_THRESHOLD:
            continue
        x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
        x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
        weaker = min(v1, v2)
        if weaker >= VISIBILITY_THRESHOLD:
            color, thickness = (255, 0, 255), 3
        else:
            color, thickness = (200, 160, 255), 2
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def run_detection(image_bgr, pose_estimator, use_contrast):
    detection_input = enhance_contrast(image_bgr) if use_contrast else image_bgr
    image_rgb = cv2.cvtColor(detection_input, cv2.COLOR_BGR2RGB)
    return pose_estimator.process(image_rgb)


def check_face_and_annotate(annotated, image_bgr, face_detector):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = face_detector.process(image_rgb)
    if result.detections:
        best = max(result.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 220, 100), 2)
        return True, float(best.score[0])
    else:
        cv2.putText(annotated, "NO FACE DETECTED - head landmarks unreliable", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return False, 0.0


def annotate_and_evaluate(image_bgr, result, class_label, filename, face_detector, was_rescued=False):
    annotated = image_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated, result.pose_landmarks, connections=None,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    draw_confidence_graded_connections(annotated, result.pose_landmarks.landmark, *annotated.shape[1::-1])
    face_detected, face_confidence = check_face_and_annotate(annotated, image_bgr, face_detector)

    landmarks = result.pose_landmarks.landmark
    detected_count = body_detected_count = head_detected_count = 0
    left_detected_count = right_detected_count = 0
    borderline_count = 0
    visibilities = []
    missing = []
    missing_left = missing_right = 0

    h, w = annotated.shape[:2]
    for lm_idx, lm_name in KEY_LANDMARKS.items():
        lm = landmarks[lm_idx]
        visibilities.append(lm.visibility)
        cx, cy = int(lm.x * w), int(lm.y * h)
        is_body = lm_name in BODY_LANDMARK_NAMES

        if lm.visibility >= VISIBILITY_THRESHOLD:
            detected_count += 1
            if is_body:
                body_detected_count += 1
                if lm_name in LEFT_BODY_NAMES:
                    left_detected_count += 1
                elif lm_name in RIGHT_BODY_NAMES:
                    right_detected_count += 1
            else:
                head_detected_count += 1
            cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
        elif lm.visibility >= BORDERLINE_THRESHOLD:
            borderline_count += 1
            missing.append(lm_name)
            cv2.circle(annotated, (cx, cy), 6, (0, 165, 255), 2)
            if is_body:
                if lm_name.startswith("left"):
                    missing_left += 1
                elif lm_name.startswith("right"):
                    missing_right += 1
        else:
            missing.append(lm_name)
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255),
                            markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
            if is_body:
                if lm_name.startswith("left"):
                    missing_left += 1
                elif lm_name.startswith("right"):
                    missing_right += 1

    avg_visibility = sum(visibilities) / len(visibilities)
    tag = " [RESCUED PASS 2]" if was_rescued else ""
    summary_text = (f"[{class_label}] body {body_detected_count}/{len(BODY_LANDMARK_NAMES)} | "
                     f"head {head_detected_count}/{len(HEAD_LANDMARK_NAMES)} | avg conf {avg_visibility:.2f}{tag}")
    cv2.putText(annotated, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated, {
        "filename": filename, "class": class_label, "pose_detected": True, "rescued_by_retry": was_rescued,
        "num_key_landmarks_detected": detected_count, "num_key_landmarks_total": len(KEY_LANDMARKS),
        "num_body_landmarks_detected": body_detected_count, "num_body_landmarks_total": len(BODY_LANDMARK_NAMES),
        "num_head_landmarks_detected": head_detected_count, "num_head_landmarks_total": len(HEAD_LANDMARK_NAMES),
        "left_side_detected": left_detected_count, "right_side_detected": right_detected_count,
        "best_side_detected": max(left_detected_count, right_detected_count),
        "face_detected": face_detected, "face_confidence": round(face_confidence, 3),
        "num_borderline": borderline_count, "avg_visibility": round(avg_visibility, 3),
        "missing_joints": ";".join(missing) if missing else "none",
        "missing_left_count": missing_left, "missing_right_count": missing_right,
    }


def empty_result(filename, class_label, reason):
    return {
        "filename": filename, "class": class_label, "pose_detected": False, "rescued_by_retry": False,
        "num_key_landmarks_detected": 0, "num_key_landmarks_total": len(KEY_LANDMARKS),
        "num_body_landmarks_detected": 0, "num_body_landmarks_total": len(BODY_LANDMARK_NAMES),
        "num_head_landmarks_detected": 0, "num_head_landmarks_total": len(HEAD_LANDMARK_NAMES),
        "left_side_detected": 0, "right_side_detected": 0, "best_side_detected": 0,
        "face_detected": False, "face_confidence": 0.0,
        "num_borderline": 0, "avg_visibility": 0.0, "missing_joints": reason,
        "missing_left_count": 0, "missing_right_count": 0,
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
        os.makedirs(os.path.join(OUTPUT_BASE_FOLDER, class_label), exist_ok=True)

    if not all_images:
        print(f"No images found under '{INPUT_BASE_FOLDER}'. Check the path.")
        return

    print(f"Found {len(all_images)} images across {CLASS_FOLDERS}. Starting PASS 1...\n")
    results_log = []
    retry_queue = []

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                       min_detection_confidence=0.5) as pose, \
         mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_DETECTION_MIN_CONFIDENCE) as face_detector:

        for idx, (class_label, filename, filepath) in enumerate(all_images, 1):
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                print(f"[{idx}/{len(all_images)}] {class_label}/{filename}: COULD NOT READ FILE")
                results_log.append(empty_result(filename, class_label, "FILE_UNREADABLE"))
                continue

            result = run_detection(image_bgr, pose, USE_CONTRAST_ENHANCEMENT)
            if result.pose_landmarks is None:
                retry_queue.append((class_label, filename, filepath, image_bgr))
                continue

            annotated, result_dict = annotate_and_evaluate(image_bgr, result, class_label, filename, face_detector)
            cv2.imwrite(os.path.join(OUTPUT_BASE_FOLDER, class_label, filename), annotated)
            results_log.append(result_dict)
            status = "OK" if result_dict["num_key_landmarks_detected"] == len(KEY_LANDMARKS) else \
                     f"PARTIAL ({len(KEY_LANDMARKS) - result_dict['num_key_landmarks_detected']} missing)"
            print(f"[{idx}/{len(all_images)}] {class_label}/{filename}: {status} | avg conf {result_dict['avg_visibility']:.2f}")

    rescued_count = 0
    still_failed = []
    if retry_queue and ENABLE_RETRY_PASS:
        print(f"\n{'=' * 60}\nPASS 2 (retry): {len(retry_queue)} images had no pose detected in PASS 1.\n{'=' * 60}\n")
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                           min_detection_confidence=RETRY_MIN_DETECTION_CONFIDENCE) as retry_pose, \
             mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_DETECTION_MIN_CONFIDENCE) as face_detector:

            for idx, (class_label, filename, filepath, image_bgr) in enumerate(retry_queue, 1):
                result = run_detection(image_bgr, retry_pose, RETRY_FORCE_CONTRAST_ENHANCEMENT)
                if result.pose_landmarks is None:
                    print(f"[retry {idx}/{len(retry_queue)}] {class_label}/{filename}: STILL NO POSE DETECTED")
                    results_log.append(empty_result(filename, class_label, "ALL"))
                    still_failed.append((class_label, filename))
                    cv2.imwrite(os.path.join(OUTPUT_BASE_FOLDER, class_label, filename), image_bgr)
                    continue

                annotated, result_dict = annotate_and_evaluate(image_bgr, result, class_label, filename, face_detector, was_rescued=True)
                cv2.imwrite(os.path.join(OUTPUT_BASE_FOLDER, class_label, filename), annotated)
                results_log.append(result_dict)
                rescued_count += 1
                print(f"[retry {idx}/{len(retry_queue)}] {class_label}/{filename}: RESCUED | "
                      f"{result_dict['num_key_landmarks_detected']}/{len(KEY_LANDMARKS)} joints | avg conf {result_dict['avg_visibility']:.2f}")
    elif retry_queue:
        for class_label, filename, filepath, image_bgr in retry_queue:
            results_log.append(empty_result(filename, class_label, "ALL"))
            cv2.imwrite(os.path.join(OUTPUT_BASE_FOLDER, class_label, filename), image_bgr)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "class", "pose_detected", "rescued_by_retry",
            "num_key_landmarks_detected", "num_key_landmarks_total",
            "num_body_landmarks_detected", "num_body_landmarks_total",
            "num_head_landmarks_detected", "num_head_landmarks_total",
            "left_side_detected", "right_side_detected", "best_side_detected",
            "face_detected", "face_confidence",
            "num_borderline", "avg_visibility", "missing_joints",
            "missing_left_count", "missing_right_count"
        ])
        writer.writeheader()
        writer.writerows(results_log)

    total = len(results_log)
    no_detection = sum(1 for r in results_log if not r["pose_detected"])
    full_body_detection = sum(1 for r in results_log if r["pose_detected"] and r["num_body_landmarks_detected"] == r["num_body_landmarks_total"])
    partial_detection = total - no_detection - full_body_detection
    avg_conf_all = sum(r["avg_visibility"] for r in results_log if r["pose_detected"]) / max(1, (total - no_detection))
    avg_head_detected = sum(r["num_head_landmarks_detected"] for r in results_log if r["pose_detected"]) / max(1, (total - no_detection))

    print("\n" + "=" * 60)
    print("POSE EXTRACTION VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total images processed:        {total}")
    if ENABLE_RETRY_PASS:
        print(f"PASS 2 retry rescued:           {rescued_count} images (out of {len(retry_queue)} that failed PASS 1)")
    print(f"No pose detected (final):       {no_detection}  ({no_detection/total*100:.1f}%)")
    print(f"All {len(BODY_LANDMARK_NAMES)} BODY joints detected: {full_body_detection}  ({full_body_detection/total*100:.1f}%)")
    print(f"Partial body detection:         {partial_detection}  ({partial_detection/total*100:.1f}%)")
    print(f"Avg HEAD joints detected (out of {len(HEAD_LANDMARK_NAMES)}): {avg_head_detected:.2f} "
          f"(NOTE: visibility alone may not vary much with orientation — check coordinates, not just this count)")
    face_detected_count = sum(1 for r in results_log if r.get("face_detected"))
    face_checked_count = sum(1 for r in results_log if r["pose_detected"])
    if face_checked_count:
        print(f"Face actually detected (dedicated face model): {face_detected_count}/{face_checked_count} "
              f"({face_detected_count/face_checked_count*100:.1f}%)")
    print(f"Average visibility confidence:  {avg_conf_all:.3f}")

    print("\nBreakdown by class:")
    for class_label in CLASS_FOLDERS:
        class_rows = [r for r in results_log if r["class"] == class_label]
        if not class_rows:
            continue
        c_total = len(class_rows)
        c_no_det = sum(1 for r in class_rows if not r["pose_detected"])
        c_full = sum(1 for r in class_rows if r["pose_detected"] and r["num_body_landmarks_detected"] == r["num_body_landmarks_total"])
        c_rescued = sum(1 for r in class_rows if r["rescued_by_retry"])
        c_avg_conf = sum(r["avg_visibility"] for r in class_rows if r["pose_detected"]) / max(1, (c_total - c_no_det))
        print(f"  {class_label}: {c_total} images | no detection: {c_no_det} ({c_no_det/c_total*100:.1f}%) "
              f"| full {len(BODY_LANDMARK_NAMES)}/{len(BODY_LANDMARK_NAMES)}: {c_full} ({c_full/c_total*100:.1f}%) "
              f"| rescued: {c_rescued} | avg conf: {c_avg_conf:.3f}")

    print(f"\nAnnotated images saved to:  {OUTPUT_BASE_FOLDER}/<class>/")
    print(f"Per-image details saved to: {SUMMARY_CSV}")

    total_missing_left = sum(r["missing_left_count"] for r in results_log)
    total_missing_right = sum(r["missing_right_count"] for r in results_log)
    if total_missing_left + total_missing_right > 0:
        print(f"\nLeft-side joints missing (total across dataset):  {total_missing_left}")
        print(f"Right-side joints missing (total across dataset): {total_missing_right}")
        if total_missing_right > total_missing_left * 1.3:
            print("-> Right side fails noticeably more than left (aggregate, multi-angle dataset).")
        elif total_missing_left > total_missing_right * 1.3:
            print("-> Left side fails noticeably more than right (aggregate, multi-angle dataset).")
    print("=" * 60)

    # BEST-SIDE quality flagging — an image is only flagged if even its
    # BETTER-captured side is unreliable. A clean side-profile shot (one
    # side fully occluded, other side fully confident) is NOT flagged.
    flagged = [r for r in results_log if r["best_side_detected"] < QUALITY_FLAG_THRESHOLD]

    if flagged:
        for class_label in CLASS_FOLDERS:
            os.makedirs(os.path.join(FLAGGED_REVIEW_FOLDER, class_label), exist_ok=True)
        for r in flagged:
            src = os.path.join(INPUT_BASE_FOLDER, r["class"], r["filename"])
            dst = os.path.join(FLAGGED_REVIEW_FOLDER, r["class"], r["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)

        with open(FLAGGED_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "class", "pose_detected", "rescued_by_retry",
                "num_key_landmarks_detected", "num_key_landmarks_total",
                "num_body_landmarks_detected", "num_body_landmarks_total",
                "num_head_landmarks_detected", "num_head_landmarks_total",
                "left_side_detected", "right_side_detected", "best_side_detected",
                "face_detected", "face_confidence",
                "num_borderline", "avg_visibility", "missing_joints",
                "missing_left_count", "missing_right_count"
            ])
            writer.writeheader()
            writer.writerows(flagged)

        flagged_safe = sum(1 for r in flagged if r["class"] == "safe")
        flagged_unsafe = sum(1 for r in flagged if r["class"] == "unsafe")
        print(f"\n{len(flagged)} images flagged as low-quality (< {QUALITY_FLAG_THRESHOLD}/6 on BOTH sides — best-side check):")
        print(f"  safe: {flagged_safe}  |  unsafe: {flagged_unsafe}")
        print(f"Copied to:   {FLAGGED_REVIEW_FOLDER}/<class>/  (originals untouched)")
        print(f"Listed in:   {FLAGGED_CSV}")
    else:
        print(f"\nNo images fell below the {QUALITY_FLAG_THRESHOLD}/6 best-side quality threshold — nothing flagged.")


if __name__ == "__main__":
    process_folder()
