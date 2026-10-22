"""
RF-DETR Evaluation — Pork Rasher Packaging Defect Detection.

This script evaluates a Roboflow-hosted object-detection workflow against
the labelled test split of the Pork Rasher Error-Packaging dataset.

The script performs the following operations:
    1. Loads the dataset configuration and class names from data.yaml.
    2. Connects to the configured Roboflow workspace and workflow.
    3. Runs inference on every supported image in the test image directory.
    4. Extracts predictions and converts bounding boxes to normalised
       corner coordinates.
    5. Saves prediction records and inference failures.
    6. Loads ground-truth bounding boxes from YOLO-format label files.
    7. Matches predictions against ground truth using intersection over union.
    8. Calculates per-class precision, recall, F1-score, AP and mAP metrics.
    9. Calculates aggregate metrics and inference-time statistics.
   10. Saves the evaluation metrics to a JSON file and prints a summary.

Input files:
    - Dataset data.yaml configuration.
    - Images from the test/images directory.
    - YOLO-format labels from the test/labels directory.

Output files:
    - results/rfdetr_v1_test/predictions.csv
    - results/rfdetr_v1_test/failures.json
    - results/rfdetr_v1_test/metrics.json

Important evaluation assumptions:
    - Prediction coordinates are assumed to use the original image's pixel
      dimensions before conversion to normalised coordinates.
    - Ground-truth labels use YOLO's normalised centre-based format.
    - Precision, recall and F1 use the configured confidence and IoU thresholds.
    - AP is calculated from confidence-ranked predictions at each IoU threshold.
    - Mean inference request time includes network communication.

The coordinate convention and evaluation methodology should be verified
before treating the resulting metrics as a definitive benchmark.
"""

import os
import csv
import time
import json
import statistics
from pathlib import Path

import yaml
from PIL import Image
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ---------- CONFIGURATION ----------
# Specify the dataset location and the directories containing test images,
# ground-truth labels and dataset class configuration.

DATASET = Path(
    "/Users/sameersimon/Downloads/"
    "Pork Rasher Error-Packaging-.v1i.yolov8"
)
IMAGE_DIR = DATASET / "test" / "images"
LABEL_DIR = DATASET / "test" / "labels"
DATA_YAML = DATASET / "data.yaml"

# Store all evaluation outputs in a dedicated results directory.
# Create the directory automatically if it does not already exist.
OUT_DIR = Path("results/rfdetr_v1_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Roboflow workspace and hosted inference workflow identifiers.
WORKSPACE = "ece4078h3"
WORKFLOW_ID = (
    "pork-rasher-errorpackaging-vpork-rasher-"
    "error-packaging-rj5da-1-rfdetr-small-t1-logic"
)

# Minimum confidence used when calculating precision, recall and F1.
# The AP calculations below use all returned predictions regardless of
# this threshold so that detections can be ranked by confidence.
THRESHOLD = 0.50

# Minimum intersection-over-union value required to match a prediction
# with a ground-truth object when calculating threshold-based metrics.
IOU_MATCH = 0.50

# Roboflow API configuration.
API_KEY = "siB1OGhXN0kGj0mPwzwM"

# Verify that the dataset configuration exists before attempting evaluation.
if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

# Load the dataset configuration, including the class names and dataset
# metadata, from the YAML file.
with open(DATA_YAML, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Read the class names. The names field can be either a dictionary indexed
# by class ID or a list ordered by class ID.
names = cfg["names"]
if isinstance(names, dict):
    class_names = [names[i] for i in sorted(names)]
else:
    class_names = list(names)

# Create a mapping from each class name to its numerical class ID.
# This allows predicted class names to be compared with YOLO label IDs.
class_to_id = {
    str(name): i for i, name in enumerate(class_names)
}

# Initialise the Roboflow inference client and configure API-key transport
# through the HTTP header.
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
).configure(
    InferenceConfiguration(api_key_transport="header")
)

# Collect supported test images in sorted order to make processing
# deterministic and easier to reproduce.
image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# The recorded dataset contains 76 test images. Print a warning rather
# than stopping if the actual number differs.
if len(image_paths) != 76:
    print(f"WARNING: expected 76 images, found {len(image_paths)}")

# Predictions retain every returned confidence for AP calculations.
# The following collections store per-image predictions, request timings,
# failed requests and CSV rows for successful predictions.
predictions_by_image = {}
timings = []
failures = []
rows = []

# ---------------------------------------------------------------------------
# 1. RUN INFERENCE ON THE TEST IMAGES
# ---------------------------------------------------------------------------
# Process every test image using the hosted workflow. Convert successful
# predictions to a common representation for subsequent metric calculations.

for i, path in enumerate(image_paths, 1):
    print(f"[{i}/{len(image_paths)}] {path.name}")

    try:
        # Start timing immediately before the hosted inference request.
        # The elapsed time includes communication with the remote service.
        start = time.perf_counter()
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": str(path)},
            use_cache=False,
        )
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        # Expected response structure from the existing main.py.
        # Extract the list of object-detection predictions returned by
        # the configured workflow.
        predictions = result[0]["predictions"]["predictions"]

        # Read the original image dimensions so that predicted pixel
        # coordinates can be normalised relative to its width and height.
        with Image.open(path) as im:
            img_w, img_h = im.size

        # Build a cleaned list containing only predictions with recognised
        # classes and valid bounding-box geometry.
        cleaned = []
        for pred in predictions:
            label = str(pred["class"])

            # Ignore predicted class names that do not appear in data.yaml.
            # This prevents predictions from being assigned an incorrect
            # class ID during ground-truth matching.
            if label not in class_to_id:
                print(f"  WARNING: unknown class name: {label}")
                continue

            # Convert the confidence score and centre-based bounding-box
            # values to floating-point numbers.
            conf = float(pred["confidence"])
            x = float(pred["x"])
            y = float(pred["y"])
            w = float(pred["width"])
            h = float(pred["height"])

            # Assumes workflow boxes use original-image pixel coordinates.
            # Verify this assumption before accepting final metrics.
            # Convert centre-based pixel coordinates to normalised corner
            # coordinates in the range expected by the YOLO ground truth.
            x1 = max(0.0, (x - w / 2) / img_w)
            y1 = max(0.0, (y - h / 2) / img_h)
            x2 = min(1.0, (x + w / 2) / img_w)
            y2 = min(1.0, (y + h / 2) / img_h)

            # Discard boxes that have no positive area after conversion.
            if x2 <= x1 or y2 <= y1:
                continue

            # Store the class ID, label, confidence and normalised box
            # for use in evaluation and output reporting.
            cleaned.append({
                "class_id": class_to_id[label],
                "class": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
            })

        # Associate the cleaned predictions with the current image filename.
        predictions_by_image[path.name] = cleaned

        # Add one CSV row for every valid prediction returned by inference.
        for p in cleaned:
            rows.append({
                "image": path.name,
                "class": p["class"],
                "confidence": p["confidence"],
                "x1_norm": p["box"][0],
                "y1_norm": p["box"][1],
                "x2_norm": p["box"][2],
                "y2_norm": p["box"][3],
                "request_time_seconds": elapsed,
                "status": "success",
            })

        print(f"  {len(cleaned)} predictions; {elapsed:.2f}s")

    except Exception as exc:
        # Record inference failures separately so that unsuccessful requests
        # are not silently interpreted as images with no predicted defects.
        failures.append({"image": path.name, "error": str(exc)})

        # Use an empty prediction list for this image. It remains represented
        # in the evaluation data, while the failure is recorded separately.
        predictions_by_image[path.name] = []
        print(f"  FAILED: {exc}")

# ---------------------------------------------------------------------------
# 2. SAVE RAW INFERENCE RESULTS
# ---------------------------------------------------------------------------
# Save successful prediction records to CSV and failed requests to JSON.
# These files support later inspection and troubleshooting.

with open(OUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
    # Define the columns written for each prediction.
    fields = [
        "image", "class", "confidence",
        "x1_norm", "y1_norm", "x2_norm", "y2_norm",
        "request_time_seconds", "status"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# Save the filenames and error messages for images that failed inference.
with open(OUT_DIR / "failures.json", "w", encoding="utf-8") as f:
    json.dump(failures, f, indent=2)

# ---------------------------------------------------------------------------
# 3. LOAD GROUND-TRUTH LABELS
# ---------------------------------------------------------------------------
# YOLO label files contain one object per line in the format:
# class_id x_center y_center width height
#
# The coordinates and dimensions are normalised relative to the image size.
# Convert them to corner coordinates to match the representation used for
# the predictions.

ground_truth = {}
for path in image_paths:
    label_file = LABEL_DIR / f"{path.stem}.txt"
    gt = []

    # Images without a corresponding label file are assigned an empty
    # ground-truth list by the existing logic.
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()

            # Skip malformed lines that do not contain exactly five values.
            if len(parts) != 5:
                continue

            # Parse the class ID and the normalised centre-based box values.
            cid, x, y, w, h = parts
            x, y, w, h = map(float, (x, y, w, h))

            # Convert YOLO centre-based coordinates to normalised corner
            # coordinates [x1, y1, x2, y2].
            gt.append({
                "class_id": int(cid),
                "box": [
                    x - w / 2, y - h / 2,
                    x + w / 2, y + h / 2
                ]
            })

    # Store the ground-truth objects under the matching image filename.
    ground_truth[path.name] = gt


# ---------------------------------------------------------------------------
# 4. INTERSECTION OVER UNION (IoU)
# ---------------------------------------------------------------------------
# IoU measures the overlap between two bounding boxes. It is calculated as
# the intersection area divided by the union area.

def iou(a, b):
    """Calculate intersection over union for two corner-based boxes.

    Both boxes must use the format [x1, y1, x2, y2] in the same coordinate
    system.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        float: Intersection-over-union score between 0 and 1. Returns 0.0
        if the union area is zero.
    """
    # Determine the intersection rectangle.
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # Calculate intersection area, ensuring non-overlapping dimensions
    # contribute zero rather than a negative area.
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of each bounding box.
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    # The union is the combined area after removing the overlap.
    union = area_a + area_b - inter

    # Avoid division by zero for boxes with no union area.
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 5. CONFIDENCE-THRESHOLDED PRECISION, RECALL AND F1
# ---------------------------------------------------------------------------
# Match predictions to ground-truth objects for one class. Predictions are
# processed in descending confidence order, and each ground-truth object
# can match at most one prediction.

def match_at_iou(cid, threshold):
    """Calculate threshold-based detection metrics for one class.

    Predictions are filtered using the global confidence threshold and
    sorted from highest to lowest confidence. Each prediction is matched
    greedily to the highest-IoU unmatched ground-truth object in the same
    image, provided that the IoU reaches the supplied threshold.

    Args:
        cid: Numerical class ID to evaluate.
        threshold: Minimum IoU required for a match.

    Returns:
        A tuple containing precision, recall, F1-score, ground-truth count,
        true-positive count, false-positive count and false-negative count.

        Recall is None when no ground-truth instances of this class exist.
    """
    # Build a per-image ground-truth dictionary containing only this class.
    gt_by_image = {
        name: [g for g in gt if g["class_id"] == cid]
        for name, gt in ground_truth.items()
    }

    # Count all ground-truth objects for the current class.
    total_gt = sum(len(v) for v in gt_by_image.values())
    detections = []

    # Collect predictions belonging to this class that pass the global
    # confidence threshold used for precision, recall and F1.
    for name, preds in predictions_by_image.items():
        for p in preds:
            if p["class_id"] == cid and p["confidence"] >= THRESHOLD:
                detections.append((p["confidence"], name, p["box"]))

    # Process higher-confidence predictions first.
    detections.sort(key=lambda d: d[0], reverse=True)

    # Track matched ground-truth indices independently for each image.
    matched = {name: set() for name in gt_by_image}
    tp, fp = [], []

    for conf, name, box in detections:
        candidates = gt_by_image[name]
        best_idx, best_iou = None, 0.0

        # Find the highest-IoU ground-truth box that has not already been
        # assigned to another prediction.
        for j, gt in enumerate(candidates):
            score = iou(box, gt["box"])
            if j not in matched[name] and score > best_iou:
                best_idx, best_iou = j, score

        # Count a true positive when an unmatched ground-truth box meets
        # the required IoU. Otherwise, the prediction is a false positive.
        if best_idx is not None and best_iou >= threshold:
            matched[name].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # Aggregate true positives, false positives and false negatives.
    tp_total, fp_total = sum(tp), sum(fp)
    fn_total = total_gt - tp_total

    # Calculate precision and recall, handling cases with zero denominators.
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0
    recall = tp_total / total_gt if total_gt else None

    # Calculate F1 only when recall is defined and precision plus recall
    # is non-zero.
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall else 0
    )

    return precision, recall, f1, total_gt, tp_total, fp_total, fn_total


# ---------------------------------------------------------------------------
# 6. AVERAGE PRECISION (AP)
# ---------------------------------------------------------------------------
# Calculate AP for one class at a specified IoU threshold. Unlike the
# threshold-based precision/recall calculation, AP uses all predictions
# for the class and evaluates them in descending confidence order.

def average_precision(cid, iou_threshold):
    """Calculate interpolated average precision for one class.

    Predictions are sorted by confidence and matched to unmatched ground-truth
    objects using the specified IoU threshold. Cumulative true-positive and
    false-positive counts are converted to precision and recall values.
    A monotonically non-increasing precision envelope is then integrated
    over changes in recall.

    Args:
        cid: Numerical class ID to evaluate.
        iou_threshold: Minimum IoU required to count a prediction as a match.

    Returns:
        float: Average precision between 0 and 1.
        None: If the class has no ground-truth instances in the evaluation set.
    """
    # Select the ground-truth objects belonging to the requested class.
    gt_by_image = {
        name: [g for g in gt if g["class_id"] == cid]
        for name, gt in ground_truth.items()
    }

    total_gt = sum(len(v) for v in gt_by_image.values())

    # AP is undefined for a class with no ground-truth instances.
    if total_gt == 0:
        return None

    detections = []

    # Collect all predictions for the class without applying the global
    # confidence threshold so that the full confidence ranking is retained.
    for name, preds in predictions_by_image.items():
        for p in preds:
            if p["class_id"] == cid:
                detections.append((p["confidence"], name, p["box"]))

    # Sort predictions by confidence from highest to lowest.
    detections.sort(key=lambda d: d[0], reverse=True)

    # Track which ground-truth objects have already been matched.
    matched = {name: set() for name in gt_by_image}
    tp, fp = [], []

    for conf, name, box in detections:
        candidates = gt_by_image[name]
        best_idx, best_iou = None, 0.0

        # Find the best unmatched ground-truth box for this prediction.
        for j, gt in enumerate(candidates):
            score = iou(box, gt["box"])
            if j not in matched[name] and score > best_iou:
                best_idx, best_iou = j, score

        # Classify the prediction using the requested IoU threshold.
        if best_idx is not None and best_iou >= iou_threshold:
            matched[name].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # If no predictions exist for the class, AP is zero when ground truth
    # is present because no ground-truth objects have been detected.
    if not tp:
        return 0.0

    import numpy as np

    # Calculate cumulative true-positive and false-positive counts.
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # Derive precision and recall at each confidence-ranked prediction.
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # Add boundary points to define the full precision-recall curve.
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Construct the interpolated precision envelope by propagating the
    # highest subsequent precision value backwards through the curve.
    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])

    # Identify recall changes and integrate the precision envelope over
    # each recall interval to obtain average precision.
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum(
        (recall[changes + 1] - recall[changes]) * precision[changes + 1]
    ))


# ---------------------------------------------------------------------------
# 7. PER-CLASS EVALUATION
# ---------------------------------------------------------------------------
# Calculate precision, recall and F1 at the configured confidence/IoU
# thresholds. Calculate AP at IoU 0.50 and mAP across IoU thresholds
# from 0.50 through 0.95 in increments of 0.05.

# Overall metrics at the chosen confidence and IoU threshold.
per_class = []
for cid, name in enumerate(class_names):
    p, r, f1, n_gt, tp, fp, fn = match_at_iou(cid, IOU_MATCH)
    ap50 = average_precision(cid, 0.50)

    # Evaluate AP at IoU thresholds 0.50, 0.55, ..., 0.95.
    aps = [
        average_precision(cid, t)
        for t in [x / 100 for x in range(50, 100, 5)]
    ]

    # Exclude undefined AP values for classes with no ground-truth objects
    # when averaging across IoU thresholds.
    aps_valid = [x for x in aps if x is not None]
    ap5095 = sum(aps_valid) / len(aps_valid) if aps_valid else None

    # Store the evaluation metrics and confusion counts for this class.
    per_class.append({
        "class": name,
        "ground_truth_instances": n_gt,
        "precision_at_threshold": p,
        "recall_at_threshold": r,
        "f1_at_threshold": f1,
        "mAP50": ap50,
        "mAP50_95": ap5095,
        "TP": tp, "FP": fp, "FN": fn,
    })


# ---------------------------------------------------------------------------
# 8. AGGREGATE METRICS
# ---------------------------------------------------------------------------
# Sum the per-class true-positive, false-positive and false-negative counts
# to calculate micro-averaged precision, recall and F1 across classes.

# Micro precision/recall/F1 across classes.
tot_tp = sum(x["TP"] for x in per_class)
tot_fp = sum(x["FP"] for x in per_class)
tot_fn = sum(x["FN"] for x in per_class)

P = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0
R = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0
F1 = 2 * P * R / (P + R) if P + R else 0

# Build lists of defined per-class AP values. Classes with no ground-truth
# instances are excluded from these macro averages.
valid_ap50 = [x["mAP50"] for x in per_class if x["mAP50"] is not None]
valid_ap5095 = [x["mAP50_95"] for x in per_class if x["mAP50_95"] is not None]

# Assemble the final summary, including test coverage, aggregate metrics,
# request timings and the detailed per-class results.
summary = {
    "images_total": len(image_paths),
    "images_successful": len(image_paths) - len(failures),
    "images_failed": len(failures),
    "failure_rate": len(failures) / len(image_paths) if image_paths else None,
    "confidence_threshold_for_P_R_F1": THRESHOLD,
    "IoU_threshold_for_P_R_F1": IOU_MATCH,
    "precision_micro": P,
    "recall_micro": R,
    "F1_micro": F1,
    "mAP50_macro_over_classes_with_ground_truth": (
        sum(valid_ap50) / len(valid_ap50) if valid_ap50 else None
    ),
    "mAP50_95_macro_over_classes_with_ground_truth": (
        sum(valid_ap5095) / len(valid_ap5095) if valid_ap5095 else None
    ),
    "mean_request_seconds": statistics.mean(timings) if timings else None,
    "median_request_seconds": statistics.median(timings) if timings else None,
    "timing_includes_network": True,
    "per_class": per_class,
}

# Save the complete evaluation summary to JSON so that results can be
# inspected later or used by other reporting tools.
with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Print the summary to the terminal for immediate inspection and report
# the absolute path where the evaluation outputs were saved.
print("\n--- RF-DETR evaluation summary ---")
print(json.dumps(summary, indent=2))
print(f"\nSaved results to: {OUT_DIR.resolve()}")