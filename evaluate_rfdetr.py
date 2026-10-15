
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
DATASET = Path(
    "/Users/sameersimon/Downloads/"
    "Pork Rasher Error-Packaging-.v1i.yolov8"
)
IMAGE_DIR = DATASET / "test" / "images"
LABEL_DIR = DATASET / "test" / "labels"
DATA_YAML = DATASET / "data.yaml"

OUT_DIR = Path("results/rfdetr_v1_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE = "ece4078h3"
WORKFLOW_ID = (
    "pork-rasher-errorpackaging-vpork-rasher-"
    "error-packaging-rj5da-1-rfdetr-small-t1-logic"
)
THRESHOLD = 0.50
IOU_MATCH = 0.50

# Roboflow API configuration
API_KEY = "siB1OGhXN0kGj0mPwzwM"

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

with open(DATA_YAML, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

names = cfg["names"]
if isinstance(names, dict):
    class_names = [names[i] for i in sorted(names)]
else:
    class_names = list(names)

class_to_id = {
    str(name): i for i, name in enumerate(class_names)
}

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
).configure(
    InferenceConfiguration(api_key_transport="header")
)

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

if len(image_paths) != 76:
    print(f"WARNING: expected 76 images, found {len(image_paths)}")

# Predictions retain every returned confidence for AP calculations.
predictions_by_image = {}
timings = []
failures = []
rows = []

for i, path in enumerate(image_paths, 1):
    print(f"[{i}/{len(image_paths)}] {path.name}")

    try:
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
        predictions = result[0]["predictions"]["predictions"]

        with Image.open(path) as im:
            img_w, img_h = im.size

        cleaned = []
        for pred in predictions:
            label = str(pred["class"])
            if label not in class_to_id:
                print(f"  WARNING: unknown class name: {label}")
                continue

            conf = float(pred["confidence"])
            x = float(pred["x"])
            y = float(pred["y"])
            w = float(pred["width"])
            h = float(pred["height"])

            # Assumes workflow boxes use original-image pixel coordinates.
            # Verify this assumption before accepting final metrics.
            x1 = max(0.0, (x - w / 2) / img_w)
            y1 = max(0.0, (y - h / 2) / img_h)
            x2 = min(1.0, (x + w / 2) / img_w)
            y2 = min(1.0, (y + h / 2) / img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            cleaned.append({
                "class_id": class_to_id[label],
                "class": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
            })

        predictions_by_image[path.name] = cleaned

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
        failures.append({"image": path.name, "error": str(exc)})
        predictions_by_image[path.name] = []
        print(f"  FAILED: {exc}")

# Save predictions and request failures.
with open(OUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
    fields = [
        "image", "class", "confidence",
        "x1_norm", "y1_norm", "x2_norm", "y2_norm",
        "request_time_seconds", "status"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with open(OUT_DIR / "failures.json", "w", encoding="utf-8") as f:
    json.dump(failures, f, indent=2)

# Load YOLO ground-truth labels: class x_center y_center width height.
ground_truth = {}
for path in image_paths:
    label_file = LABEL_DIR / f"{path.stem}.txt"
    gt = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, x, y, w, h = parts
            x, y, w, h = map(float, (x, y, w, h))
            gt.append({
                "class_id": int(cid),
                "box": [
                    x - w / 2, y - h / 2,
                    x + w / 2, y + h / 2
                ]
            })
    ground_truth[path.name] = gt


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_at_iou(cid, threshold):
    """Greedy confidence-ranked one-to-one matching."""
    gt_by_image = {
        name: [g for g in gt if g["class_id"] == cid]
        for name, gt in ground_truth.items()
    }
    total_gt = sum(len(v) for v in gt_by_image.values())
    detections = []

    for name, preds in predictions_by_image.items():
        for p in preds:
            if p["class_id"] == cid and p["confidence"] >= THRESHOLD:
                detections.append((p["confidence"], name, p["box"]))

    detections.sort(key=lambda d: d[0], reverse=True)
    matched = {name: set() for name in gt_by_image}
    tp, fp = [], []

    for conf, name, box in detections:
        candidates = gt_by_image[name]
        best_idx, best_iou = None, 0.0
        for j, gt in enumerate(candidates):
            score = iou(box, gt["box"])
            if j not in matched[name] and score > best_iou:
                best_idx, best_iou = j, score

        if best_idx is not None and best_iou >= threshold:
            matched[name].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_total, fp_total = sum(tp), sum(fp)
    fn_total = total_gt - tp_total
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0
    recall = tp_total / total_gt if total_gt else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall else 0
    )
    return precision, recall, f1, total_gt, tp_total, fp_total, fn_total


def average_precision(cid, iou_threshold):
    """Interpolated AP over all confidence-ranked predictions."""
    gt_by_image = {
        name: [g for g in gt if g["class_id"] == cid]
        for name, gt in ground_truth.items()
    }
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return None

    detections = []
    for name, preds in predictions_by_image.items():
        for p in preds:
            if p["class_id"] == cid:
                detections.append((p["confidence"], name, p["box"]))
    detections.sort(key=lambda d: d[0], reverse=True)

    matched = {name: set() for name in gt_by_image}
    tp, fp = [], []
    for conf, name, box in detections:
        candidates = gt_by_image[name]
        best_idx, best_iou = None, 0.0
        for j, gt in enumerate(candidates):
            score = iou(box, gt["box"])
            if j not in matched[name] and score > best_iou:
                best_idx, best_iou = j, score
        if best_idx is not None and best_iou >= iou_threshold:
            matched[name].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0

    import numpy as np
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])

    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum(
        (recall[changes + 1] - recall[changes]) * precision[changes + 1]
    ))


# Overall metrics at the chosen confidence and IoU threshold.
per_class = []
for cid, name in enumerate(class_names):
    p, r, f1, n_gt, tp, fp, fn = match_at_iou(cid, IOU_MATCH)
    ap50 = average_precision(cid, 0.50)
    aps = [
        average_precision(cid, t)
        for t in [x / 100 for x in range(50, 100, 5)]
    ]
    aps_valid = [x for x in aps if x is not None]
    ap5095 = sum(aps_valid) / len(aps_valid) if aps_valid else None

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

# Micro precision/recall/F1 across classes.
tot_tp = sum(x["TP"] for x in per_class)
tot_fp = sum(x["FP"] for x in per_class)
tot_fn = sum(x["FN"] for x in per_class)
P = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0
R = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0
F1 = 2 * P * R / (P + R) if P + R else 0

valid_ap50 = [x["mAP50"] for x in per_class if x["mAP50"] is not None]
valid_ap5095 = [x["mAP50_95"] for x in per_class if x["mAP50_95"] is not None]

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

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n--- RF-DETR evaluation summary ---")
print(json.dumps(summary, indent=2))
print(f"\nSaved results to: {OUT_DIR.resolve()}")
