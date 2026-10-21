"""
MMA3001 report figures — robust version.

Run from your project folder:
    python -u make_report_figures.py

Reads:
  results/rfdetr_v1_test/metrics.json
  results/rfdetr_v1_test/failures.json (optional)
  results/rfdetr_v1_test/predictions.csv (optional)
  results/yolov8_test_evaluation/yolov8n_corrected_augmented/yolov8n_corrected_augmented_test_metrics.json
  Dataset train/labels/*.txt (optional)
  results/threshold_results.csv (optional)

Writes figures and supporting CSV/JSON files to ./report_figures/

If your project stores the dataset elsewhere, set DATASET_ROOT below to its path.
"""

from pathlib import Path
import csv
import json
import sys
import traceback

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print("ERROR: matplotlib is not installed in this Python environment.")
    print("Install it with: python -m pip install matplotlib")
    raise

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_figures"
RF_METRICS = ROOT / "results" / "rfdetr_v1_test" / "metrics.json"
RF_FAILURES = ROOT / "results" / "rfdetr_v1_test" / "failures.json"
RF_PREDICTIONS = ROOT / "results" / "rfdetr_v1_test" / "predictions.csv"
YOLO_METRICS = (
    ROOT / "results" / "yolov8_test_evaluation"
    / "yolov8n_corrected_augmented_test_metrics.json"
)
THRESHOLD_CSV = ROOT / "results" / "threshold_results.csv"

# Optional: if the dataset is outside the project folder, put its full path here.
DATASET_ROOT = None
CLASS_NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]

# Do not use fallback metrics: stop if the fresh test JSON is missing.
YOLO_MODEL_LABEL = "YOLOv8n (corrected augmentation)"


def load_json(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, filename):
    destination = OUT / filename
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {destination} ({destination.stat().st_size:,} bytes)")


def find_dataset_root():
    if DATASET_ROOT:
        p = Path(DATASET_ROOT).expanduser()
        if p.exists():
            return p
        print(f"WARNING: DATASET_ROOT does not exist: {p}")
    candidates = [
        ROOT,
        ROOT / "dataset",
        ROOT / "data",
        ROOT / "Pork Rasher Error-Packaging-.v1i.yolov8",
        Path.home() / "Downloads" / "Pork Rasher Error-Packaging-.v1i.yolov8",
    ]
    for candidate in candidates:
        if (candidate / "train" / "labels").is_dir():
            return candidate
    return None


def read_training_class_counts():
    dataset_root = find_dataset_root()
    if not dataset_root:
        print("WARNING: Could not find dataset train/labels; skipping class distribution graph.")
        return None

    labels_dir = dataset_root / "train" / "labels"
    counts = {name: 0 for name in CLASS_NAMES}
    txt_files = list(labels_dir.glob("*.txt"))
    if not txt_files:
        print(f"WARNING: No label .txt files found in {labels_dir}; skipping class distribution graph.")
        return None

    for label_file in txt_files:
        try:
            with label_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if 0 <= class_id < len(CLASS_NAMES):
                        counts[CLASS_NAMES[class_id]] += 1
        except OSError as exc:
            print(f"WARNING: Could not read {label_file}: {exc}")

    return dataset_root, counts


def main():
    print(f"Script: {Path(__file__).resolve()}")
    print(f"Project root: {ROOT}")
    print(f"Output folder: {OUT}")
    print(f"RF-DETR metrics: {RF_METRICS}")
    print(f"YOLO metrics: {YOLO_METRICS}")
    print(f"Metrics file exists: {RF_METRICS.exists()}")

    if not RF_METRICS.exists():
        raise FileNotFoundError(
            f"Could not find {RF_METRICS}. Run the RF-DETR test evaluation first "
            "or update RF_METRICS to the correct location."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    rf = load_json(RF_METRICS)
    if not isinstance(rf, dict):
        raise ValueError("RF-DETR metrics JSON must contain a JSON object.")

    per_class = rf.get("per_class", [])
    if not isinstance(per_class, list) or not per_class:
        raise ValueError(
            "metrics.json has no non-empty 'per_class' list. "
            f"Top-level keys found: {list(rf.keys())}"
        )

    # RF-DETR overall values from the custom evaluator.
    required_rf_keys = [
        "precision_micro",
        "recall_micro",
        "mAP50_macro_over_classes_with_ground_truth",
        "mAP50_95_macro_over_classes_with_ground_truth",
    ]
    missing = [key for key in required_rf_keys if key not in rf]
    if missing:
        raise KeyError(
            f"Missing expected RF-DETR metric keys: {missing}. "
            f"Available keys: {list(rf.keys())}"
        )

    rp = float(rf["precision_micro"])
    rr = float(rf["recall_micro"])
    rm50 = float(rf["mAP50_macro_over_classes_with_ground_truth"])
    rm95 = float(rf["mAP50_95_macro_over_classes_with_ground_truth"])
    rf1 = 2 * rp * rr / (rp + rr) if (rp + rr) else 0.0
    rmean = rf.get("mean_request_seconds")
    rmedian = rf.get("median_request_seconds")

    print(f"YOLO metrics path: {YOLO_METRICS}")
    print(f"YOLO metrics file exists: {YOLO_METRICS.exists()}")
    if not YOLO_METRICS.exists():
        raise FileNotFoundError(
            f"Could not find the corrected-augmentation YOLO test metrics file: {YOLO_METRICS}\\n"
            "Confirm the JSON filename and location in results/yolov8_test_evaluation."
        )
    yolo = load_json(YOLO_METRICS, required=True)
    if not isinstance(yolo, dict):
        raise ValueError("YOLO metrics JSON must contain a JSON object.")
    required_yolo_keys = ["precision", "recall", "map50", "map50_95"]
    missing_yolo_keys = [key for key in required_yolo_keys if key not in yolo]
    if missing_yolo_keys:
        raise KeyError(
            f"YOLO metrics JSON is missing required keys: {missing_yolo_keys}. "
            f"Available keys: {list(yolo.keys())}"
        )
    yp = float(yolo["precision"])
    yr = float(yolo["recall"])
    ym50 = float(yolo["map50"])
    ym95 = float(yolo["map50_95"])
    yolo_source = str(YOLO_METRICS)
    print(
        f"Loaded {YOLO_MODEL_LABEL} metrics: "
        f"precision={yp:.4f}, recall={yr:.4f}, "
        f"mAP50={ym50:.4f}, mAP50-95={ym95:.4f}"
    )

    # Figure 1: overall metric comparison.
    metric_names = ["Precision", "Recall", "mAP@50", "mAP@50:95"]
    yolo_values = [yp, yr, ym50, ym95]
    rf_values = [rp, rr, rm50, rm95]
    x = list(range(len(metric_names)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar([i - width / 2 for i in x], yolo_values, width, label=YOLO_MODEL_LABEL)
    ax.bar([i + width / 2 for i in x], rf_values, width, label="RF-DETR")
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8n corrected augmentation vs RF-DETR — test metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    save_fig(fig, "01_model_metrics_comparison.png")

    # Figure 2: RF-DETR per-class recall.
    names, recalls, instance_counts = [], [], []
    for item in per_class:
        name = str(item.get("class_name", item.get("name", item.get("class", "unknown"))))
        gt = item.get("ground_truth_instances", item.get("instances", item.get("gt_instances", 0)))
        recall = item.get("recall", None)
        if recall is None:
            print(f"WARNING: Missing recall for class '{name}'; skipping it.")
            continue
        names.append(name)
        recalls.append(float(recall))
        try:
            instance_counts.append(int(gt))
        except (TypeError, ValueError):
            instance_counts.append(0)

    if names:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        bars = ax.bar(names, recalls)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title("RF-DETR recall by defect class")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for bar, rec, count in zip(bars, recalls, instance_counts):
            ax.annotate(
                f"{rec:.3f}\n(n={count})",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )
        save_fig(fig, "02_rfdetr_per_class_recall.png")
    else:
        print("WARNING: No usable per-class recall values; skipping recall graph.")

    # Figure 3: training-set class distribution, if dataset labels are available.
    class_data = read_training_class_counts()
    if class_data:
        dataset_root, counts = class_data
        fig, ax = plt.subplots(figsize=(9, 5.4))
        bars = ax.bar(list(counts.keys()), list(counts.values()))
        ax.set_ylabel("Number of labelled instances")
        ax.set_title("Training-set defect class distribution")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for bar in bars:
            ax.annotate(
                f"{int(bar.get_height())}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )
        save_fig(fig, "03_training_class_distribution.png")
        with (OUT / "training_class_counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "training_instances"])
            writer.writerows(counts.items())
        print(f"Training dataset root: {dataset_root}")

    # Figure 4: confidence threshold curve, if threshold sweep data exists.
    if THRESHOLD_CSV.exists():
        rows = []
        with THRESHOLD_CSV.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            headers = rows[0].keys()
            threshold_key = next((k for k in headers if k.lower() in {"threshold", "confidence_threshold", "conf"}), None)
            f1_key = next((k for k in headers if k.lower() in {"f1", "f1_score"}), None)
            precision_key = next((k for k in headers if k.lower() == "precision"), None)
            recall_key = next((k for k in headers if k.lower() == "recall"), None)
            if threshold_key and (f1_key or (precision_key and recall_key)):
                thresholds = [float(row[threshold_key]) for row in rows]
                fig, ax = plt.subplots(figsize=(8, 5))
                if f1_key:
                    f1s = [float(row[f1_key]) for row in rows]
                    ax.plot(thresholds, f1s, marker="o", label="F1")
                if precision_key:
                    ax.plot(thresholds, [float(row[precision_key]) for row in rows], marker="o", label="Precision")
                if recall_key:
                    ax.plot(thresholds, [float(row[recall_key]) for row in rows], marker="o", label="Recall")
                ax.set_xlabel("Confidence threshold")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1)
                ax.set_title("Performance versus confidence threshold")
                ax.grid(alpha=0.25)
                ax.legend()
                save_fig(fig, "04_confidence_threshold_sweep.png")
            else:
                print(f"WARNING: {THRESHOLD_CSV} lacks expected threshold/F1/precision/recall columns; skipping threshold graph.")
        else:
            print(f"WARNING: {THRESHOLD_CSV} is empty; skipping threshold graph.")
    else:
        print("INFO: No threshold_results.csv found; skipping threshold-sweep graph.")

    # Figure 5: inference timing, if available.
    timing_values, timing_labels = [], []
    if rmean is not None:
        timing_values.append(float(rmean))
        timing_labels.append("RF-DETR API\n(mean)")
    if rmedian is not None:
        timing_values.append(float(rmedian))
        timing_labels.append("RF-DETR API\n(median)")

    # The local YOLO timing reported previously was 33.43 ms/image.
    timing_values.append(0.03343)
    timing_labels.append("YOLOv8n local\n(reported)")
    if timing_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(timing_labels, timing_values)
        ax.set_ylabel("Seconds per image")
        ax.set_title("Inference time per image (timing methods differ)")
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f} s",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )
        save_fig(fig, "05_inference_timing.png")

    # Figure 6: simple computational workflow diagram.
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    workflow = [
        ("Input image", 0.08),
        ("Pre-processing", 0.25),
        ("Object detector", 0.43),
        ("Confidence filtering", 0.62),
        ("Defect output", 0.81),
    ]
    for label, xpos in workflow:
        ax.text(
            xpos, 0.52, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.6", fill=False)
        )
    for i in range(len(workflow) - 1):
        x1 = workflow[i][1] + 0.07
        x2 = workflow[i + 1][1] - 0.07
        ax.annotate(
            "", xy=(x2, 0.52), xytext=(x1, 0.52),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.4)
        )
    ax.set_title("Computational workflow for pork rasher defect detection", pad=12)
    save_fig(fig, "06_computational_workflow.png")

    # Save source provenance and metric values used in plots.
    provenance = {
        "project_root": str(ROOT),
        "rfdetr_metrics_file": str(RF_METRICS),
        "rfdetr_metrics_file_exists": RF_METRICS.exists(),
        "yolo_metrics_source": yolo_source,
        "metrics": {
            YOLO_MODEL_LABEL: {"precision": yp, "recall": yr, "mAP50": ym50, "mAP50_95": ym95},
            "RF-DETR": {"precision": rp, "recall": rr, "F1_calculated": rf1, "mAP50": rm50, "mAP50_95": rm95},
        },
        "caveat": "Confirm that both models use the same test set, IoU/confidence settings, and evaluation method before treating the comparison as definitive.",
    }
    if rmean is not None:
        provenance["rfdetr_mean_request_seconds"] = float(rmean)
    if rmedian is not None:
        provenance["rfdetr_median_request_seconds"] = float(rmedian)

    with (OUT / "figure_data_sources.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    with (OUT / "model_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "recall", "F1_calculated", "mAP50", "mAP50_95", "source"])
        writer.writerow([YOLO_MODEL_LABEL, yp, yr, 2 * yp * yr / (yp + yr) if yp + yr else 0, ym50, ym95, yolo_source])
        writer.writerow(["RF-DETR", rp, rr, rf1, rm50, rm95, str(RF_METRICS)])

    print("\nAll available report figures have been generated.")
    print(f"Output folder: {OUT}")
    print("Files created:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
