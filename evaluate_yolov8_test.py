#!/usr/bin/env python3
"""Evaluate exported YOLOv8n weights on the held-out Pork Rasher test split.

Usage (from the project directory):
    python evaluate_yolov8_test.py

Uses files that have ALREADY been extracted:
  - Dataset: a folder containing test/images, test/labels, and data.yaml
  - Model exports: ./results/Notebook_results (or another matching folder)

No ZIP files are opened or extracted. The script fixes dataset YAML paths,
evaluates BOTH baseline and corrected-augmentation best.pt weights, and writes
JSON metrics. The top-level yolov8n_test_metrics.json is for the corrected-
augmentation model, which is the default chosen model for the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
# The user's model export is already extracted here.
EXPORT_DIR_CANDIDATES = [
    BASE_DIR / "results" / "Notebook_results",
    BASE_DIR / "Notebook_results",
    BASE_DIR / "results" / "MMA3001_YOLOv8n_Export",
    BASE_DIR / "MMA3001_YOLOv8n_Export",
]
OUTPUT_DIR = BASE_DIR / "results" / "yolov8_test_evaluation"

# Report figure script reads these exact keys from yolov8n_test_metrics.json.
SELECTED_MODEL = "corrected_augmented" 
IMGSZ = 640
BATCH = 16
CONF = 0.001  # low threshold for AP computation; standard Ultralytics val default


def find_dataset_root(search_root: Path) -> Path:
    """Find the extracted dataset, prioritising the user's confirmed path."""
    direct = search_root / "Pork Rasher Error-Packaging-.v1i.yolov8"
    candidates = [direct]
    # Also allow an alternate folder name or one extra nesting level.
    if search_root.is_dir():
        candidates.extend(
            p for p in search_root.iterdir()
            if p.is_dir() and p.name != "results" and not p.name.startswith(".") and p != direct
        )
        for parent in list(candidates):
            if parent.is_dir():
                candidates.extend(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith("."))
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if ((candidate / "test" / "images").is_dir()
                and (candidate / "test" / "labels").is_dir()
                and (candidate / "data.yaml").is_file()):
            return candidate
    raise FileNotFoundError(
        "Could not find the extracted dataset. Expected folders like:\n"
        f"  {direct / 'test' / 'images'}\n"
        f"  {direct / 'test' / 'labels'}\n"
        f"  {direct / 'data.yaml'}"
    )


def find_export_root() -> Path:
    """Find the extracted Notebook_results/model-export directory."""
    for candidate in EXPORT_DIR_CANDIDATES:
        if candidate.is_dir() and list(candidate.glob("**/best.pt")):
            return candidate
    # Fallback: find a directory under results that contains model weights.
    results_dir = BASE_DIR / "results"
    if results_dir.is_dir():
        for weights in results_dir.glob("**/best.pt"):
            if "yolov8_test_evaluation" not in str(weights):
                return weights.parents[2] if len(weights.parents) > 2 else weights.parent
    searched = "\n".join(f"  - {p}" for p in EXPORT_DIR_CANDIDATES)
    raise FileNotFoundError(f"Could not find extracted model weights. Checked:\n{searched}")


def find_weight(export_root: Path, model_name: str) -> Path:
    matches = list(export_root.glob(f"**/{model_name}/weights/best.pt"))
    if not matches:
        raise FileNotFoundError(f"Could not find best.pt for model '{model_name}' under {export_root}")
    # Prefer model_weights copy if present; otherwise use training_outputs copy.
    matches.sort(key=lambda p: ("model_weights" not in str(p), len(str(p))))
    return matches[0]


def write_fixed_data_yaml(dataset_root: Path) -> Path:
    """Write a YAML with absolute paths, avoiding broken ../ paths in Roboflow ZIP."""
    try:
        import yaml  # provided as an Ultralytics dependency
    except ImportError as exc:
        raise RuntimeError("PyYAML is missing. Install it with: python -m pip install pyyaml") from exc

    source_yaml = dataset_root / "data.yaml"
    if source_yaml.exists():
        config = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    else:
        config = {}
    names = config.get("names", ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"])
    config["path"] = str(dataset_root.resolve())
    config["train"] = str((dataset_root / "train" / "images").resolve())
    config["val"] = str((dataset_root / "valid" / "images").resolve())
    config["test"] = str((dataset_root / "test" / "images").resolve())
    config["nc"] = len(names)
    config["names"] = names
    fixed_yaml = OUTPUT_DIR / "data_fixed.yaml"
    fixed_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return fixed_yaml


def metric_value(obj: Any, *attrs: str) -> float | None:
    for attr in attrs:
        try:
            value = getattr(obj, attr)
            if value is not None:
                return float(value)
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def evaluate_one(model_name: str, weights: Path, data_yaml: Path, dataset_root: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    print(f"\nEvaluating {model_name}")
    print(f"Weights: {weights}")
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        conf=CONF,
        plots=True,
        project=str(OUTPUT_DIR),
        name=model_name,
        exist_ok=True,
        verbose=True,
    )
    box = results.box
    metrics: dict[str, Any] = {
        "model": model_name,
        "weights": str(weights),
        "dataset_yaml": str(data_yaml),
        "split": "test",
        "image_size": IMGSZ,
        "test_images": len(list((dataset_root / "test" / "images").glob("*"))),
        "precision": metric_value(box, "mp"),
        "recall": metric_value(box, "mr"),
        "map50": metric_value(box, "map50"),
        "map50_95": metric_value(box, "map"),
        "per_class": {},
        "results_directory": str(getattr(results, "save_dir", OUTPUT_DIR / model_name)),
        "metric_source": "Ultralytics model.val(split='test') on held-out test split",
    }
    names = model.names
    p = getattr(box, "p", None)
    r = getattr(box, "r", None)
    ap50 = getattr(box, "ap50", None)
    ap = getattr(box, "ap", None)
    class_ids = getattr(box, "ap_class_index", [])
    for i, class_id in enumerate(class_ids):
        cid = int(class_id)
        class_name = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
        metrics["per_class"][class_name] = {
            "class_id": cid,
            "precision": float(p[i]) if p is not None and len(p) > i else None,
            "recall": float(r[i]) if r is not None and len(r) > i else None,
            "map50": float(ap50[i]) if ap50 is not None and len(ap50) > i else None,
            "map50_95": float(ap[i].mean()) if ap is not None and len(ap) > i else None,
        }
    # Keep only standard JSON numbers (avoid numpy scalar serialization problems).
    out_path = OUTPUT_DIR / f"{model_name}_test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(json.dumps({k: metrics[k] for k in ("precision", "recall", "map50", "map50_95")}, indent=2))
    return metrics


def main() -> int:
    dataset_root = find_dataset_root(BASE_DIR)
    export_root = find_export_root()
    print(f"Dataset root: {dataset_root}")
    print(f"Model export root: {export_root}")
    print(f"Test images: {len(list((dataset_root / 'test' / 'images').glob('*')))}")
    print(f"Test labels: {len(list((dataset_root / 'test' / 'labels').glob('*.txt')))}")

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("Ultralytics is not installed. Install dependencies first:", file=sys.stderr)
        print("  python -m pip install ultralytics pyyaml", file=sys.stderr)
        return 3

    data_yaml = write_fixed_data_yaml(dataset_root)
    baseline_weights = find_weight(export_root, "yolov8n_baseline")
    augmented_weights = find_weight(export_root, "yolov8n_corrected_augmented")
    all_metrics = {
        "baseline": evaluate_one("yolov8n_baseline", baseline_weights, data_yaml, dataset_root),
        "corrected_augmented": evaluate_one("yolov8n_corrected_augmented", augmented_weights, data_yaml, dataset_root),
    }
    selected = all_metrics[SELECTED_MODEL]
    report_path = BASE_DIR / "yolov8n_test_metrics.json"
    report_path.write_text(json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8")
    summary_path = OUTPUT_DIR / "comparison_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nReport metrics saved for '{SELECTED_MODEL}' to: {report_path}")
    print(f"Both model results saved to: {summary_path}")
    print("Next, run: python make_report_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
