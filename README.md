# Automated Pork Rasher Packaging Defect Detection

Computer vision project for detecting and localising packaging defects in pork rasher images using object detection.

## Project Overview

This project investigates the use of computer vision to identify packaging defects in pork rasher products. A trained object detection model processes images and returns predicted defect classes, confidence scores and bounding boxes.

The deployed inference pipeline uses a Roboflow-hosted RF-DETR model. YOLOv8 is also used for training experiments and performance comparisons.

## Defect Classes

The dataset includes the following classes:

- `loose-meat`
- `packaging-error`
- `twisted-meat`
- `unsealed`
- `wrinkle`

Class availability and evaluation performance depend on the dataset split and the examples available for each class.

## Features

- Hosted inference through the Roboflow API.
- Object detection with confidence scores and bounding boxes.
- Batch processing of multiple images.
- Annotated image generation.
- CSV export of model predictions.
- YOLOv8 training and evaluation experiments.
- Comparison of object detection performance.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── test_images/
├── results/
├── notebooks/
└── docs/
```

## Requirements

- Python 3.10 or a compatible version supported by the installed inference SDK.
- Roboflow Inference SDK.
- Pillow.
- Matplotlib.
- A Roboflow account and access to the deployed workflow.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set your Roboflow API key as an environment variable. Do not hardcode credentials in the source code.

```bash
export ROBOFLOW_API_KEY="YOUR_API_KEY"
```

Configure the workspace and workflow identifiers in `main.py` to match your deployed Roboflow workflow.

## Running Inference

Place supported images in the `test_images/` directory, then run:

```bash
python main.py
```

The batch-testing script processes images, filters predictions using a configurable confidence threshold, draws bounding boxes and saves results.

Generated outputs include:

- `results/predictions.csv`
- `results/annotated_images/`

Output filenames and locations depend on the configuration in `main.py`.

## Model Evaluation

The project evaluates detection performance using metrics such as:

- Precision
- Recall
- Mean Average Precision at IoU 0.50 (mAP@50)
- Mean Average Precision across IoU thresholds 0.50–0.95 (mAP@50–95)

Predictions must be compared against ground-truth annotations to calculate meaningful detection accuracy metrics.

## Reproducibility

Experiments should document:

- Dataset version and train/validation/test splits.
- Model architecture and pretrained weights.
- Training parameters and augmentation settings.
- Confidence and IoU thresholds used during evaluation.
- Hardware and software versions.
- Per-class and overall evaluation metrics.

## Limitations

Performance may vary between defect classes due to class imbalance, limited examples and annotation quality. Hosted inference also depends on network connectivity and service availability.

## Academic Context

Developed as part of MMA3001, focusing on the engineering application of computer vision for automated defect detection.

## Acknowledgements

Dataset management and hosted inference are provided by Roboflow. YOLOv8 experiments use the Ultralytics framework.

