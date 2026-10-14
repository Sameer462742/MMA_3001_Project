# Automated Pork Rasher Packaging Defect Detection

Computer vision project for detecting and localising packaging defects in pork rasher images using object detection.

## Project Overview

This project investigates the use of computer vision to identify packaging defects in pork rasher products. A trained object-detection model processes images and returns predicted defect classes, confidence scores and bounding boxes.

The deployed inference pipeline uses a Roboflow-hosted RF-DETR model. YOLOv8 is also used for training experiments and performance comparisons.

The repository includes two ways to inspect images:

- **Streamlit application (`app.py`)** — a browser-based interface for uploading images, viewing annotated predictions, checking summary statistics and downloading reports.
- **Batch inference script (`main.py`)** — a command-line workflow that processes images from `test_images/` and writes results to `results/`, if this script is included in the repository.

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
- Browser-based image upload and multi-image inspection using Streamlit.
- Batch processing of images.
- Annotated image generation.
- CSV export of model predictions and inspection statuses.
- Download of annotated images as a ZIP archive from the app.
- Summary counts and a chart of detections by class.
- Separate reporting for inference errors and images with no detections above the threshold.
- YOLOv8 training and evaluation experiments.
- Comparison of object-detection performance.

> **Important:** `NO_DEFECT_DETECTED` means no prediction passed the selected confidence threshold. It does not prove that an image is defect-free. Predictions must be evaluated against labelled ground truth to measure accuracy.

## Project Structure

A typical project layout is:

```text
.
├── app.py                    # Streamlit browser application
├── main.py                   # Optional batch inference script
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
├── .env.example              # Example configuration only; no real API key
├── test_images/              # Optional images for batch inference
├── results/                  # Generated batch outputs, if using main.py
├── notebooks/                # Training and analysis notebooks
└── docs/                     # Additional documentation
```

The Streamlit application accepts images directly through the browser, so a local `test_images/` folder is not required to use `app.py`. Keep generated results and large datasets out of version control unless there is a clear project reason to include them.

## Requirements

- Python version compatible with the installed Roboflow Inference SDK and Streamlit dependencies.
- Roboflow Inference SDK.
- Streamlit.
- Pillow.
- pandas.
- A Roboflow account with access to the deployed workflow.
- Internet access while making hosted inference requests.

Matplotlib is used by the original batch script if retained. YOLOv8 training experiments may require additional dependencies.

## Installation

Open Terminal in the project directory. If using Anaconda, activate the relevant environment first.

```bash
python -m pip install -r requirements.txt
```

Ensure `requirements.txt` includes at least these packages for the Streamlit app:

```text
streamlit
inference-sdk
pillow
pandas
```

If you use `main.py`, include any extra packages imported by that script, such as `matplotlib`.

## Configuration and API Key

A valid Roboflow API key and access to the deployed workflow are required.

### Recommended configuration

Keep the API key outside source control. For macOS/Linux Terminal:

```bash
export ROBOFLOW_API_KEY="YOUR_NEW_API_KEY"
```

Launch the app from the same Terminal session so the variable is available. For Streamlit Community Cloud, add `ROBOFLOW_API_KEY` through the app's Secrets settings.

### Local hardcoded-key version

The local-testing version of `app.py` may instead contain this placeholder near the top:

```python
API_KEY = "siB1OGhXN0kGj0mPwzwM"
```

Replace the placeholder with a newly generated key only if you deliberately choose to hardcode it for local testing. This is **not recommended for a shared repository**. Do not commit or push the file while it contains a real key. If a key is exposed, revoke or rotate it.

Configure the workspace and workflow identifiers to match the deployed Roboflow workflow:

- Workspace: `ece4078h3`
- Workflow: `pork-rasher-errorpackaging-vpork-rasher-error-packaging-rj5da-1-rfdetr-small-t1-logic`

## Running the Streamlit Application

**Do not launch the Streamlit app with `python app.py`.** Use the Streamlit command from the project directory:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

Streamlit prints a local URL, commonly:

```text
http://localhost:8501
```

Open that address in your browser if it does not open automatically.

### Using the application

1. Start the app with `python -m streamlit run app.py`.
2. In the sidebar, choose the minimum confidence threshold. The initial default is 50%; it is a starting value, not a validated optimum.
3. Upload one or more supported images: JPG, JPEG, PNG, BMP or WEBP.
4. Click **Run defect detection**.
5. Wait while each image is sent to the Roboflow workflow.
6. Review the summary, detections by class and individual image results.
7. Expand an image result to compare the original image with annotated predictions, confidence scores and status.
8. Click **Download CSV report** to save the prediction table.
9. Click **Download annotated images (ZIP)** to save the annotated images.

### Inspection statuses

- `DEFECT_DETECTED`: at least one valid prediction passed the selected threshold.
- `NO_DEFECT_DETECTED`: no valid prediction passed the threshold. This does not guarantee that the image is defect-free.
- `INFERENCE_ERROR`: the image could not be processed. Review the displayed error; do not treat a failed inspection as defect-free.

The CSV contains image names, predicted classes, confidence values, bounding-box coordinates, threshold, status, inference time and error information where applicable. Images with no detections are included in the report.

## Running Batch Inference with `main.py`

If `main.py` is included, place supported images in `test_images/` and run:

```bash
python main.py
```

The script processes images, filters predictions using its configured confidence threshold, draws bounding boxes and saves results. Output locations depend on the settings in `main.py`. The original batch script writes:

- `results/predictions.csv`
- `results/annotated_images/`

The batch script and Streamlit app are separate entry points. The Streamlit app accepts uploads directly and does not require `test_images/`.

## Model Evaluation

The project evaluates object-detection performance using metrics such as:

- **Precision** — the proportion of predicted detections that are correct.
- **Recall** — the proportion of labelled objects that are detected.
- **mAP@50** — mean average precision at an intersection-over-union (IoU) threshold of 0.50.
- **mAP@50–95** — mean average precision averaged over IoU thresholds from 0.50 to 0.95.

Predictions must be compared against ground-truth annotations to calculate meaningful detection accuracy metrics. Dashboard counts and confidence scores are not substitutes for ground-truth evaluation.

For credible model comparisons:

- Evaluate RF-DETR and YOLOv8 on the same labelled held-out test images.
- Keep the test set separate from threshold selection and model tuning.
- Report overall and per-class results.
- Document the confidence threshold, IoU settings, dataset version and evaluation procedure.
- Investigate false negatives, false positives and classes with few examples.
- Measure inference time under consistent conditions when comparing runtime.

## Reproducibility

Experiments should document:

- Dataset version and train/validation/test splits.
- Model architecture and pretrained weights.
- Training parameters and augmentation settings.
- Confidence and IoU thresholds used during evaluation.
- Hardware and software versions.
- Per-class and overall evaluation metrics.
- Annotation corrections and dataset changes.

Keep API keys and other credentials out of source control. Avoid committing large datasets, model weights, generated images and result files unless needed and permitted.

## Troubleshooting

### `missing ScriptRunContext` warnings

The app was probably started with `python app.py`. Stop it with `Ctrl+C` and run:

```bash
python -m streamlit run app.py
```

### API key not found

If using an environment variable, set `ROBOFLOW_API_KEY` in the same Terminal session before launching the app. If using the local hardcoded-key version, ensure the placeholder in `app.py` has been replaced with a valid, newly generated key.

### Workflow response format error

The app expects the workflow response to contain a prediction list. If the workflow output schema changes, inspect the Roboflow workflow output and update the response parsing function in `app.py`.

### Bounding boxes do not align with the image

The app assumes the workflow's coordinates refer to the original uploaded image dimensions. Check the workflow's coordinate convention and any resize/crop steps. Apply coordinate scaling if required.

### API or network errors

Check internet access, API key permissions, workspace and workflow identifiers, and Roboflow service availability. Failed inspections are reported as `INFERENCE_ERROR`.

## Limitations

- Performance may vary between defect classes due to class imbalance, limited examples and annotation quality.
- A confidence score is not a guarantee that a prediction is correct.
- A lack of detections above the selected threshold does not certify that the product is defect-free.
- Hosted inference depends on network connectivity, service availability, API access and usage limits.
- The current interface is a prototype for project evaluation, not a certified production food-safety inspection system.
- Bounding-box visualisation must be checked to confirm that coordinates match the uploaded image dimensions.

## Academic Context

Developed as part of MMA3001, Numerical Methods and Machine Learning, focusing on the engineering application of computer vision for automated packaging defect detection.

## Acknowledgements

Dataset management and hosted inference are provided by Roboflow. YOLOv8 experiments use the Ultralytics framework.
