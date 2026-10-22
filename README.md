# Automated Pork Rasher Packaging Defect Detection

A computer-vision project for detecting and localising packaging and product-presentation defects in pork rasher images using object detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Defect Classes](#defect-classes)
- [Features](#features)
- [Example Input and Output](#example-input-and-output)
- [Dataset](#dataset)
- [Models and Training](#models-and-training)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration and API Key](#configuration-and-api-key)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Batch Inference](#batch-inference)
- [Automated Testing](#automated-testing)
- [HTML Source-Code Documentation](#html-source-code-documentation)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Academic Context](#academic-context)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project investigates computer-vision techniques for identifying packaging and product-presentation defects in pork rasher product images. The system processes an image and produces predicted defect classes, confidence scores and bounding boxes indicating the locations of detected defects.

Two object-detection approaches were investigated:

- **RF-DETR:** A Roboflow-hosted inference workflow used by the deployed application.
- **YOLOv8n:** A locally trained model used for baseline training, augmentation experiments and model evaluation.

A Streamlit application provides a browser-based interface for uploading images, reviewing predictions, viewing annotated images and exporting inspection results. The project also includes quantitative evaluation, per-class analysis and comparison of the two detection approaches.

The intended application is to support automated visual inspection by identifying potentially defective pork rasher packaging for further review.

> **Important:** This project is a prototype decision-support tool, not a certified food-safety inspection system. `NO_DEFECT_DETECTED` means that no prediction passed the selected confidence threshold. It does not prove that an image or product is defect-free.

## Defect Classes

The dataset defines five defect classes:

| Class | Description |
|---|---|
| `loose-meat` | Loose meat identified by the dataset annotations |
| `packaging-error` | Packaging-related error identified by the dataset annotations |
| `twisted-meat` | Twisted meat identified by the dataset annotations |
| `unsealed` | Unsealed packaging identified by the dataset annotations |
| `wrinkle` | Wrinkling identified by the dataset annotations |

The class distribution is imbalanced. In the training data, `packaging-error` has substantially more labelled instances than the other classes, while `wrinkle` has very few. This imbalance can affect how effectively a model learns to detect each defect type.

The held-out test split used for the reported evaluation contains no labelled `wrinkle` instances. Consequently, the reported test results cannot establish the model's performance for that class.

## Features

The project includes the following functionality:

- Hosted inference through a Roboflow workflow using RF-DETR.
- Object detection with predicted class labels, confidence scores and bounding boxes.
- Streamlit interface for uploading one or multiple images.
- Side-by-side review of original images and annotated predictions.
- Configurable minimum confidence threshold.
- Inspection statuses for detected defects, no qualifying detections and inference errors.
- Summary counts and detections-by-class visualisation.
- CSV export of predictions and inspection statuses.
- ZIP download of annotated images.
- Local YOLOv8n training and evaluation experiments.
- Overall and per-class performance analysis.
- Automated tests for supported application functionality.
- HTML source-code documentation generated with `pdoc`.

## Example Input and Output

The application accepts image files through its Streamlit interface. After the user selects **Run defect detection**, the uploaded image is sent to the configured Roboflow workflow. The returned predictions are processed and displayed in the application.

A useful project demonstration should include:

1. **Input:** An original pork rasher image without annotations.
2. **Output:** The corresponding image with predicted bounding boxes, defect labels and confidence scores.
3. **Application view:** An optional screenshot showing the upload interface, inspection status, prediction results and export controls.

Use actual project images and application outputs for demonstrations rather than illustrative results that could be mistaken for model predictions.

A single example demonstrates how the application operates, but it does not establish model accuracy. Accuracy and detection performance must be evaluated against labelled ground-truth annotations.

## Dataset

The reported dataset is the Roboflow Pork Rasher Error Packaging dataset, Version 1 (V1), exported in YOLOv8 format for local training and evaluation.

### Dataset split

| Split | Number of images |
|---|---:|
| Training | 1,764 |
| Validation | 115 |
| Test | 76 |
| **Total** | **1,955** |

The training class distribution recorded during the project was:

| Class | Training instances |
|---|---:|
| `loose-meat` | 127 |
| `packaging-error` | 1,372 |
| `twisted-meat` | 236 |
| `unsealed` | 479 |
| `wrinkle` | 18 |

These are recorded training-instance counts, not image counts. An image can contain multiple labelled instances.

The imbalance is particularly pronounced for `packaging-error` and `wrinkle`. The small number of `wrinkle` examples may limit the model's ability to generalise to this class.

The held-out test evaluation used 76 images and 91 labelled object instances across four represented classes. There were no labelled `wrinkle` instances in that test split.

### Dataset citation

Before final submission, add the exact Roboflow dataset URL, dataset version and access date required by the project's referencing style. The dataset URL and access date should be verified rather than guessed.

When retraining a model, record the dataset version, split definitions, annotation changes and class mapping so that results can be reproduced.

## Models and Training

### YOLOv8n

YOLOv8n was used for local object-detection training and augmentation experiments. The recorded training configuration included:

| Parameter | Recorded value |
|---|---|
| Model | `yolov8n.pt` |
| Framework | Ultralytics YOLOv8 |
| Image size | 640 × 640 |
| Epochs | 100 |
| Batch size | 16 |
| Number of classes | 5 for the pork rasher dataset |

The training configuration must match the actual dataset YAML file and class definitions used in each experiment. The dataset configuration should specify the correct paths for the training, validation and test images, together with the class names and their corresponding indices.

The project also investigated augmentation changes. The metrics presented below refer to the recorded **corrected-augmentation YOLOv8n experiment**, not every earlier training run.

### RF-DETR

RF-DETR was used through a Roboflow-hosted workflow. The Streamlit application sends images to this workflow and processes the returned predictions.

The hosted workflow and local YOLOv8n model use different inference pathways. RF-DETR inference includes communication with the hosted service, whereas local YOLOv8n inference runs on the machine used for evaluation.

The two approaches should be compared using the same held-out images, class mappings and evaluation methodology before making a definitive performance ranking.

## Model Evaluation

The project evaluates detection performance using precision, recall, F1-score, mean average precision at an IoU threshold of 0.50 (mAP@50), and mean average precision across IoU thresholds from 0.50 to 0.95 (mAP@50–95).

### Reported overall results

| Metric | YOLOv8n corrected augmentation | RF-DETR workflow |
|---|---:|---:|
| Precision | 0.5204 | 0.7733 |
| Recall | 0.8009 | 0.6374 |
| F1-score | 0.6309* | 0.6988 |
| mAP@50 | 0.5987 | 0.5487 |
| mAP@50–95 | 0.2102 | 0.2363 |
| Mean time per image | 33.43 ms (reported) | 703.90 ms |

\* The YOLOv8n F1-score is calculated from the reported precision and recall:

\[
F_1=\frac{2PR}{P+R}
\]

The RF-DETR metrics were generated using a custom evaluation script, while the YOLOv8n metrics were generated using Ultralytics evaluation. Therefore, these values are **provisional and should not be interpreted as a definitive head-to-head benchmark** until both models are evaluated using the same labelled images, class mapping, bounding-box conventions and compatible metric calculations.

The timing measurements also cover different execution pathways. The RF-DETR measurement includes hosted-service and network overhead, while the YOLOv8n value is a reported local inference measurement. They are not a controlled hardware-only speed comparison.

### Metric definitions

- **Precision:** The proportion of predicted detections that match ground-truth objects.
- **Recall:** The proportion of labelled ground-truth objects that the model detects.
- **F1-score:** The harmonic mean of precision and recall.
- **mAP@50:** Mean average precision calculated at an intersection-over-union (IoU) threshold of 0.50.
- **mAP@50–95:** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.
- **Inference time:** The measured time required to produce predictions. The meaning of this measurement depends on whether preprocessing, network communication and postprocessing are included.

### YOLOv8n per-class results

The following results were recorded for the corrected-augmentation YOLOv8n evaluation.

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| `loose-meat` | 1.0000 | 0.9668 | 0.9950 | 0.2728 |
| `packaging-error` | 0.2980 | 0.6881 | 0.3442 | 0.0834 |
| `twisted-meat` | 0.2814 | 0.7143 | 0.2883 | 0.1138 |
| `unsealed` | 0.5022 | 0.8344 | 0.7673 | 0.3709 |
| `wrinkle` | Not measurable | Not measurable | Not measurable | Not measurable |

The `wrinkle` class cannot be evaluated from this test split because it contains no labelled ground-truth instances.

### RF-DETR per-class results

The following values were recorded by the project's custom RF-DETR evaluator.

| Class | Precision | Recall | F1-score | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|
| `loose-meat` | 1.000 | 0.667 | 0.800 | 0.667 | 0.333 |
| `packaging-error` | 0.647 | 0.379 | 0.478 | 0.414 | 0.086 |
| `twisted-meat` | 0.500 | 0.714 | 0.588 | 0.357 | 0.163 |
| `unsealed` | 0.870 | 0.769 | 0.816 | 0.757 | 0.362 |
| `wrinkle` | Not measurable | Not measurable | Not measurable | Not measurable | Not measurable |

The RF-DETR evaluation indicates that `packaging-error` is an important area for improvement, with recorded recall of approximately 0.379. This suggests that a substantial proportion of labelled instances of this class were missed.

The test set contains only three labelled `loose-meat` instances and no labelled `wrinkle` instances. Consequently, performance values for rare classes should be interpreted cautiously because a small number of detections can substantially affect the reported metrics.

Confidence scores and dashboard detection counts are not substitutes for evaluation against ground-truth annotations.

## Project Structure

A typical repository layout for this project is shown below. Some entries are optional and should only be retained in this tree if they exist in the actual repository.

```text
.
├── app.py                    # Streamlit browser application
├── main.py                   # Optional batch inference script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git exclusions
├── .env.example              # Example configuration, without real secrets
├── test_app.py               # Automated application tests, if at project root
├── tests/
│   └── test_app.py           # Automated tests, if stored in this folder
├── test_images/              # Optional local test images
├── results/                  # Evaluation outputs and generated results
├── notebooks/                # Optional training and analysis notebooks
└── docs/                     # HTML and additional documentation
```

The exact layout depends on the version of the repository. Do not keep both `test_app.py` and `tests/test_app.py` unless both files are genuinely needed. Likewise, use the test command appropriate to the actual location of the test file.

The Streamlit interface accepts image uploads directly, so the `test_images/` folder is not required to use the web application. Avoid committing API keys, unnecessary copies of the dataset, large model weights or generated outputs unless they are required and permitted.

## Requirements

The hosted inference application requires compatible versions of:

- Python
- Streamlit
- Roboflow Inference SDK (`inference-sdk`)
- Pillow
- pandas

The application also uses libraries required by its visualisation and reporting functionality, such as Matplotlib where applicable.

YOLOv8 training and evaluation require additional packages, including Ultralytics and compatible PyTorch dependencies.

Use the project's `requirements.txt` as the primary installation reference. Ensure the versions are compatible with the Python interpreter and operating system being used.

Hosted inference also requires an active internet connection, a valid Roboflow API key and access to the configured workflow.

## Installation

Open Terminal and navigate to the project root directory.

Activate the Python environment used for the project. For example, if using Conda and an environment named `mma_3001`:

```bash
conda activate mma_3001
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Check that Python and pip refer to the same environment:

```bash
python --version
python -m pip --version
```

If a dependency is missing, install it into the active environment rather than a different system-wide Python installation.

The project has been worked on in a macOS environment. YOLO training dependencies, particularly PyTorch and torchvision, must have compatible versions. If package installation fails, check the compatibility of the Python version, PyTorch version, torchvision version and Ultralytics version before changing packages.

## Configuration and API Key

A valid Roboflow API key and access to the deployed workflow are required for hosted inference.

### Recommended configuration: environment variable

On macOS or Linux, set the API key in the same Terminal session used to start Streamlit:

```bash
export ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
```

Replace the placeholder with your own valid key. Do not include the actual key in this README.

Then start the application from that Terminal session.

For a deployment platform such as Streamlit Community Cloud, configure the key using the platform's secrets settings.

### API security

- Never commit a real API key to GitHub.
- Do not include keys in screenshots, notebooks, logs or documentation.
- Do not place a real key in `.env.example`.
- If a key has been exposed, revoke or rotate it.
- If the application source code contains a hardcoded real key, remove it and use secure configuration before pushing the repository.

The application must read the environment variable using the configuration approach implemented in `app.py`. Setting an environment variable will not resolve the error if the code is still expecting a different variable name or a hardcoded value.

### Roboflow workflow configuration

The workflow identifiers documented during development are:

- **Workspace:** `ece4078h3`
- **Workflow:** `pork-rasher-errorpackaging-vpork-rasher-error-packaging-rj5da-1-rfdetr-small-t1-logic`

Verify these identifiers against the deployed Roboflow workflow if the workflow has changed. Keep credentials separate from the workflow identifiers.

## Running the Streamlit Application

From the project root, run:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

Streamlit normally prints a local address, such as:

```text
http://localhost:8501
```

Open the displayed address in a browser if it does not open automatically.

**Do not start the web application using `python app.py`.** Streamlit needs to launch the script through its own runtime.

### Using the application

1. Activate the correct Python environment.
2. Configure `ROBOFLOW_API_KEY` in the same Terminal session.
3. Start the application using Streamlit.
4. Set the minimum confidence threshold. The recorded initial default is 0.50 (50%); this is a starting value, not a validated optimum.
5. Upload one or more supported images: JPG, JPEG, PNG, BMP or WEBP.
6. Select **Run defect detection**.
7. Wait for the images to be processed by the hosted Roboflow workflow.
8. Review the summary, detections by class and individual image results.
9. Expand an image result to compare the original image with annotated predictions, confidence scores and inspection status.
10. Select **Download CSV report** to save the prediction table.
11. Select **Download annotated images (ZIP)** to save the annotated images.

The time required to process each image can vary with network conditions and hosted-service response time.

### Inspection statuses

The application uses the following status meanings:

- `DEFECT_DETECTED`: At least one valid prediction passed the selected confidence threshold.
- `NO_DEFECT_DETECTED`: No valid prediction passed the threshold. This does not guarantee that the image is defect-free.
- `INFERENCE_ERROR`: The image could not be processed. Review the error; a failed inspection must not be treated as a defect-free result.

The CSV report includes image names, predicted classes, confidence values, bounding-box coordinates, the selected threshold, status, inference time and error information where available. Images with no qualifying detections should also be represented in the report so that inspection outcomes are not silently omitted.

## Batch Inference

If `main.py` is present in the repository and configured for batch processing, place supported images in `test_images/` and run:

```bash
python main.py
```

The output paths depend on the actual implementation of the script. Previously documented output locations include:

```text
results/
├── predictions.csv
└── annotated_images/
```

Confirm the output paths in `main.py` before relying on them.

Batch inference and the Streamlit application are separate entry points. The web interface accepts uploads directly and does not require the `test_images/` folder.

## Automated Testing

Automated tests were added to check application functionality independently of manually running the full hosted inference workflow.

The recorded local test run reported:

```text
Ran 10 tests in 0.013s

OK
```

**Recorded result: 10 tests passed, with 0 failures.**

This result refers to the test run reported during project development. Re-run the tests against the current version of the repository after making further code changes.

### Running the tests

If the test file is located at the project root:

```bash
python test_app.py
```

If the test file is located inside a `tests/` directory and uses Python's `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

If the repository uses pytest and the required dependencies are installed:

```bash
python -m pytest -v
```

Use the command matching the actual test file and framework. These commands are alternatives, not a requirement to run all three.

### Interpreting Streamlit warnings during tests

During the recorded direct test run, Streamlit emitted warnings such as:

```text
missing ScriptRunContext
```

These warnings occurred because Streamlit-related code was imported or executed outside a normal Streamlit application runtime.

They do not automatically indicate that a unit test has failed. The test runner's final result should be checked separately. However, passing unit tests do not by themselves prove that the complete application, hosted API integration or user interface works correctly in a browser.

### What should be tested

The testing process should cover the application's core behaviour, including:

- Handling supported image inputs.
- Processing valid prediction data.
- Applying the confidence threshold.
- Handling empty prediction results.
- Constructing inspection statuses.
- Preparing result data for reporting.
- Handling invalid responses or inference errors.

The exact coverage depends on the assertions implemented in the current test file. Do not claim that a particular behaviour is tested unless the test code actually checks it.

### Recommended manual integration test

After running the automated tests, launch the Streamlit application and check the complete workflow:

1. Upload a valid image.
2. Run inference with a valid API key.
3. Confirm that predictions and annotated images display correctly.
4. Check the confidence threshold behaviour.
5. Download and inspect the CSV report.
6. Download and inspect the annotated-images ZIP.
7. Test the handling of an image that produces no qualifying detections.
8. Check that API or network errors are clearly reported.

This integration check complements the automated tests; it does not replace them.

## HTML Source-Code Documentation

The project includes HTML documentation for the Python source code. The `pdoc` tool can generate or refresh documentation from `app.py`.

From the project root, install `pdoc` if it is not already available:

```bash
python -m pip install pdoc
```

Generate the HTML documentation:

```bash
python -m pdoc app.py -o docs
```

This generates documentation in the `docs/` directory from the module's documented functions, classes and docstrings.

To locate the generated HTML files:

```bash
find docs -name "*.html"
```

If the generated file is `docs/app.html`, open it on macOS with:

```bash
open docs/app.html
```

For useful source-code documentation, add clear docstrings to functions and classes in `app.py`. Re-run the `pdoc` command whenever the source code changes.

**Important:** Documentation generated by `pdoc` is different from a manually written static file such as `docs/index.html`. If the repository already contains a static HTML page, check the generated output and ensure that the submitted `docs/` directory contains the intended documentation. Do not assume that a static page has been refreshed simply because `pdoc` was run.

## Reproducibility

For each training or evaluation experiment, record:

- Dataset name, version and train/validation/test splits.
- Model architecture and pretrained weights.
- Training parameters, including image size, batch size and number of epochs.
- Augmentation settings.
- Confidence and IoU thresholds used during evaluation.
- Evaluation script and metric implementation.
- Class names, class mapping and bounding-box conventions.
- Hardware, operating system, Python and relevant software versions.
- Overall and per-class metrics.
- Annotation corrections and dataset changes.
- Timing methodology, including whether network and API latency are included.
- Test results and any known limitations.

For a fair comparison, evaluate RF-DETR and YOLOv8n on the same labelled held-out test images using compatible evaluation methods. Keep the test set separate from model tuning and confidence-threshold selection.

If the dataset or annotations change, record the new version and avoid comparing results as though the evaluation conditions were unchanged.

## Troubleshooting

### 1. `missing ScriptRunContext` warnings

If the application was started with:

```bash
python app.py
```

stop it with `Ctrl+C` and launch it through Streamlit:

```bash
python -m streamlit run app.py
```

If the warning appears during a direct unit-test run, inspect the final test result. Streamlit can emit context warnings when its functionality is accessed outside the normal app runtime. Do not assume that the warning is harmless in every situation; verify that the actual application runs correctly.

### 2. `ROBOFLOW_API_KEY` not found

Confirm that:

- The environment variable is set.
- The key is valid.
- The key has permission to access the configured workflow.
- The application reads the same environment variable name.
- Streamlit is started from the Terminal session in which the variable was set.

You can check whether the variable exists without printing the secret:

```bash
if [ -n "$ROBOFLOW_API_KEY" ]; then
  echo "ROBOFLOW_API_KEY is set"
else
  echo "ROBOFLOW_API_KEY is not set"
fi
```

Do not print the actual key in logs or screenshots.

### 3. `FileNotFoundError` for `data.yaml`

A YOLO training or evaluation script may fail if the configured dataset path points to a location that does not exist.

For example, the previously encountered configuration referred to:

```text
/Users/sameersimon/Downloads/Pork Rasher Error-Packaging-.v1i.yolov8/data.yaml
```

If the dataset has moved or was extracted into a different folder, the script will not be able to find that file.

On macOS, locate the actual YAML file with:

```bash
find "/Users/sameersimon" -name "data.yaml" 2>/dev/null
```

If the search returns several files, identify the one belonging to the intended Pork Rasher dataset version. Update the path used by the training or evaluation script to match that file.

Check that the dataset YAML file exists and that its training, validation and test paths point to the correct directories. A valid YAML file can still fail if the image directories it references are missing.

Avoid assuming that a previous absolute path remains valid after moving the project or dataset.

### 4. PyArrow or Streamlit error involving the `confidence` column

A reported error during dataframe display was:

```text
pyarrow.lib.ArrowInvalid: Could not convert '' with type str:
tried to convert to double; Conversion failed for column confidence
with type object
```

This can occur when the `confidence` column contains a mixture of numeric values and empty strings. Streamlit may use PyArrow to convert the dataframe for display, and the mixed data types can prevent conversion.

Convert the column to numeric values before passing the dataframe to Streamlit:

```python
df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce"
)

st.dataframe(df)
```

This assumes that `pandas` is imported as `pd` and that `df` is the dataframe being displayed.

With `errors="coerce"`, invalid or empty values become `NaN` rather than remaining strings. If the column should always contain a valid confidence score, also investigate why missing or empty values are being produced. Do not replace missing confidence scores with arbitrary numeric values.

If other columns contain mixed types, inspect those columns as well.

### 5. Workflow response format error

If predictions are missing or the response cannot be parsed, inspect the output schema returned by the Roboflow workflow.

Ensure that the response-parsing logic in `app.py` matches the actual response structure, including the prediction fields, class labels, confidence values and bounding-box coordinates.

A change to the workflow output may require a corresponding change to the application.

### 6. Bounding boxes do not align with the image

Check the workflow's coordinate convention and any image resizing or cropping steps.

Confirm whether the returned bounding boxes use the original uploaded image dimensions or the dimensions of a resized image. If the coordinate systems differ, transform the coordinates before drawing the boxes.

Also verify that the width, height and coordinate conventions used for the predictions match those expected by the visualisation code.

### 7. API or network errors

Check:

- Internet connectivity.
- API-key validity and permissions.
- Workspace and workflow identifiers.
- Roboflow service availability.
- The returned error message.
- Whether the application is handling failed requests correctly.

A failed inference must remain marked as `INFERENCE_ERROR`. It must not be silently treated as a successful inspection with no detected defects.

### 8. Python or dependency compatibility errors

If imports fail or package installation produces dependency conflicts, confirm that the intended Python environment is active.

Run:

```bash
python --version
python -m pip --version
python -m pip check
```

Use `python -m pip` to install packages into the active interpreter's environment. For YOLO training, check compatibility between Python, PyTorch, torchvision and Ultralytics before upgrading or downgrading packages.

## Limitations

The project has several limitations that should be considered when interpreting the results:

- **Class imbalance:** Performance can vary across defect classes because their training examples are unevenly represented.
- **Limited rare-class evaluation:** The held-out test split contains no labelled `wrinkle` instances, so performance for that class cannot be measured from the reported results.
- **Evaluation differences:** The RF-DETR and YOLOv8n metrics were produced using different evaluation procedures. A consistent evaluation is required before making a definitive model ranking.
- **Potential missed detections:** A model may fail to detect a real defect, particularly when confidence thresholds are high or a class is underrepresented.
- **False positives:** A predicted defect may not correspond to a genuine defect.
- **Confidence interpretation:** A confidence score does not guarantee that a prediction is correct.
- **Hosted inference dependency:** The RF-DETR workflow depends on network connectivity, API access, hosted-service availability and service latency.
- **Timing comparability:** Hosted inference time and local model inference time are measured through different execution pathways.
- **Bounding-box alignment:** Prediction visualisation must be checked against the original image dimensions and coordinate conventions.
- **Prototype status:** The system has not been certified for production food-safety inspection or validated as a replacement for a complete industrial quality-control process.

The application should therefore be treated as a prototype for automated defect-detection assistance. Further data collection, consistent evaluation, error analysis and real-world validation would be needed before production use.

## Academic Context

Developed as part of **MMA3001 — Numerical Methods and Machine Learning**, applying computer vision and object detection to automated packaging-defect detection.

## Acknowledgements

The project uses:

- **Roboflow** for dataset management and hosted inference.
- **RF-DETR** through the configured hosted workflow.
- **Ultralytics** for YOLOv8n training and evaluation.
- **Streamlit** for the browser-based application interface.
- **pdoc** for generating HTML source-code documentation.

Add the exact dataset and software documentation references required by the project's academic referencing style before submission.
