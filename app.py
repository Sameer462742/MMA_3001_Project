"""
Pork Rasher Defect Detection — Streamlit Application.

This application provides a browser-based interface for detecting potential
packaging and product-presentation defects in pork rasher images.

Uploaded images are sent to a Roboflow-hosted inference workflow. The returned
predictions are validated, filtered using a configurable confidence threshold,
and displayed with annotated bounding boxes and confidence scores.

The application also provides:
    - Support for uploading multiple images in one inspection run.
    - Image validation and prediction-response handling.
    - Inspection statuses for detected defects, no qualifying detections,
      and inference errors.
    - An inspection summary and detections-by-class chart.
    - CSV export of predictions and inspection statuses.
    - ZIP download of annotated images.
    - Inference-time reporting for individual images.

Requirements:
    - Python
    - Streamlit
    - Roboflow Inference SDK (inference-sdk)
    - Pillow
    - pandas

Before running:
    1. Install the dependencies required by the project.
    2. Configure a valid Roboflow API key.
    3. Verify that the configured workspace and workflow are accessible.
    4. Start the application using:
           streamlit run app.py

Security:
    Never publish a real API key in source code or commit it to version control.
    Use an environment variable or Streamlit secrets for credential management.

Limitations:
    This is a prototype decision-support tool. A prediction that passes the
    confidence threshold is not necessarily correct, and an image with no
    qualifying detections is not guaranteed to be defect-free.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# ---------------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION
# ---------------------------------------------------------------------------
# Store the Roboflow connection details, supported image formats and default
# confidence threshold used by the inspection interface.

# Paste your NEW Roboflow API key here for local testing only.
# Do not commit this file with a real key to GitHub.
API_KEY = "siB1OGhXN0kGj0mPwzwM"

# Identify the Roboflow workspace and the workflow used for inference.
WORKSPACE = "ece4078h3"
WORKFLOW_ID = (
    "pork-rasher-errorpackaging-vpork-rasher-"
    "error-packaging-rj5da-1-rfdetr-small-t1-logic"
)

# Base URL for the hosted Roboflow inference service.
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

# Supported file extensions for uploaded images.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default minimum confidence score for displaying a detection.
# This is an initial setting and should be validated using labelled data.
DEFAULT_CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
# These functions handle API configuration, prediction parsing and validation,
# image annotation, inference execution and downloadable output preparation.


def get_api_key() -> str:
    """Return the configured Roboflow API key.

    The key is read from the API_KEY constant defined near the top of this
    file. A placeholder or empty value is rejected before inference begins.

    Returns:
        str: The configured API key after surrounding whitespace is removed.

    Raises:
        ValueError: If the API key is empty or the placeholder has not been
            replaced with a valid configuration value.
    """
    api_key = API_KEY.strip()
    if not api_key or api_key == "PASTE_YOUR_NEW_ROBOFLOW_API_KEY_HERE":
        raise ValueError(
            "Set API_KEY near the top of app.py to your NEW Roboflow API key."
        )
    return api_key


@st.cache_resource(show_spinner=False)
def create_client(api_key: str) -> InferenceHTTPClient:
    """Create and cache a configured Roboflow inference client.

    Streamlit caches the client resource so that repeated application reruns
    can reuse the configured client instead of constructing a new one each time.

    Args:
        api_key: A valid Roboflow API key.

    Returns:
        InferenceHTTPClient: A configured client connected to the hosted
        Roboflow inference service.
    """
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    )
    return client.configure(
        InferenceConfiguration(api_key_transport="header")
    )


def extract_predictions(response: Any) -> list[dict[str, Any]]:
    """Extract the prediction list from a Roboflow workflow response.

    The function supports common response formats in which the response is
    either a list containing a dictionary or a dictionary returned directly.
    Within the payload, predictions may be nested inside another dictionary
    or provided directly as a list.

    An unrecognised response structure raises an error rather than silently
    being interpreted as an image with no detected defects.

    Args:
        response: The raw object returned by ``client.run_workflow``.

    Returns:
        A list of prediction dictionaries. An empty list indicates that no
        predictions were returned; it does not necessarily prove that an
        image is defect-free.

    Raises:
        ValueError: If the response is not a recognised dictionary structure
            or does not contain a supported prediction list.
    """
    # Workflow responses are often a list containing a dictionary. Accept
    # either that shape or a dictionary directly to make parsing less brittle.
    payload = response[0] if isinstance(response, list) and response else response

    if not isinstance(payload, dict):
        raise ValueError("Unexpected workflow response: expected a dictionary.")

    predictions_obj = payload.get("predictions")

    # Common object-detection response:
    # {"predictions": {"predictions": [...]}}
    if isinstance(predictions_obj, dict):
        predictions = predictions_obj.get("predictions")
        if isinstance(predictions, list):
            return predictions

    # Also accept a direct list:
    # {"predictions": [...]}
    if isinstance(predictions_obj, list):
        return predictions_obj

    # Some workflows use a named output. Rather than silently treating an
    # unknown response as "no defects", raise a clear error for inspection.
    raise ValueError(
        "Could not find a prediction list in the workflow response. "
        "Check the workflow's output format."
    )


def validate_prediction(prediction: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalise a single object-detection prediction.

    The expected prediction contains a class label, confidence score and
    centre-based bounding-box coordinates. Invalid entries are rejected so
    that malformed data does not cause the annotation loop to fail.

    The returned dictionary uses consistent field names for later processing.

    Args:
        prediction: A raw prediction dictionary from the workflow.

    Returns:
        A normalised prediction dictionary containing:
            - class: Predicted defect label.
            - confidence: Numeric confidence score between 0 and 1.
            - x_center: Horizontal centre coordinate.
            - y_center: Vertical centre coordinate.
            - box_width: Bounding-box width.
            - box_height: Bounding-box height.

        Returns None if required fields are missing, cannot be converted to
        numeric values, contain missing numeric values, or fail the specified
        confidence and bounding-box dimension checks.
    """
    try:
        label = str(prediction["class"])
        confidence = float(prediction["confidence"])
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
    except (KeyError, TypeError, ValueError):
        return None

    if not label or not all(
        pd.notna(value) for value in (confidence, x, y, width, height)
    ):
        return None
    if not (0.0 <= confidence <= 1.0) or width <= 0 or height <= 0:
        return None

    return {
        "class": label,
        "confidence": confidence,
        "x_center": x,
        "y_center": y,
        "box_width": width,
        "box_height": height,
    }


def annotate_image(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    threshold: float,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Draw qualifying predictions on a copy of the supplied image.

    Predictions below the specified confidence threshold are skipped.
    For each accepted prediction, the function converts centre-based
    bounding-box coordinates into corner coordinates, clips the coordinates
    to the image boundaries, draws a rectangle and adds a class/confidence
    label.

    The original image object is not modified because annotation is performed
    on an RGB copy.

    Args:
        image: Original image to annotate.
        predictions: List of normalised prediction dictionaries.
        threshold: Minimum confidence required for a prediction to be drawn.

    Returns:
        A tuple containing:
            - The annotated RGB image.
            - The list of predictions that passed the threshold and produced
              valid visible bounding boxes.

    Note:
        This function assumes that the workflow's coordinates refer to the
        original uploaded image dimensions. Confirm this assumption against
        the configured workflow and inspect the resulting annotations.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    accepted: list[dict[str, Any]] = []

    # Load a default font for detection labels.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for prediction in predictions:
        if prediction["confidence"] < threshold:
            continue

        # Convert centre-based coordinates to corner coordinates.
        x = prediction["x_center"]
        y = prediction["y_center"]
        width = prediction["box_width"]
        height = prediction["box_height"]

        # Calculate the top-left and bottom-right corners, clipping each
        # coordinate so that the rectangle remains within the image bounds.
        x1 = max(0, min(x - width / 2, image_width - 1))
        y1 = max(0, min(y - height / 2, image_height - 1))
        x2 = max(0, min(x + width / 2, image_width - 1))
        y2 = max(0, min(y + height / 2, image_height - 1))

        # Ignore boxes that collapse after clipping.
        if x2 <= x1 or y2 <= y1:
            continue

        # Format the label to include the class name and confidence percentage.
        label_text = f'{prediction["class"]}: {prediction["confidence"]:.1%}'

        # Draw the detection box around the predicted defect.
        draw.rectangle((x1, y1, x2, y2), outline="#ff3030", width=3)

        # Add a filled label background so the text remains readable.
        text_box = draw.textbbox((x1, max(0, y1 - 18)), label_text, font=font)
        draw.rectangle(text_box, fill="#ff3030")
        draw.text(
            (x1, max(0, y1 - 18)),
            label_text,
            fill="white",
            font=font,
        )

        # Keep track of predictions that were actually drawn.
        accepted.append(prediction)

    return annotated, accepted


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes.

    An in-memory buffer is used so the image can be passed directly to
    Streamlit display and download functions without creating a separate
    image file on disk.

    Args:
        image: PIL image to encode.

    Returns:
        bytes: PNG-encoded image content.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_inference(
    client: InferenceHTTPClient,
    uploaded_file: Any,
) -> tuple[list[dict[str, Any]], float]:
    """Run hosted defect detection for one uploaded image.

    The function validates the file extension and image contents, writes the
    uploaded bytes to a temporary file for the inference SDK, runs the
    configured Roboflow workflow and measures the elapsed inference time.
    The workflow response is then parsed and its predictions are normalised.

    Args:
        client: Configured Roboflow inference client.
        uploaded_file: Streamlit UploadedFile object representing the image.

    Returns:
        A tuple containing:
            - A list of valid, normalised predictions.
            - Elapsed workflow execution time in seconds.

    Raises:
        ValueError: If the image extension is unsupported, the uploaded file
            is empty, the file is not a readable image, or the workflow
            response has an unrecognised structure.
        Exception: Network and API errors are allowed to propagate to the
            caller so that the interface can display an inference error.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")

    # The SDK accepts an image path. Use a temporary file and remove it after
    # inference, rather than requiring users to manually save uploads.
    uploaded_bytes = uploaded_file.getvalue()
    if not uploaded_bytes:
        raise ValueError("The uploaded file is empty.")

    # Verify that the uploaded bytes represent a readable image before
    # sending the file to the hosted inference workflow.
    try:
        with Image.open(io.BytesIO(uploaded_bytes)) as check_image:
            check_image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("This file is not a readable image.") from error

    # Start timing immediately before creating and submitting the temporary
    # image file. The elapsed time includes the hosted workflow request.
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
        temporary_file.write(uploaded_bytes)
        temporary_file.flush()

        # Submit the temporary image to the configured workflow.
        response = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temporary_file.name},
            use_cache=False,
        )

    elapsed = time.perf_counter() - start
    raw_predictions = extract_predictions(response)

    # Keep dictionary entries that can be validated and normalised.
    normalised = []
    for raw_prediction in raw_predictions:
        if isinstance(raw_prediction, dict):
            prediction = validate_prediction(raw_prediction)
            if prediction is not None:
                normalised.append(prediction)

    return normalised, elapsed


def make_zip(annotated_images: dict[str, bytes]) -> bytes:
    """Create an in-memory ZIP archive of annotated images.

    Each dictionary entry maps an output filename to the PNG-encoded image
    bytes that should be included in the archive.

    Args:
        annotated_images: Dictionary of image filenames and image contents.

    Returns:
        bytes: Complete ZIP archive that can be offered as a download.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in annotated_images.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------------------------
# Build the interactive browser interface, accept uploaded images, execute
# inference when requested, and present the results and downloadable reports.


# Configure the browser page title, icon and wide layout.
st.set_page_config(
    page_title="Pork Rasher Defect Inspection",
    page_icon="🔎",
    layout="wide",
)

# Display the application title and a brief description of its purpose.
st.title("Pork Rasher Defect Inspection")
st.write(
    "Upload images of pork rasher packaging to run the hosted defect-detection "
    "workflow. Review the annotated images, confidence scores and downloadable report."
)
st.caption(
    "This is a decision-support prototype. A result with no displayed detections "
    "does not guarantee that the product is defect-free."
)

# Place inspection settings and workflow information in the sidebar.
with st.sidebar:
    st.header("Inspection settings")

    # Allow the user to choose the minimum confidence score for displaying
    # predictions. The selected threshold is applied during annotation.
    confidence_threshold = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help=(
            "Predictions below this value are hidden. Choose a final threshold "
            "using validation data, not by visual preference alone."
        ),
    )

    st.write(f"Current threshold: **{confidence_threshold:.0%}**")
    st.divider()

    # Show the workflow used for hosted inference.
    st.subheader("Workflow")
    st.code(WORKFLOW_ID, language=None)
    st.caption("API credentials are read from an environment variable or Streamlit secrets.")

# Allow users to upload one or more images in supported formats.
uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
    accept_multiple_files=True,
    help="Supported formats: JPG, JPEG, PNG, BMP and WEBP.",
)

# Disable the inference button until at least one image has been uploaded.
run_button = st.button(
    "Run defect detection",
    type="primary",
    disabled=not uploaded_files,
)

# Start the inspection process only when the user clicks the button and
# has supplied at least one image.
if run_button and uploaded_files:
    # Retrieve the API key and create the cached inference client.
    # Stop this run and display a clear message if configuration is invalid.
    try:
        api_key = get_api_key()
        client = create_client(api_key)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    # These collections hold the report rows, downloadable annotated images
    # and information required to render each image's result in the interface.
    result_rows: list[dict[str, Any]] = []
    annotated_bytes: dict[str, bytes] = {}
    display_results: list[dict[str, Any]] = []

    # Provide progress feedback while images are processed one at a time.
    progress = st.progress(0, text="Starting inspection...")
    status_area = st.empty()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status_area.info(
            f"Processing image {index} of {len(uploaded_files)}: "
            f"{uploaded_file.name}"
        )

        try:
            # Retrieve the original upload bytes and decode the image as RGB
            # for consistent display and annotation.
            original_bytes = uploaded_file.getvalue()
            with Image.open(io.BytesIO(original_bytes)) as opened_image:
                original_image = opened_image.convert("RGB")

            # Run the hosted workflow and annotate predictions that pass
            # the user-selected confidence threshold.
            predictions, elapsed = run_inference(client, uploaded_file)
            annotated_image, accepted = annotate_image(
                original_image, predictions, confidence_threshold
            )

            # Determine the image status from the qualifying predictions.
            # No qualifying detections do not guarantee that no defects exist.
            status = "DEFECT_DETECTED" if accepted else "NO_DEFECT_DETECTED"
            display_results.append({
                "name": uploaded_file.name,
                "original": original_image,
                "annotated": annotated_image,
                "detections": accepted,
                "status": status,
                "elapsed": elapsed,
                "error": "",
            })

            # Convert the annotated image to PNG bytes for the ZIP download.
            annotated_bytes[f"{Path(uploaded_file.name).stem}_annotated.png"] = (
                image_to_png_bytes(annotated_image)
            )

            if accepted:
                # Add one CSV row per qualifying prediction. Each row includes
                # the image name, class, confidence, bounding-box data,
                # threshold, status and measured inference time.
                for prediction in accepted:
                    result_rows.append({
                        "image": uploaded_file.name,
                        "class": prediction["class"],
                        "confidence": round(prediction["confidence"], 4),
                        "x_center": prediction["x_center"],
                        "y_center": prediction["y_center"],
                        "box_width": prediction["box_width"],
                        "box_height": prediction["box_height"],
                        "threshold": confidence_threshold,
                        "status": status,
                        "inference_seconds": round(elapsed, 3),
                        "error": "",
                    })
            else:
                # Keep images with zero detections in the report too.
                result_rows.append({
                    "image": uploaded_file.name,
                    "class": "",
                    "confidence": "",
                    "x_center": "",
                    "y_center": "",
                    "box_width": "",
                    "box_height": "",
                    "threshold": confidence_threshold,
                    "status": status,
                    "inference_seconds": round(elapsed, 3),
                    "error": "",
                })

        except Exception as error:
            # Record the failure without interrupting the processing of
            # other uploaded images. An inference failure must not be
            # interpreted as a successful inspection with no defects.
            message = f"{type(error).__name__}: {error}"
            display_results.append({
                "name": uploaded_file.name,
                "original": None,
                "annotated": None,
                "detections": [],
                "status": "INFERENCE_ERROR",
                "elapsed": None,
                "error": message,
            })
            result_rows.append({
                "image": uploaded_file.name,
                "class": "",
                "confidence": "",
                "x_center": "",
                "y_center": "",
                "box_width": "",
                "box_height": "",
                "threshold": confidence_threshold,
                "status": "INFERENCE_ERROR",
                "inference_seconds": "",
                "error": message,
            })

        # Update progress after each uploaded image has been handled.
        progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)}")

    # Clear the temporary progress indicators after the run completes.
    status_area.empty()
    progress.empty()

    # Convert the report rows to a dataframe and retain the completed run
    # in Streamlit session state. This allows results to remain visible
    # when Streamlit reruns the script after a user interaction.
    results_df = pd.DataFrame(result_rows)
    st.session_state["inspection_results"] = display_results
    st.session_state["inspection_df"] = results_df
    st.session_state["inspection_images_zip"] = make_zip(annotated_bytes)
    st.session_state["inspection_threshold"] = confidence_threshold

# Show the most recent completed run, if one exists in session state.
if "inspection_results" in st.session_state:
    results = st.session_state["inspection_results"]
    results_df = st.session_state["inspection_df"]

    # Calculate summary statistics for the current inspection run.
    total_images = len(results)
    failed_count = sum(item["status"] == "INFERENCE_ERROR" for item in results)
    defect_count = sum(item["status"] == "DEFECT_DETECTED" for item in results)
    no_detection_count = sum(item["status"] == "NO_DEFECT_DETECTED" for item in results)
    total_detections = sum(len(item["detections"]) for item in results)

    st.divider()
    st.header("Inspection summary")

    # Display the key inspection statistics in four metric cards.
    metric_columns = st.columns(4)
    metric_columns[0].metric("Images submitted", total_images)
    metric_columns[1].metric("Images with detections", defect_count)
    metric_columns[2].metric("Total detections", total_detections)
    metric_columns[3].metric("Inference errors", failed_count)

    st.caption(
        f"No detections above threshold: {no_detection_count} image(s). "
        "These are not automatically certified as defect-free."
    )

    # Warn the user when any images could not be inspected successfully.
    if failed_count:
        st.warning(
            f"{failed_count} image(s) could not be inspected. Review their error "
            "messages below; do not treat them as defect-free."
        )

    # Build a table containing all accepted predictions for the class summary.
    st.subheader("Detection counts by class")
    class_rows = []
    for item in results:
        for prediction in item["detections"]:
            class_rows.append({
                "Class": prediction["class"],
                "Confidence": prediction["confidence"],
                "Image": item["name"],
            })

    # Aggregate detections by class and display a chart when predictions exist.
    if class_rows:
        class_df = pd.DataFrame(class_rows)
        class_counts = (
            class_df.groupby("Class")
            .size()
            .rename("Detections")
            .sort_values(ascending=False)
            .reset_index()
        )
        st.bar_chart(class_counts.set_index("Class"))
    else:
        st.info("No predictions passed the selected confidence threshold.")

    # Present the results for each image in an expandable section.
    st.subheader("Image results")
    for item in results:
        with st.expander(f'{item["name"]} — {item["status"]}', expanded=False):
            # Failed images have no valid annotated output, so display the
            # error message instead of attempting to render missing images.
            if item["status"] == "INFERENCE_ERROR":
                st.error(f'Inspection failed: {item["error"]}')
                continue

            # Display the original and annotated images side by side.
            left, right = st.columns(2)
            with left:
                st.image(item["original"], caption="Original image", use_container_width=True)
            with right:
                st.image(item["annotated"], caption="Annotated predictions", use_container_width=True)

            # Show the inspection outcome and elapsed inference time.
            st.write(f'**Status:** `{item["status"]}`')
            st.write(f'**Inference time:** {item["elapsed"]:.2f} seconds')

            if item["detections"]:
                # Display a compact table of the qualifying class labels
                # and confidence percentages for this image.
                detection_table = pd.DataFrame([
                    {
                        "Defect class": prediction["class"],
                        "Confidence": f'{prediction["confidence"]:.1%}',
                    }
                    for prediction in item["detections"]
                ])
                st.dataframe(detection_table, hide_index=True, use_container_width=True)
            else:
                st.info("No detections passed the confidence threshold.")

    # Provide download controls for the full CSV report and annotated images.
    st.subheader("Download reports")
    download_columns = st.columns(2)

    with download_columns[0]:
        # Convert the dataframe to UTF-8 CSV bytes for downloading.
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV report",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_columns[1]:
        # Offer the annotated PNG images as a compressed ZIP archive.
        st.download_button(
            "Download annotated images (ZIP)",
            data=st.session_state["inspection_images_zip"],
            file_name="annotated_images.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # Allow users to inspect the raw dataframe underlying the CSV export.
    with st.expander("View raw CSV data"):
        st.dataframe(results_df, use_container_width=True, hide_index=True)

# Display a final reminder about the difference between confidence and
# verified model performance.
st.divider()
st.caption(
    "Engineering note: prediction confidence is not the same as verified accuracy. "
    "Evaluate the model against labelled ground-truth images before using it for "
    "quality-control decisions."
)