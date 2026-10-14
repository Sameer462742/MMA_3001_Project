"""
Pork Rasher Defect Detection — Streamlit application.

This app sends uploaded images to a Roboflow-hosted workflow, displays the
predicted defects and confidence scores, and lets the user download a CSV
report and annotated images.

Before running:
    1. Install dependencies:
       pip install streamlit inference-sdk pillow pandas

    2. Paste your NEW Roboflow API key into API_KEY near the top of this file.
       Do not commit a real key to GitHub.

    3. Start the application from this folder:
           streamlit run app.py

Security:
    A hard-coded key is suitable only for local testing. Keep it out of version control.
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

# Paste your NEW Roboflow API key here for local testing only.
# Do not commit this file with a real key to GitHub.
API_KEY = "siB1OGhXN0kGj0mPwzwM"

WORKSPACE = "ece4078h3"
WORKFLOW_ID = (
    "pork-rasher-errorpackaging-vpork-rasher-"
    "error-packaging-rj5da-1-rfdetr-small-t1-logic"
)
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """Return the locally configured Roboflow API key.

    For local testing, paste a newly generated key into the API_KEY constant
    near the top of this file. Never commit a real key to a public repository.

    Returns:
        The configured API key.

    Raises:
        ValueError: If the placeholder has not been replaced.
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

    Args:
        api_key: A valid Roboflow API key.

    Returns:
        A configured InferenceHTTPClient.
    """
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    )
    return client.configure(
        InferenceConfiguration(api_key_transport="header")
    )


def extract_predictions(response: Any) -> list[dict[str, Any]]:
    """Extract a list of predictions from common workflow response shapes.

    Args:
        response: Raw object returned by ``client.run_workflow``.

    Returns:
        A list of prediction dictionaries. An empty list means no predictions
        were returned; it does not necessarily prove that an image is defect-free.

    Raises:
        ValueError: If the response does not contain a recognised prediction list.
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
    """Validate and normalise one centre-based bounding-box prediction.

    Expected coordinates are x/y centre, width, and height, with confidence
    between 0 and 1. Invalid entries are ignored rather than crashing the batch.

    Args:
        prediction: Raw prediction from the workflow.

    Returns:
        A normalised dictionary, or None if required fields are invalid.
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
    """Draw qualifying bounding boxes and labels on a copy of an image.

    Args:
        image: Original image.
        predictions: Normalised predictions.
        threshold: Minimum confidence required for a prediction to be shown.

    Returns:
        A tuple containing the annotated image and qualifying predictions.

    Note:
        This assumes the workflow's coordinates refer to the original uploaded
        image dimensions. Confirm this visually for your specific workflow.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    accepted: list[dict[str, Any]] = []

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

        x1 = max(0, min(x - width / 2, image_width - 1))
        y1 = max(0, min(y - height / 2, image_height - 1))
        x2 = max(0, min(x + width / 2, image_width - 1))
        y2 = max(0, min(y + height / 2, image_height - 1))

        # Ignore boxes that collapse after clipping.
        if x2 <= x1 or y2 <= y1:
            continue

        label_text = f'{prediction["class"]}: {prediction["confidence"]:.1%}'
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

        accepted.append(prediction)

    return annotated, accepted


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes for display and downloads."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_inference(
    client: InferenceHTTPClient,
    uploaded_file: Any,
) -> tuple[list[dict[str, Any]], float]:
    """Run the hosted workflow for one uploaded image.

    Args:
        client: Configured Roboflow inference client.
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Normalised valid predictions and elapsed inference time in seconds.

    Raises:
        ValueError: If the image or workflow response is invalid.
        Exception: Network/API errors are passed to the caller for display.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")

    # The SDK accepts an image path. Use a temporary file and remove it after
    # inference, rather than requiring users to manually save uploads.
    uploaded_bytes = uploaded_file.getvalue()
    if not uploaded_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(uploaded_bytes)) as check_image:
            check_image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("This file is not a readable image.") from error

    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
        temporary_file.write(uploaded_bytes)
        temporary_file.flush()

        response = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temporary_file.name},
            use_cache=False,
        )

    elapsed = time.perf_counter() - start
    raw_predictions = extract_predictions(response)

    normalised = []
    for raw_prediction in raw_predictions:
        if isinstance(raw_prediction, dict):
            prediction = validate_prediction(raw_prediction)
            if prediction is not None:
                normalised.append(prediction)

    return normalised, elapsed


def make_zip(annotated_images: dict[str, bytes]) -> bytes:
    """Package annotated PNG images into an in-memory ZIP archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in annotated_images.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pork Rasher Defect Inspection",
    page_icon="🔎",
    layout="wide",
)

st.title("Pork Rasher Defect Inspection")
st.write(
    "Upload images of pork rasher packaging to run the hosted defect-detection "
    "workflow. Review the annotated images, confidence scores and downloadable report."
)
st.caption(
    "This is a decision-support prototype. A result with no displayed detections "
    "does not guarantee that the product is defect-free."
)

with st.sidebar:
    st.header("Inspection settings")
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
    st.subheader("Workflow")
    st.code(WORKFLOW_ID, language=None)
    st.caption("API credentials are read from an environment variable or Streamlit secrets.")

uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
    accept_multiple_files=True,
    help="Supported formats: JPG, JPEG, PNG, BMP and WEBP.",
)

run_button = st.button(
    "Run defect detection",
    type="primary",
    disabled=not uploaded_files,
)

if run_button and uploaded_files:
    try:
        api_key = get_api_key()
        client = create_client(api_key)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    result_rows: list[dict[str, Any]] = []
    annotated_bytes: dict[str, bytes] = {}
    display_results: list[dict[str, Any]] = []

    progress = st.progress(0, text="Starting inspection...")
    status_area = st.empty()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status_area.info(
            f"Processing image {index} of {len(uploaded_files)}: "
            f"{uploaded_file.name}"
        )

        try:
            # Keep a fresh stream position for repeated reads by Streamlit/PIL.
            original_bytes = uploaded_file.getvalue()
            with Image.open(io.BytesIO(original_bytes)) as opened_image:
                original_image = opened_image.convert("RGB")

            predictions, elapsed = run_inference(client, uploaded_file)
            annotated_image, accepted = annotate_image(
                original_image, predictions, confidence_threshold
            )

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
            annotated_bytes[f"{Path(uploaded_file.name).stem}_annotated.png"] = (
                image_to_png_bytes(annotated_image)
            )

            if accepted:
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
            # An API or parsing error must not be reported as "no defect".
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

        progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)}")

    status_area.empty()
    progress.empty()

    results_df = pd.DataFrame(result_rows)
    st.session_state["inspection_results"] = display_results
    st.session_state["inspection_df"] = results_df
    st.session_state["inspection_images_zip"] = make_zip(annotated_bytes)
    st.session_state["inspection_threshold"] = confidence_threshold

# Show the most recent completed run.
if "inspection_results" in st.session_state:
    results = st.session_state["inspection_results"]
    results_df = st.session_state["inspection_df"]

    total_images = len(results)
    failed_count = sum(item["status"] == "INFERENCE_ERROR" for item in results)
    defect_count = sum(item["status"] == "DEFECT_DETECTED" for item in results)
    no_detection_count = sum(item["status"] == "NO_DEFECT_DETECTED" for item in results)
    total_detections = sum(len(item["detections"]) for item in results)

    st.divider()
    st.header("Inspection summary")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Images submitted", total_images)
    metric_columns[1].metric("Images with detections", defect_count)
    metric_columns[2].metric("Total detections", total_detections)
    metric_columns[3].metric("Inference errors", failed_count)

    st.caption(
        f"No detections above threshold: {no_detection_count} image(s). "
        "These are not automatically certified as defect-free."
    )

    if failed_count:
        st.warning(
            f"{failed_count} image(s) could not be inspected. Review their error "
            "messages below; do not treat them as defect-free."
        )

    st.subheader("Detection counts by class")
    class_rows = []
    for item in results:
        for prediction in item["detections"]:
            class_rows.append({
                "Class": prediction["class"],
                "Confidence": prediction["confidence"],
                "Image": item["name"],
            })
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

    st.subheader("Image results")
    for item in results:
        with st.expander(f'{item["name"]} — {item["status"]}', expanded=False):
            if item["status"] == "INFERENCE_ERROR":
                st.error(f'Inspection failed: {item["error"]}')
                continue

            left, right = st.columns(2)
            with left:
                st.image(item["original"], caption="Original image", use_container_width=True)
            with right:
                st.image(item["annotated"], caption="Annotated predictions", use_container_width=True)

            st.write(f'**Status:** `{item["status"]}`')
            st.write(f'**Inference time:** {item["elapsed"]:.2f} seconds')

            if item["detections"]:
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

    st.subheader("Download reports")
    download_columns = st.columns(2)
    with download_columns[0]:
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV report",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_columns[1]:
        st.download_button(
            "Download annotated images (ZIP)",
            data=st.session_state["inspection_images_zip"],
            file_name="annotated_images.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("View raw CSV data"):
        st.dataframe(results_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Engineering note: prediction confidence is not the same as verified accuracy. "
    "Evaluate the model against labelled ground-truth images before using it for "
    "quality-control decisions."
)
