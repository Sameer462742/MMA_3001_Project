"""
Batch inference script for pork rasher packaging defect detection.

This script sends images from the local test_images directory to a
Roboflow-hosted RF-DETR workflow. Predictions that meet the configured
confidence threshold are annotated on the corresponding images and
recorded in a CSV report.

The script also displays up to six annotated images and prints a summary
of the batch-processing results.

Output files:
    results/predictions.csv:
        CSV report containing detected objects, confidence scores,
        bounding-box coordinates and processing statuses.

    results/annotated_images/:
        Directory containing the annotated output images.

Requirements:
    - inference-sdk
    - Pillow
    - matplotlib

The script requires a valid Roboflow API key, access to the configured
workspace and workflow, and supported images in the test_images folder.

Note:
    This is a prototype defect-detection tool. A prediction above the
    confidence threshold does not guarantee that a defect is genuine,
    and an image with no qualifying predictions is not necessarily
    defect-free.
"""

from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
# Define the Roboflow connection details, project directories,
# confidence threshold and supported image formats.

API_KEY = "siB1OGhXN0kGj0mPwzwM"

WORKSPACE = "ece4078h3"

WORKFLOW_ID = (
    "pork-rasher-errorpackaging-vpork-rasher-"
    "error-packaging-rj5da-1-rfdetr-small-t1-logic"
)

# Resolve all project paths relative to this Python script so that
# the script does not depend on the terminal's current directory.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "results"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"

# Create the output directories if they do not already exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Only predictions with confidence greater than or equal to this
# threshold are retained, drawn on images and written as detections.
CONFIDENCE_THRESHOLD = 0.50

# File extensions accepted by the batch-processing script.
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# --------------------------------------------------
# 2. VALIDATE CONFIGURATION
# --------------------------------------------------
# Check that the API key has been configured and that the input
# directory exists before attempting to connect to Roboflow.

if (
    not API_KEY
    or API_KEY == "PASTE_YOUR_NEW_API_KEY_HERE"
):
    raise RuntimeError(
        "Replace API_KEY with your new Roboflow API key."
    )

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Test image folder not found: {INPUT_DIR}\n"
        "Create a test_images folder and add images."
    )

# --------------------------------------------------
# 3. CONNECT TO ROBOFLOW
# --------------------------------------------------
# Initialise the inference client using the configured workspace
# credentials. The API key is configured for header-based transport.

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

print("Connected to Roboflow.")
print(f"Input folder: {INPUT_DIR}")

# --------------------------------------------------
# 4. FIND TEST IMAGES
# --------------------------------------------------
# Collect supported image files directly inside test_images.
# Sorting the paths makes the processing order predictable.

image_paths = sorted(
    path for path in INPUT_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in SUPPORTED_EXTENSIONS
)

# Stop if the input folder does not contain any supported images.
if not image_paths:
    raise FileNotFoundError(
        f"No supported images found in {INPUT_DIR}"
    )

print(f"Found {len(image_paths)} images.")

# --------------------------------------------------
# 5. PROCESS IMAGES
# --------------------------------------------------
# Process each image individually. Successful predictions are
# filtered by confidence, visualised, saved and written to the CSV.
# Exceptions are caught per image so that a failed image does not
# stop the processing loop for the remaining files.

csv_path = OUTPUT_DIR / "predictions.csv"

# Initialise counters used to summarise the completed batch.
total_detections = 0
images_with_detections = 0
failed_images = 0

# Open the CSV report in write mode, replacing any previous report.
# UTF-8 encoding supports a wide range of image filenames and labels.
with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    # Define the columns recorded for every detection or image status.
    fieldnames = [
        "image",
        "class",
        "confidence",
        "x_center",
        "y_center",
        "box_width",
        "box_height",
        "status"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )
    writer.writeheader()

    # Enumerate the input images starting at 1 for readable progress
    # messages in the terminal.
    for index, image_path in enumerate(
        image_paths, start=1
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            # Submit the current image to the configured Roboflow
            # workflow. The returned result contains the predictions.
            # Cache use is enabled through the SDK configuration here.
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": str(image_path)},
                use_cache=True
            )

            # Extract the list of object predictions from the workflow
            # response using the expected response structure.
            predictions = (
                result[0]["predictions"]["predictions"]
            )

            # Load the original image and convert it to RGB so that
            # the annotation and output operations use a consistent
            # image mode.
            image = Image.open(
                image_path
            ).convert("RGB")

            draw = ImageDraw.Draw(image)
            image_width, image_height = image.size

            # Store predictions that pass the confidence threshold.
            valid_detections = []

            # Examine each prediction returned by the workflow.
            for pred in predictions:

                # Convert the prediction confidence to a numeric value.
                confidence = float(
                    pred["confidence"]
                )

                # Ignore predictions below the configured threshold.
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Retrieve the predicted defect label.
                label = pred["class"]

                # Read the bounding-box centre coordinates and dimensions.
                # These values are expected to use the workflow's
                # prediction image coordinate space.
                x_center = float(pred["x"])
                y_center = float(pred["y"])
                box_width = float(pred["width"])
                box_height = float(pred["height"])

                # Convert the centre-based bounding box to corner
                # coordinates for drawing with Pillow.
                x1 = x_center - box_width / 2
                y1 = y_center - box_height / 2
                x2 = x_center + box_width / 2
                y2 = y_center + box_height / 2

                # Restrict each corner coordinate to the image bounds
                # to prevent the rectangle from extending beyond the
                # visible image area.
                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width - 1))
                y2 = max(0, min(y2, image_height - 1))

                # Draw the bounding box around the predicted defect.
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )

                # Display the predicted class and confidence near
                # the top-left corner of the bounding box.
                draw.text(
                    (x1, max(0, y1 - 18)),
                    f"{label}: {confidence:.1%}",
                    fill="red",
                    stroke_width=1,
                    stroke_fill="white"
                )

                # Retain the prediction for detection counts and
                # terminal reporting.
                valid_detections.append(pred)

                # Record this qualifying detection in the CSV report,
                # including its centre coordinates and box dimensions.
                writer.writerow({
                    "image": image_path.name,
                    "class": label,
                    "confidence": round(confidence, 4),
                    "x_center": x_center,
                    "y_center": y_center,
                    "box_width": box_width,
                    "box_height": box_height,
                    "status": "detection"
                })

            # Save the annotated image as a JPEG in the output folder.
            # The original input image is not overwritten.
            output_path = (
                ANNOTATED_DIR
                / f"{image_path.stem}_result.jpg"
            )

            image.save(output_path)

            # Update the total number of qualifying detections.
            total_detections += len(valid_detections)

            if valid_detections:
                # Count images containing at least one qualifying
                # detection and display the detected classes.
                images_with_detections += 1

                print(
                    f"Detections: {len(valid_detections)}"
                )

                for pred in valid_detections:
                    print(
                        f"  {pred['class']}: "
                        f"{float(pred['confidence']):.1%}"
                    )
            else:
                # Record an explicit status when no prediction passes
                # the threshold. This is not proof that the image has
                # no actual defects.
                writer.writerow({
                    "image": image_path.name,
                    "class": "",
                    "confidence": "",
                    "x_center": "",
                    "y_center": "",
                    "box_width": "",
                    "box_height": "",
                    "status": "no_detection_above_threshold"
                })

                print("No detections above threshold.")

            print(f"Saved: {output_path}")

            # Display the first six annotated results for a quick
            # visual review. Remaining images are still processed
            # and saved, but are not displayed by this condition.
            if index <= 6:
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.title(image_path.name)
                plt.axis("off")
                plt.show()

        except Exception as error:
            # Count the current image as failed and report the error.
            # Processing then continues with the next image.
            failed_images += 1

            print(
                f"Error processing {image_path.name}: "
                f"{error}"
            )

            # Include the failed image in the CSV so that its outcome
            # is recorded rather than silently omitted.
            writer.writerow({
                "image": image_path.name,
                "class": "",
                "confidence": "",
                "x_center": "",
                "y_center": "",
                "box_width": "",
                "box_height": "",
                "status": f"error: {error}"
            })

# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------
# Calculate the number of images that completed without raising an
# exception and print the main batch-processing statistics and paths.

processed = len(image_paths) - failed_images

print("\n" + "=" * 50)
print("BATCH TEST SUMMARY")
print("=" * 50)

print(f"Images found:           {len(image_paths)}")
print(f"Images processed:       {processed}")
print(f"Images with detections: {images_with_detections}")
print(
    "Images without detections: "
    f"{processed - images_with_detections}"
)
print(f"Total detections:       {total_detections}")
print(f"Failed images:          {failed_images}")
print(
    f"Confidence threshold:   "
    f"{CONFIDENCE_THRESHOLD:.0%}"
)
print(f"CSV report:             {csv_path}")
print(f"Annotated images:       {ANNOTATED_DIR}")