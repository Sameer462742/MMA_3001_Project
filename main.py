
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

# Paste your NEW Roboflow API key here
API_KEY = "siB1OGhXN0kGj0mPwzwM"

WORKSPACE = "ece4078h3"

WORKFLOW_ID = (
    "pork-rasher-errorpackaging-vpork-rasher-"
    "error-packaging-rj5da-1-rfdetr-small-t1-logic"
)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "results"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Only display predictions at or above this confidence
CONFIDENCE_THRESHOLD = 0.50

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# --------------------------------------------------
# 2. VALIDATE CONFIGURATION
# --------------------------------------------------

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

image_paths = sorted(
    path for path in INPUT_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in SUPPORTED_EXTENSIONS
)

if not image_paths:
    raise FileNotFoundError(
        f"No supported images found in {INPUT_DIR}"
    )

print(f"Found {len(image_paths)} images.")

# --------------------------------------------------
# 5. PROCESS IMAGES
# --------------------------------------------------

csv_path = OUTPUT_DIR / "predictions.csv"

total_detections = 0
images_with_detections = 0
failed_images = 0

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

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

    for index, image_path in enumerate(
        image_paths, start=1
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            # Send image to the hosted workflow
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": str(image_path)},
                use_cache=True
            )

            predictions = (
                result[0]["predictions"]["predictions"]
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            draw = ImageDraw.Draw(image)
            image_width, image_height = image.size

            valid_detections = []

            for pred in predictions:

                confidence = float(
                    pred["confidence"]
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                label = pred["class"]

                # Predictions are centre-based coordinates
                # in the workflow's prediction image space
                x_center = float(pred["x"])
                y_center = float(pred["y"])
                box_width = float(pred["width"])
                box_height = float(pred["height"])

                x1 = x_center - box_width / 2
                y1 = y_center - box_height / 2
                x2 = x_center + box_width / 2
                y2 = y_center + box_height / 2

                # Keep coordinates within the image bounds
                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width - 1))
                y2 = max(0, min(y2, image_height - 1))

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )

                draw.text(
                    (x1, max(0, y1 - 18)),
                    f"{label}: {confidence:.1%}",
                    fill="red",
                    stroke_width=1,
                    stroke_fill="white"
                )

                valid_detections.append(pred)

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

            # Save annotated image
            output_path = (
                ANNOTATED_DIR
                / f"{image_path.stem}_result.jpg"
            )

            image.save(output_path)

            total_detections += len(valid_detections)

            if valid_detections:
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

            # Display the first six results
            if index <= 6:
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.title(image_path.name)
                plt.axis("off")
                plt.show()

        except Exception as error:
            failed_images += 1

            print(
                f"Error processing {image_path.name}: "
                f"{error}"
            )

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
