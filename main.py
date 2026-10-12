
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Roboflow connection
client = InferenceHTTPClient(
     api_url="https://serverless.roboflow.com",
     api_key="siB1OGhXN0kGj0mPwzwM"
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

# Input image
image_path = (
    "/Users/sameersimon/Local Documents/"
    "MMA_3001_Project/"
    "frame_with_red_102_jpg.rf."
    "8659eac39d6481b97dd7cd9cd5b53d10.jpg"
)

# Run the deployed workflow
result = client.run_workflow(
    workspace_name="ece4078h3",
    workflow_id=(
        "pork-rasher-errorpackaging-vpork-rasher-"
        "error-packaging-rj5da-1-rfdetr-small-t1-logic"
    ),
    images={"image": image_path},
    use_cache=True
)

# Load image and prepare drawing
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Workflow output
predictions = result[0]["predictions"]["predictions"]

for pred in predictions:
    x_center = pred["x"]
    y_center = pred["y"]
    width = pred["width"]
    height = pred["height"]

    # Convert centre coordinates to box corners
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    label = pred["class"]
    confidence = pred["confidence"]

    # Draw bounding box and label
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=4
    )

    draw.text(
        (x1, max(0, y1 - 20)),
        f"{label}: {confidence:.1%}",
        fill="red",
        stroke_width=1,
        stroke_fill="white"
    )

    print(
        f"Detected: {label} "
        f"(confidence: {confidence:.2%})"
    )

# Save and display
output_path = "prediction_result.jpg"
image.save(output_path)

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Roboflow Defect Detection")
plt.show()

print(f"Saved annotated image to: {output_path}")
