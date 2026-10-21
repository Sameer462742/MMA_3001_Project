**# Automated Pork Rasher Packaging Defect Detection**



Computer-vision project for detecting and localising packaging defects in pork rasher product images using object detection.



**## Project Overview**



This project investigates computer-vision methods for identifying packaging and product-presentation defects in pork rasher images. The system processes an image and returns predicted defect classes, confidence scores and bounding boxes.



The project uses two detection approaches:



\- **\*\*RF-DETR:\*\*** a Roboflow-hosted workflow used by the deployed inference application.

\- **\*\*YOLOv8n:\*\*** a locally trained model used for baseline and augmentation experiments.



A Streamlit application provides a browser-based interface for uploading images, reviewing annotated predictions and exporting inspection results. The project also includes model evaluation and comparison work.



\> **\*\*Important:\*\*** This is a prototype decision-support tool, not a certified food-safety inspection system. \`NO_DEFECT_DETECTED\` means that no prediction passed the selected confidence threshold; it does not prove that the image is defect-free.



**## Defect Classes**



The dataset defines five classes:



\- \`loose-meat\`

\- \`packaging-error\`

\- \`twisted-meat\`

\- \`unsealed\`

\- \`wrinkle\`



Class representation is imbalanced. In the training split, \`packaging-error\` has substantially more labelled instances than the other classes, while \`wrinkle\` has very few. The held-out test split used in the reported evaluation contains no labelled \`wrinkle\` instances, so performance for that class cannot be measured from this test split.



**## Features**



\- Hosted inference through a Roboflow workflow using RF-DETR.

\- Object detection with predicted class labels, confidence scores and bounding boxes.

\- Streamlit interface for uploading one or multiple images.

\- Side-by-side review of original images and annotated predictions.

\- Inspection statuses for detected defects, no qualifying detections and inference errors.

\- Summary counts and detections-by-class visualisation.

\- CSV export of predictions and inspection statuses.

\- ZIP download of annotated images.

\- YOLOv8n training and evaluation experiments.

\- Overall and per-class performance analysis.



**## Example Input and Output**



The application accepts an image file through the Streamlit interface. After **\*\*Run defect detection\*\*** is selected, the image is sent to the hosted workflow. The application displays the returned predictions, including class labels, confidence scores and bounding boxes where available.



For project demonstrations and reports, use an actual example consisting of:



1\. **\*\*Input:\*\*** the original pork rasher image without annotations.

2\. **\*\*Output:\*\*** the same image with the model's predicted boxes, labels and confidence scores.

3\. **\*\*Application view (optional):\*\*** a screenshot showing the upload, result status and CSV/ZIP export controls.



A single example demonstrates the application's operation but does not establish model accuracy. Accuracy must be measured against labelled ground-truth annotations.



**## Model Evaluation**



The project reports precision, recall, F1-score, mAP\@50 and mAP\@50–95. The reported test evaluation uses the Roboflow Pork Rasher Error Packaging dataset, Version 1, with 76 images in the held-out test split and 91 labelled object instances across four represented classes.



**### Reported results**



\| Metric | YOLOv8n corrected augmentation | RF-DETR workflow |

\|---|---:|---:|

\| Precision | 0.5204 | 0.7733 |

\| Recall | 0.8009 | 0.6374 |

\| F1-score | 0.6309\* | 0.6988 |

\| mAP\@50 | 0.5987 | 0.5487 |

\| mAP\@50–95 | 0.2102 | 0.2363 |

\| Mean time per image | 33.43 ms (reported) | 703.90 ms |



\\\*The YOLOv8n F1-score is calculated from its reported precision and recall. RF-DETR metrics were produced by a custom evaluator, while YOLOv8n metrics were produced by Ultralytics evaluation. These results should be treated as **\*\*provisional rather than a definitive head-to-head benchmark\*\*** until both models are checked using the same labelled test images, class mapping, bounding-box conventions and compatible metric calculations.



The timing measurements also cover different execution pathways: RF-DETR timing includes network and hosted-service overhead, while the YOLOv8n value is reported as local model timing. They should not be interpreted as a controlled hardware-only speed comparison.



The RF-DETR per-class results indicate that \`packaging-error\` is a key weakness, with reported recall of 0.3793. This means many labelled instances of that class were missed in the evaluation. The test set contains only three \`loose-meat\` instances and no \`wrinkle\` instances, so results for rare classes should be interpreted cautiously.



**### Metric definitions**



\- **\*\*Precision:\*\*** proportion of predicted detections that match ground truth.

\- **\*\*Recall:\*\*** proportion of labelled ground-truth objects that are detected.

\- **\*\*F1-score:\*\*** harmonic mean of precision and recall.

\- **\*\*mAP\@50:\*\*** mean average precision at an intersection-over-union (IoU) threshold of 0.50.

\- **\*\*mAP\@50–95:\*\*** mean average precision averaged across IoU thresholds from 0.50 to 0.95.



Confidence scores and dashboard detection counts are not substitutes for ground-truth evaluation.



**## Dataset**



The reported dataset is the Roboflow Pork Rasher Error Packaging dataset, Version 1 (V1). The recorded split contains:



\- Training: 1,764 images

\- Validation: 115 images

\- Test: 76 images

\- Total: 1,955 images



The training class distribution is strongly imbalanced. Dataset versions, split definitions and annotation changes should be documented whenever models are retrained. The test split used for the reported results has no labelled \`wrinkle\` instances.



**\*\*Dataset citation:\*\*** Add the exact Roboflow dataset URL and access date required by your referencing style before submission.



**## Project Structure**



A typical repository layout is:



\`\`\`text

.

├── app.py                    # Streamlit browser application

├── main.py                   # Batch inference script, if included

├── requirements.txt          # Python dependencies

├── README.md

├── .gitignore

├── .env.example              # Example configuration only; no real API key

├── test_images/              # Optional images for batch inference

├── results/                  # Evaluation outputs and generated results

├── notebooks/                # Training and analysis notebooks, if included

└── docs/                     # Additional documentation, if included

\`\`\`



This is an illustrative layout. Keep only paths that actually exist in the repository. The Streamlit application accepts image uploads directly, so \`test_images/\` is not required to use the web interface. Avoid committing API keys, large datasets, model weights or generated outputs unless they are required and permitted.



**## Requirements**



The Streamlit application requires compatible versions of:



\- Python

\- Streamlit

\- Roboflow Inference SDK (\`inference-sdk\`)

\- Pillow

\- pandas



YOLOv8 training and evaluation may require additional packages, such as Ultralytics and its compatible PyTorch dependencies. Use the project's \`requirements.txt\` as the installation source of truth and ensure its package versions are compatible with your Python environment.



An active internet connection and access to the deployed Roboflow workflow are required for hosted inference.



**## Installation**



Open Terminal in the project directory and activate the Python environment used for the project. Install the dependencies:



\`\`\`bash

python -m pip install -r requirements.txt

\`\`\`



If the environment is not yet configured, create and activate an environment using your preferred Python environment manager, then install the requirements. Check that the installed package versions are supported by your Python version.



**## Configuration and API Key**



A valid Roboflow API key and access to the deployed workflow are required for hosted inference.



**### Recommended: environment variable**



On macOS or Linux, set the API key in the same Terminal session used to start the application:



\`\`\`bash

export ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"

\`\`\`



Then launch the app from that Terminal session. For deployment platforms such as Streamlit Community Cloud, configure the key through the platform's secrets settings.



Do not commit a real API key to GitHub or include it in screenshots, notebooks, logs or this README. If a key has been exposed, revoke or rotate it. If the application currently hardcodes a real key, remove it from the source code and switch to a secure configuration method before pushing the repository.



The currently documented workflow configuration is:



\- **\*\*Workspace:\*\*** \`ece4078h3\`

\- **\*\*Workflow:\*\*** \`pork-rasher-errorpackaging-vpork-rasher-error-packaging-rj5da-1-rfdetr-small-t1-logic\`



Verify these identifiers against the deployed Roboflow workflow if it is changed.



**## Running the Streamlit Application**



From the project directory, run:



\`\`\`bash

python -m streamlit run app.py

\`\`\`



Alternatively:



\`\`\`bash

streamlit run app.py

\`\`\`



Streamlit normally prints a local address such as:



\`\`\`text

http\://localhost:8501

\`\`\`



Open the displayed address in a browser if it does not open automatically. Do not start the application with \`python app.py\`.



**### Using the application**



1\. Start the app.

2\. Set the minimum confidence threshold. The initial default is 0.50 (50%); this is a starting value, not a validated optimum.

3\. Upload one or more supported images: JPG, JPEG, PNG, BMP or WEBP.

4\. Select **\*\*Run defect detection\*\***.

5\. Wait while each image is sent to the Roboflow workflow.

6\. Review the summary, detections by class and individual image results.

7\. Expand an image result to compare the original image with annotated predictions, confidence scores and inspection status.

8\. Select **\*\*Download CSV report\*\*** to save the prediction table.

9\. Select **\*\*Download annotated images (ZIP)\*\*** to save the annotated images.



**### Inspection statuses**



\- \`DEFECT_DETECTED\`: at least one valid prediction passed the selected confidence threshold.

\- \`NO_DEFECT_DETECTED\`: no valid prediction passed the threshold. This does not guarantee that the image is defect-free.

\- \`INFERENCE_ERROR\`: the image could not be processed. Review the error; a failed inspection must not be treated as defect-free.



The CSV report includes image names, predicted classes, confidence values, bounding-box coordinates, threshold, status, inference time and error information where available. Images with no detections are included in the report.



**## Batch Inference with \`main.py\` (if included)**



If \`main.py\` is present in the repository and configured for batch processing, place supported images in \`test_images/\` and run:



\`\`\`bash

python main.py

\`\`\`



The script's output paths depend on its implementation and configuration. If the current script uses the previously documented paths, outputs are written to:



\- \`results/predictions.csv\`

\- \`results/annotated_images/\`



Confirm these paths in \`main.py\` before relying on them. Batch inference and the Streamlit app are separate entry points; the app accepts uploads directly and does not require \`test_images/\`.



**## Reproducibility**



Record the following for each experiment:



\- Dataset name, version and train/validation/test splits.

\- Model architecture and pretrained weights.

\- Training parameters and augmentation settings.

\- Confidence and IoU thresholds used for evaluation.

\- Evaluation script, metric implementation and class mapping.

\- Hardware, operating system and software versions.

\- Overall and per-class metrics.

\- Annotation corrections and any dataset changes.

\- Timing procedure, including whether measurements include network/API latency.



For a fair model comparison, evaluate RF-DETR and YOLOv8n on the same labelled held-out test images, use compatible metric calculations and keep the test set separate from model tuning and confidence-threshold selection.




## HTML Source-Code Documentation

The project includes HTML documentation for the Python source code. To generate or refresh the documentation from `app.py` using **pdoc**, run these commands from the project root:

```bash
python -m pip install pdoc
python -m pdoc app.py -o docs
```

This generates HTML documentation in the `docs/` directory from the module's documented functions, classes and docstrings. Open the generated HTML file in a browser; to find the generated filename, run:

```bash
find docs -name "*.html"
```

For example, if the generated file is `docs/app.html`, open it on macOS with:

```bash
open docs/app.html
```

For useful API documentation, add clear docstrings to functions and classes in `app.py`. Re-run the `pdoc` command whenever the source code changes, then include the refreshed `docs/` output in the repository.

**Note:** The `pdoc` command generates documentation from the source code; it is different from a manually written static `docs/index.html`. If `docs/index.html` already exists, check the output generated by `pdoc` and ensure the submitted `docs/` folder contains the intended version of the documentation.


**## Troubleshooting**



**### \`missing ScriptRunContext\` warnings**



The app may have been started with \`python app.py\`. Stop it with \`Ctrl+C\` and run:



\`\`\`bash

python -m streamlit run app.py

\`\`\`



**### API key not found**



Confirm that \`ROBOFLOW_API_KEY\` is set in the same Terminal session used to start Streamlit, and that the key has permission to access the configured workflow.



**### Workflow response format error**



Inspect the Roboflow workflow output schema and ensure the response-parsing logic in \`app.py\` matches the returned prediction fields.



**### Bounding boxes do not align with the image**



Check the workflow's coordinate convention and any image resizing or cropping steps. Transform coordinates if the returned boxes do not use the original uploaded image dimensions.



**### API or network errors**



Check internet connectivity, API-key permissions, the workspace and workflow identifiers, and hosted-service availability. Failed inspections should remain marked as \`INFERENCE_ERROR\`.



**## Limitations**



\- Performance may vary between classes because of class imbalance, limited examples and annotation quality.

\- The test split contains no labelled \`wrinkle\` instances, so that class cannot be evaluated using the reported test results.

\- The RF-DETR and YOLOv8n metrics were generated by different evaluation procedures and require a consistent evaluation before a definitive ranking can be claimed.

\- A confidence score does not guarantee that a prediction is correct.

\- No detections above the selected threshold do not certify that a product is defect-free.

\- Hosted inference depends on internet connectivity, API access, workflow availability and service latency.

\- Bounding-box visualisation must be checked against the original image dimensions.

\- The prototype has not been certified for production food-safety inspection or validated on a live production line.



**## Academic Context**



Developed as part of **\*\*MMA3001 — Numerical Methods and Machine Learning\*\***, applying computer vision to automated packaging-defect detection.



**## Acknowledgements**



The project uses Roboflow for dataset management and hosted inference, and the Ultralytics framework for YOLOv8 experiments. Add the exact dataset and software documentation references required by the project's referencing style.
