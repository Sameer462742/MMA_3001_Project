"""Unit tests for the Pork Rasher Defect Detection application.

These tests exercise local helper functions only. They do not call the
hosted Roboflow API, so they can run without network access or an API key.
"""
import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

# Import the app module from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app


class TestApiKeyConfiguration(unittest.TestCase):
    def test_placeholder_api_key_is_rejected(self):
        with patch.object(app, "API_KEY", "PASTE_YOUR_NEW_ROBOFLOW_API_KEY_HERE"):
            with self.assertRaisesRegex(ValueError, "Set API_KEY"):
                app.get_api_key()

    def test_configured_api_key_is_trimmed(self):
        with patch.object(app, "API_KEY", "  example-test-key  "):
            self.assertEqual(app.get_api_key(), "example-test-key")


class TestPredictionParsing(unittest.TestCase):
    def test_extracts_top_level_prediction_list(self):
        response = {"predictions": [{"class": "unsealed", "confidence": 0.9}]}
        self.assertEqual(app.extract_predictions(response), response["predictions"])

    def test_extracts_nested_prediction_list(self):
        response = {"predictions": {"predictions": [{"class": "wrinkle"}]}}
        self.assertEqual(app.extract_predictions(response), [{"class": "wrinkle"}])

    def test_rejects_unrecognised_response_shape(self):
        with self.assertRaises(ValueError):
            app.extract_predictions({"unexpected": []})

    def test_valid_prediction_is_normalised(self):
        prediction = {
            "class": "packaging-error",
            "confidence": 0.87,
            "x": 50,
            "y": 40,
            "width": 20,
            "height": 10,
        }
        result = app.validate_prediction(prediction)
        self.assertIsNotNone(result)
        self.assertEqual(result["class"], "packaging-error")
        self.assertAlmostEqual(float(result["confidence"]), 0.87)

    def test_invalid_prediction_is_rejected(self):
        result = app.validate_prediction({"class": "", "confidence": "not-a-number"})
        self.assertIsNone(result)


class TestImageAndExportHelpers(unittest.TestCase):
    def test_image_to_png_bytes_creates_valid_png(self):
        image = Image.new("RGB", (8, 6), "white")
        payload = app.image_to_png_bytes(image)
        reopened = Image.open(io.BytesIO(payload))
        self.assertEqual(reopened.format, "PNG")
        self.assertEqual(reopened.size, (8, 6))

    def test_make_zip_contains_annotated_images(self):
        image = Image.new("RGB", (4, 4), "red")
        image_bytes = app.image_to_png_bytes(image)
        archive_bytes = app.make_zip({"sample.png": image_bytes})

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertIn("sample.png", archive.namelist())
            self.assertEqual(archive.read("sample.png"), image_bytes)

    def test_make_zip_supports_empty_input(self):
        archive_bytes = app.make_zip({})
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(archive.namelist(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
