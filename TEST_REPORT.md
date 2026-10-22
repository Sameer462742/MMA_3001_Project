# Automated Test Report — Pork Rasher Defect Detection

## Scope
The unit tests in `tests/test_app.py` cover:
- Rejecting the default placeholder API key and trimming a configured key.
- Parsing top-level and nested prediction lists.
- Rejecting an unsupported workflow response shape.
- Validating a prediction and rejecting malformed input.
- Serialising a PIL image to PNG bytes.
- Creating ZIP archives with image files, including an empty archive.

## Test framework
Python `unittest` tests are compatible with `pytest` discovery. The tests are
designed to run locally and do not call the hosted Roboflow API.

## Execution record
- Command attempted: `python -m pytest -q`
- Outcome: **Blocked during test collection; tests did not run.**
- Environment issue: `ModuleNotFoundError: No module named 'streamlit'` while importing `app.py`.
- Interpretation: This is a missing dependency in the execution environment, not a test assertion failure. No passing test result can be claimed from this attempt.

## How to run
From the project root, install the project's dependencies and run:

```bash
python -m pip install -r requirements.txt
python -m pytest -v
```

Alternatively:

```bash
python -m unittest discover -s tests -v
```

After running successfully in the target environment, update this report with
the date, Python version, number of tests run, and pass/fail results.

## Limitations
- The hosted Roboflow service and API credentials are not tested.
- Model accuracy is not evaluated by these unit tests.
- These tests do not perform an end-to-end browser test of the Streamlit UI.
- Passing unit tests alone does not establish production or food-safety suitability.
