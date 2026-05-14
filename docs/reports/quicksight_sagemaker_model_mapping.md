# QuickSight SageMaker Model Mapping Report

## 1. Executive Summary
The forensic audit of the `chucaw-sagemaker` repository confirms a successful transition from a legacy JSON-based model to a new CSV-based classifier designed for QuickSight data augmentation. Past QuickSight integration failures were definitively caused by targeting the legacy KNN model (`chucaw-t2m-trained-knn-model`) with CSV data, leading to a content-type mismatch (`Expected application/json`). The new model (`chucaw-quicksight-abrupt-classifier-v0`) is correctly implemented, packaged, and verified to handle headerless CSV inputs.

## 2. Evidence Table

| Evidence ID | Source | Observation | Implications |
| :--- | :--- | :--- | :--- |
| E-01 | `inference/inference.py` | Requires `application/json`; raises error on `text/csv`. | Confirms cause of legacy failures. |
| E-02 | `src/quicksight_abrupt_classifier/inference.py` | Implements `text/csv` logic; expects 3 columns (lat, lon, t2m). | Correct contract for QuickSight. |
| E-03 | `scripts/package_quicksight_abrupt_model.py` | Isolated packaging of `src/quicksight_abrupt_classifier`. | Artifact is clean of legacy code. |
| E-04 | `AWS list-transform-jobs` | Old job: `InputFilter = "$[0:1]"`. | Risk: QuickSight might filter columns incorrectly. |
| E-05 | `Local Smoke Test` | Produced correct 2-column CSV output. | Model logic is behaviorally correct. |

## 3. Repository Map

### Core Logic
- **New Classifier**: `src/quicksight_abrupt_classifier/inference.py`
- **Legacy KNN**: `inference/inference.py`

### Infrastructure & Deploy
- **Packaging**: `scripts/package_quicksight_abrupt_model.py`
- **Deployment**: `scripts/deploy_quicksight_abrupt_batch_model.py`
- **Smoke Test**: `scripts/run_quicksight_abrupt_batch_smoke.py`
- **Registry**: `scripts/register_quicksight_abrupt_model_package.py`

### Schemas
- **Primary (CSV)**: `schemas/quicksight_abrupt_classifier_schema.json`
- **Secondary (text/csv)**: `schemas/quicksight_abrupt_classifier_schema_textcsv.json`

## 4. Model Inventory

| Model Name | Purpose | Contract | Status |
| :--- | :--- | :--- | :--- |
| `chucaw-quicksight-abrupt-classifier-v0` | QuickSight Augmentation | CSV (3-in, 2-out) | **Healthy/Active** |
| `chucaw-t2m-trained-knn-model` | Legacy API Experiment | JSON (lat/lon in) | **Legacy/Obsolete** |

## 5. Contract Comparison

| Feature | Legacy JSON KNN | New CSV QuickSight Classifier |
| :--- | :--- | :--- |
| **Input Format** | JSON Object `{"lat": X, "lon": Y}` | CSV Row (headerless) `lat,lon,t2m` |
| **Output Format** | JSON Object `{"t2m": X, ...}` | CSV Row (headerless) `label,score` |
| **Content Type** | `application/json` | `text/csv`, `CSV`, `application/csv` |
| **Usage** | Real-time Endpoint | Batch Transform |
| **Columns** | 2 | 3 |

## 6. QuickSight Failure Reconstruction
The observed failure `Unsupported content type: text/csv. Expected application/json` occurred because:
1. QuickSight sent a CSV manifest/payload to SageMaker.
2. The `TransformJob` was configured to use `chucaw-t2m-trained-knn-model`.
3. The legacy `inference.py` (which expects JSON) rejected the `text/csv` payload in its `input_fn`.

## 7. DataProcessing/InputFilter Risk Analysis
**Critical Risk**: The old QuickSight job used `InputFilter = "$[0:1]"`. 
- **Legacy model** required 2 columns. 
- **New model** requires 3 columns.
If QuickSight's internal mapping for the new model is still based on an old or cached configuration that only sends 2 columns, the job will fail with:
`ValueError: Row 1: expected exactly 3 columns (lat, lon, t2m), got 2`.

**Mitigation**: Ensure the QuickSight "Predictive Field" configuration is refreshed and explicitly maps all 3 input fields (`lat`, `lon`, `t2m`) to dataset columns.

## 8. Remaining Uncertainties
- Whether QuickSight UI caches model metadata (schema/ARN) aggressively.
- Whether `inputContentType: "CSV"` vs `"text/csv"` in the schema file causes different UI behaviors in the QuickSight Sagemaker mapping screen.

## 9. Next Minimal Experiment
1. Go to QuickSight.
2. Remove the existing SageMaker augmentation from the dataset.
3. Re-add it using the `chucaw-quicksight-abrupt-classifier-v0` model.
4. Upload `schemas/quicksight_abrupt_classifier_schema_textcsv.json` if prompted for a schema.
5. Map exactly 3 fields: `latitude` -> `lat`, `longitude` -> `lon`, `value` -> `t2m`.
6. Run the transform and check SageMaker logs for the row count error if it fails.

## 10. Recommended Fixes
1. **[High]** Explicitly use 3-column mapping in QuickSight UI.
2. **[Medium]** If `InputFilter` errors occur, modify `inference.py` to be more lenient (e.g., skip validation of column count if it's > 3, or allow 2 columns with a default t2m). *Currently not recommended to keep the prototype strict.*
3. **[Low]** Decommission the legacy KNN model in SageMaker to prevent accidental selection.
