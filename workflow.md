# Project Workflow

This document describes the reproducible research workflow implemented by this repository. It is not a clinical-use protocol.

## Workflow overview

```text
Place source archives outside the repository
  -> audit archive contents, targets, and file mappings
  -> generate fixed-seed stratified manifests
  -> extract features and fit preprocessing on training rows
  -> train candidate models
  -> select settings on validation macro-F1
  -> evaluate once on the reserved test partition
  -> run development-only cross-validation diagnostics
  -> save metrics, models, and reports
  -> optionally launch the Streamlit demonstration
```

## Reproduction

Run from the project root in PowerShell. Set the data directory to the location of the supplied archives:

```powershell
python -m pip install -r requirements.txt
$env:ASD_DATA_DIR = 'E:\College\7th Semester\Project\Data Set'
$env:PYTHONPATH = '.'
python -m src.data.audit_datasets
python -m src.train_experiments
python -m unittest discover -s tests -v
```

Launch the optional interface after model artifacts have been generated:

```powershell
streamlit run app.py
```

## Stages

### 1. Data and provenance audit

The source ZIP files remain outside the repository. The audit records archive contents, schemas, label distributions, missingness, duplicates, and modality mapping results. It also documents unresolved provenance and licensing information.

### 2. Target and mapping checks

For Track A, the audit verifies that the label matches the threshold rule computed from A1-A10. The total score and target are excluded from learned-model inputs. For Track B, audio, motion, and physiology files are linked using exact sample IDs. Image paths do not resolve to indexed samples and are excluded.

### 3. Splitting

Seed 42 creates stratified train/validation/test partitions saved in `data/manifests/`. The nominal split is 70/15/15. Observed counts are 738/158/158 for Track A and 70/15/15 for Track B. Model selection uses the validation set; the test partition is reserved for final reporting.

### 4. Feature preparation and model selection

Track A uses ten binary questionnaire item codes. Track B creates compact audio, motion, and physiology summaries. Scaling parameters are fit on training rows only. Candidate logistic-regression regularization values and decision-tree depths are selected using validation macro-F1.

### 5. Fusion comparison

Track B compares modality-specific classifiers with:
- **Early fusion:** concatenate all three feature blocks before classification.
- **Late fusion:** average the class probabilities from the modality-specific branches with equal weights.

These are simple fusion baselines. There is no trainable attention or Transformer layer.

### 6. Evaluation and diagnostics

The selected settings are evaluated on the held-out test data using accuracy, balanced accuracy, and macro-F1. Track A also saves a Wilson confidence interval for accuracy and binary curve summaries. Two repeats of five-fold stratified cross-validation run on the development data (training plus validation) as a stability diagnostic; this is not another independent test set.

### 7. Outputs

Training writes model parameters, metrics, curves, coefficients, and an experiment registry under `artifacts/`. Audits write manifests and reports. The optional interface reads these artifacts to display results and limitations.

## Interpretation

Track A's perfect rule and logistic results reflect reconstruction of a deterministic questionnaire-derived label, not clinical accuracy. Track B has only 15 test samples and is provider-described as synthetic/augmented; its measurements are highly uncertain. Report the target, split, test sample count, and metric together, and do not imply generalization to real participants.
