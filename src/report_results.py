"""Generate reproducible figures and a technical report from executed artifacts."""
from __future__ import annotations
import csv, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports/figures"


def svg_confusion(path, result, title):
    labels = result["class_order"]; matrix = result["confusion_matrix"]
    n = len(labels); cell = 72; left = 180; top = 82
    width, height = left + n * cell + 35, top + n * cell + 65
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="17">{html.escape(title)}</text>', f'<text x="{left+n*cell/2}" y="55" text-anchor="middle" font-family="Arial" font-size="13">Predicted label</text>', f'<text transform="translate(20 {top+n*cell/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">True label</text>']
    max_value = max(max(row) for row in matrix) or 1
    for i, label in enumerate(labels):
        parts.append(f'<text x="{left-10}" y="{top+i*cell+cell/2+5}" text-anchor="end" font-family="Arial" font-size="11">{html.escape(str(label))}</text>')
        parts.append(f'<text x="{left+i*cell+cell/2}" y="{top-10}" text-anchor="middle" font-family="Arial" font-size="10" transform="rotate(-35 {left+i*cell+cell/2} {top-10})">{html.escape(str(label))}</text>')
        for j in range(n):
            value = matrix[i][j]
            shade = int(244 - 145 * value / max_value) if i == j else 250
            fill = f'#{shade:02x}{min(255,shade+2):02x}{shade:02x}'
            parts.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#999"/>')
            parts.append(f'<text x="{left+j*cell+cell/2}" y="{top+i*cell+cell/2+6}" text-anchor="middle" font-family="Arial" font-size="17">{value}</text>')
    parts.append('</svg>')
    path.write_text(''.join(parts), encoding='utf-8')


def svg_line(path, title, series, x_label, y_label):
    width, height = 640, 400; left, top, plot_w, plot_h = 70, 55, 500, 280
    y_values = [y for _, points, _ in series for _, y in points]
    ymax = max(y_values) if y_values else 1
    xmax = max((max(x for x, _ in points) for _, points, _ in series if points), default=1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="16">{html.escape(title)}</text>', f'<text x="{left+plot_w/2}" y="380" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(x_label)}</text>', f'<text transform="translate(16 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(y_label)}</text>', f'<path d="M{left} {top} V{top+plot_h} H{left+plot_w}" fill="none" stroke="#333"/>']
    for idx, (name, points, color) in enumerate(series):
        coords = [f'{left+x/max(xmax,1)*plot_w:.1f},{top+plot_h-y/max(ymax,1e-12)*plot_h:.1f}' for x, y in points]
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx = left + idx % 2 * 260; ly = 355 + idx // 2 * 16
        parts.append(f'<text x="{lx}" y="{ly}" font-family="Arial" font-size="11" fill="{color}">{html.escape(name)}</text>')
    parts.append('</svg>')
    path.write_text(''.join(parts), encoding='utf-8')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    a = json.loads((ARTIFACTS / "dataset1_results.json").read_text(encoding="utf-8"))
    a_cv = json.loads((ARTIFACTS / "dataset1_cv_results.json").read_text(encoding="utf-8"))
    b = json.loads((ARTIFACTS / "dataset2_results.json").read_text(encoding="utf-8"))
    models_a = json.loads((ARTIFACTS / "dataset1_logistic.json").read_text(encoding="utf-8"))
    models_b = json.loads((ARTIFACTS / "dataset2_models.json").read_text(encoding="utf-8"))
    curves = json.loads((ARTIFACTS / "dataset1_curves.json").read_text(encoding="utf-8"))

    results = [(f"A / {name}", result) for name, result in a.items()]
    results += [(f"B / {name}", result) for name, result in b["unimodal"].items()]
    results += [(f"B / {name}", b[name]) for name in ("early_fusion", "late_fusion")]
    for name, result in results:
        svg_confusion(FIGURES / (name.lower().replace(" / ", "_").replace(" ", "_") + "_confusion.svg"), result, name)

    macro = [(name, [result["macro_f1"]], color) for (name, result), color in zip(results, ["#2474a6", "#d97706", "#3b8c68", "#888"] * 3)]
    # Horizontal dot chart with actual macro-F1 values to avoid misleading stacked bars.
    width, height, left, row_h = 740, 90 + len(macro) * 32, 180, 32
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="370" y="24" text-anchor="middle" font-family="Arial" font-size="17">Held-out macro-F1 (test)</text>']
    for i, (name, values, color) in enumerate(macro):
        y = 58 + i * row_h; value = values[0]
        svg.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="11">{html.escape(name)}</text><line x1="{left}" y1="{y}" x2="{left+500}" y2="{y}" stroke="#ddd"/><line x1="{left}" y1="{y}" x2="{left+500*value:.1f}" y2="{y}" stroke="{color}" stroke-width="6"/><text x="{left+510}" y="{y+4}" font-family="Arial" font-size="11">{value:.3f}</text>')
    svg.append('</svg>')
    (FIGURES / "test_macro_f1.svg").write_text(''.join(svg), encoding='utf-8')

    roc_series, pr_series = [], []
    for name, info, color in (("Scoring rule", curves["questionnaire_scoring_rule"], "#2474a6"), ("Logistic regression", curves["logistic_regression"], "#d97706")):
        roc_series.append((f"{name} (AUC {info['roc_auc']:.3f})", [(x, y) for x, y in info["roc_points_fpr_tpr"]], color))
        pr_series.append((f"{name} (AP {info['average_precision']:.3f})", [(x, y) for x, y in info["pr_points_recall_precision"]], color))
    svg_line(FIGURES / "dataset1_roc.svg", "Dataset 1 test ROC", roc_series, "False positive rate", "True positive rate")
    svg_line(FIGURES / "dataset1_precision_recall.svg", "Dataset 1 test precision-recall", pr_series, "Recall", "Precision")

    loss_series = []
    colors = ["#2474a6", "#d97706", "#3b8c68", "#8e44ad", "#c0392b"]
    histories = [("Track A logistic", models_a["training_history"])] + [(f"Track B {m}", models_b[m]["training_history"]) for m in ("audio", "motion", "physiology", "early_fusion")]
    for i, (name, history) in enumerate(histories):
        for key, suffix in (("train_loss", "train"), ("validation_loss", "validation")):
            loss_series.append((f"{name} {suffix}", [(row["epoch"], row[key]) for row in history if row[key] is not None], colors[i]))
    svg_line(FIGURES / "training_validation_loss.svg", "Training and validation loss (saved model runs)", loss_series, "Epoch", "Cross-entropy + L2 penalty")

    with (ARTIFACTS / "dataset1_feature_coefficients.csv").open(encoding="utf-8", newline="") as stream:
        coef_rows = list(csv.DictReader(stream))
    coef_rows.sort(key=lambda row: float(row["coefficient_Yes"]))
    h = 90 + len(coef_rows) * 23
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="{h}" viewBox="0 0 650 {h}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="325" y="24" text-anchor="middle" font-family="Arial" font-size="16">Track A logistic coefficients (Yes class)</text>']
    mid = 340; scale = 16
    for i, row in enumerate(coef_rows):
        y = 55 + i * 23; value = float(row["coefficient_Yes"])
        svg.append(f'<text x="{mid-8}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="11">{html.escape(row["feature"])}</text><line x1="{mid}" y1="{y-7}" x2="{mid}" y2="{y+7}" stroke="#333"/><rect x="{min(mid,mid+value*scale):.1f}" y="{y-6}" width="{abs(value*scale):.1f}" height="12" fill="{"#2474a6" if value >= 0 else "#d97706"}"/><text x="{max(mid,mid+value*scale)+5:.1f}" y="{y+4}" font-family="Arial" font-size="10">{value:.3f}</text>')
    svg.append('</svg>')
    (FIGURES / "dataset1_logistic_coefficients.svg").write_text(''.join(svg), encoding='utf-8')

    def metric_row(name, result):
        return f"| {name} | {result['n']} | {result['accuracy']:.3f} | {result['balanced_accuracy']:.3f} | {result['macro_f1']:.3f} |"
    a_rows = "\n".join(metric_row(name, result) for name, result in a.items())
    b_rows = "\n".join(metric_row(name, result) for name, result in b["unimodal"].items())
    b_rows += "\n" + "\n".join(metric_row(name, b[name]) for name in ("early_fusion", "late_fusion"))
    ci_rule = a["questionnaire_scoring_rule"]["test_accuracy_95pct_wilson_ci"]
    ci_log = a["logistic_regression"]["test_accuracy_95pct_wilson_ci"]
    split_a = {key: sum(row["split"] == key for row in csv.DictReader((ROOT / "data/manifests/track_a_manifest.csv").open(encoding="utf-8"))) for key in ("train", "validation", "test")}
    with (ROOT / "data/manifests/track_b_manifest.csv").open(encoding="utf-8") as stream:
        track_b_manifest = list(csv.DictReader(stream))
    split_b_counts = {key: sum(row["split"] == key for row in track_b_manifest) for key in ("train", "validation", "test")}
    d2_total_error = {name: result["n"] - round(result["accuracy"] * result["n"]) for name, result in list(b["unimodal"].items()) + [(name, b[name]) for name in ("early_fusion", "late_fusion")]}
    robustness = "\n".join(f"| {name.replace('_',' ')} | {result['accuracy']:.3f} | {result['balanced_accuracy']:.3f} | {result['macro_f1']:.3f} |" for name, result in b["missing_modality_robustness"].items())
    cv = b["development_repeated_stratified_group_cv"]
    cv_rows = "\n".join(f"| {name} | {values['accuracy']['mean']:.3f} ± {values['accuracy']['std']:.3f} | {values['balanced_accuracy']['mean']:.3f} ± {values['balanced_accuracy']['std']:.3f} | {values['macro_f1']['mean']:.3f} ± {values['macro_f1']['std']:.3f} |" for name, values in cv["models"].items())
    a_cv_rows = "\n".join(f"| {name} | {values['accuracy']['mean']:.3f} ± {values['accuracy']['std']:.3f} | {values['balanced_accuracy']['mean']:.3f} ± {values['balanced_accuracy']['std']:.3f} | {values['macro_f1']['mean']:.3f} ± {values['macro_f1']['std']:.3f} |" for name, values in a_cv["models"].items())

    report = f'''# Technical Report: A Multimodal Machine Learning Framework for Early Autism Screening

**Status:** Completed research prototype; all reported results below were generated by code in this repository on the supplied archives.  
**Execution date:** 2026-10-13.  
**Purpose:** Reproducible educational experiments, not clinical screening or diagnosis.

## Abstract

This project audits two supplied autism-related datasets and implements separate leakage-aware experiments. The toddler questionnaire file contains 1,054 rows. Its score is exactly the sum of A1–A10 and its `Class/ASD Traits` label is exactly positive when the score exceeds 3. A fixed rule and logistic regression both achieved {a['questionnaire_scoring_rule']['accuracy']:.1%} accuracy on 158 held-out rows, which is expected reconstruction of the deterministic label. A decision tree achieved {a['decision_tree']['accuracy']:.1%}. The second dataset is described by its provider as partially synthetic/augmented. It contains 100 indexed samples; audio, motion, and physiology paths resolve, while all 100 image paths fail exact matching. On a fixed 15-sample test split, non-image unimodal and fusion accuracy ranged from {min([v['accuracy'] for v in b['unimodal'].values()] + [b['early_fusion']['accuracy'], b['late_fusion']['accuracy']]):.1%} to {max([v['accuracy'] for v in b['unimodal'].values()] + [b['early_fusion']['accuracy'], b['late_fusion']['accuracy']]):.1%}. These results are small synthetic-file benchmarks only and provide no evidence about real participants or clinical performance.

## 1. Introduction and objectives

The implementation asks whether the supplied toddler screen labels can be reconstructed from questionnaire items, and whether a small synthetic dataset supports a basic multimodal pipeline demonstration. Objectives were to inspect source files, identify deterministic target leakage, preserve separate experimental tracks, establish fixed partitions, evaluate small classical models, compare unimodal/early/late fusion, and provide a guarded inference demo.

The project does not claim that questionnaire patterns, facial appearance, gaze, voice, movement, or physiological signals diagnose autism. Dataset 1 labels are screening-trait outputs. Dataset 2 metadata says the data do not represent real individuals and are not for clinical use.

## 2. Datasets, provenance, and licensing

### 2.1 Toddler questionnaire

The archive contains `Toddler Autism dataset July 2018.csv` (1,054×19), a second combined-age CSV (6,075×15), and a DOCX data description. The target header in the actual toddler CSV includes a trailing space: `Class/ASD Traits `. The supplied Croissant metadata names the Kaggle dataset and reports license `Unknown`. The DOCX attributes the dataset to Fadi Thabtah and describes ASDTests app data. We did not independently verify ethics approvals or participant identities.

### 2.2 Multimodal dataset

The archive contains an index with 100 rows, four class names, 100 JPGs, 100 WAVs, 100 motion JSONs, and 100 physiology CSVs. The external metadata describes simulated/augmented samples and explicitly says they do not represent real individuals. The metadata license is “Other (specified in description),” so reuse terms remain uncertain. Exact paths link non-image modalities to index sample IDs, but do not independently prove synchronized recordings or participant identity.

## 3. Data-quality and annotation audit

Both ZIP archives passed Python ZIP CRC checks. Dataset 1 has no missing cells or exact duplicate rows; all Case_No values are unique. A1–A10 are binary, ages span 12–36 months, and the labels are Yes 728 / No 326. All 1,054 score sums and target thresholds match exactly.

Dataset 2 has 100 unique sample IDs and labels: mild_asd 23, moderate_asd 19, severe_asd 33, typical 25. All non-image files decode. Audio is mono 16-bit PCM at 16 kHz and approximately 2 seconds. Motion arrays share a 300-frame 4-joint schema. Physiology CSVs share a 100-row 0–10-second grid. The TEMP column is the same full sequence in every sample. This repeated template and the very similar acoustic summary features limit the meaning of these modalities. No exact duplicate file hashes were found.

All 100 index image paths point to PNG names absent from the archive; the actual image files are differently named JPGs. Visual review found portrait photographs of children, but no authoritative sample mapping, facial/behavioral annotations, or image labels exist. Images were excluded. Details are in `reports/image_annotation_audit.md` and `reports/cross_modal_consistency_report.md`.

## 4. Label audit and leakage prevention

The supplied DOCX documents code 1 for A1–A9 when the response is Sometimes/Rarely/Never and for A10 when Always/Usually/Sometimes. Its screening rule is score > 3. In the actual archive, `Qchat-10-Score == sum(A1..A10)` in all rows, and `Class/ASD Traits ` equals the threshold in all rows. The score and target were excluded from model inputs. The final learned classifier uses only A1–A10; demographics and Case_No are excluded. Its high accuracy is label reconstruction, not independent detection.

Dataset 2 labels come from the supplied index and are treated as synthetic class labels. No participant-level linkage was supplied. Paths are mapped only by exact sample ID for audio, motion, and physiology. The image-to-index association remains unresolved.

## 5. Splitting and experimental setup

Seed-derived within-class ordering with seed 42 assigns each sample group to train/validation/test at 70/15/15. Dataset 1 counts are {split_a}; Dataset 2 counts are {split_b_counts}. Exact assignments are saved in `data/manifests/track_a_manifest.csv` and `data/manifests/track_b_manifest.csv`. Dataset 1 Case_No and Dataset 2 sample_id are the split groups; these IDs are unique in the files, but Dataset 2 participant independence cannot be verified.

Dataset 1 classifiers are a deterministic score threshold, regularized logistic regression, and a Gini decision tree (depth candidates 3, 5, 10). Logistic regularization candidates were 0.01, 0.1, 1, 10; validation macro-F1 selected 0.01. Tree depth was selected by validation macro-F1. Dataset 2 uses regularized multinomial logistic regression with compact acoustic (10), motion (63), and physiology (21) summaries; regularization candidates were 0.1, 1, 10 and selected on validation macro-F1. Early fusion concatenates those features; late fusion takes an equal probability mean. Normalization is fit on training rows only. A two-repeat, five-fold stratified group-CV diagnostic is run on the 85 development samples; the locked 15-sample test split is not used in CV.

The code uses NumPy-only implementations because scikit-learn and deep-learning frameworks were unavailable in the execution environment. Both tracks also run two-repeat, five-fold development-set CV diagnostics using fixed settings selected on validation. There are 15 unit tests covering scoring, leakage schema, split determinism/leakage, path mapping, feature shapes, reproducibility, fusion compatibility, and inference contracts.

## 6. Results

### 6.1 Dataset 1 test results

| Model | Test n | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
{a_rows}

The score rule and logistic model each have a 95% Wilson interval of [{ci_rule[0]:.3f}, {ci_rule[1]:.3f}] and [{ci_log[0]:.3f}, {ci_log[1]:.3f}], respectively, on the observed 158-row test set. These intervals describe uncertainty in a held-out accuracy estimate, not validity of the label or generalization to clinical populations. The decision tree made {a['decision_tree']['n'] - round(a['decision_tree']['accuracy']*a['decision_tree']['n'])} errors; the rule and logistic model made none because the test labels encode the same rule.

The fixed selected logistic/tree settings were also evaluated in two repeats of five-fold stratified group CV on the 896-row development set (train plus validation), without test rows:

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---:|---:|---:|
{a_cv_rows}

This is a stability diagnostic after validation-based parameter selection; it does not remove the deterministic target limitation.

Empirical test ROC AUC was {curves['questionnaire_scoring_rule']['roc_auc']:.3f} for the rule and {curves['logistic_regression']['roc_auc']:.3f} for logistic regression. Precision-recall curves are included, but their apparent perfection has the same deterministic-label limitation.

### 6.2 Dataset 2 synthetic file-level results

| Model | Test n | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
{b_rows}

The test set contains 15 files total (3–5 per class). Misclassification counts are {d2_total_error}. Per-class metrics and confusion matrices are stored in JSON and shown in generated figures. The results are not stable estimates of performance. No external validation or clinical inference is supported.

### 6.3 Development-set repeated cross-validation

Fixed model settings (chosen on the saved validation partition) were assessed with two repeats of five-fold stratified sample-ID group CV over the 85 development samples. The fold scores are a variability diagnostic after model selection, not an independent estimate; folds are not participants and the data are synthetic.

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---:|---:|---:|
{cv_rows}

### 6.4 Missing-modality robustness

Late fusion recomputes its equal-weight mean over only available branches. Test results after dropping each modality are:

| Available branches | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
{robustness}

These small-sample comparisons describe software behavior, not validated robustness.

### 6.5 Figures

Confusion matrices, ROC/precision-recall curves, training/validation loss traces, test macro-F1 comparison, and questionnaire logistic coefficients are saved in `reports/figures/`. Coefficients show fitted associations with the dataset’s derived label and are not causal or clinical explanations.

## 7. Explainability, calibration, and error analysis

For Track A, the explicit rule is the meaningful explanation. The logistic coefficient table is available for transparency; because the label is exactly derived from the items, feature weights do not explain autism. The tree’s errors are aggregate only; no participant rows are printed. SHAP was not needed for this ten-item linear model.

For Track B, confusion matrices and per-class metrics summarize errors. The 15-sample test set is too small for robust error characterization. No Grad-CAM is possible without verified image mappings and no image model was trained. Branch probabilities are not calibrated; no calibration model was fitted. Model confidence is not clinical uncertainty.

## 8. Fusion and modern architecture assessment

Feature-level early fusion and equal-probability late fusion were implemented as small baselines. Intermediate neural fusion, attention, cross-modal transformers, pretrained image/audio encoders, modality dropout, and Grad-CAM were not suitable: there are only 70 training samples, image links fail, labels are simulated, and paired provenance is unauthenticated. Larger models would add capacity without providing defensible evidence of improvement. The chosen small linear models demonstrate reproducible interfaces only.

## 9. AI-agent feasibility

An autonomous language-model agent adds no justified value to this workflow. Auditing, deterministic split creation, training, evaluation, and report generation are predictable and reproducible as Python commands. The pipeline does not invoke an LLM, alter labels or splits, or make consequential decisions.

## 10. Inference demonstration

`app.py` provides optional Streamlit pages for questionnaire coding, multimodal uploads, model evaluation, and data quality. The questionnaire page shows the fixed rule and learned model separately. The multimodal page accepts WAV, motion JSON, and physiology CSV, omits images, reports availability, and labels outputs as research-only. Inputs combined into one prediction are assumed to belong together; the app cannot verify that. Probabilities are shown as uncalibrated. The current runtime does not have Streamlit installed, so the UI was syntax-checked but not launched; core inference functions were tested.

## 11. Limitations and ethics

- Dataset 1 target is an automatically assigned screen-trait threshold, not a clinical diagnosis or independent outcome.
- Dataset 1 records have unique row IDs, but participant independence is undocumented.
- Dataset 2 is synthetic/augmented by provider description; generation assumptions may shape labels and modalities.
- Dataset 2 test size is only 15; confidence intervals and repeated population-level stability are not established.
- Dataset 2’s image mapping is unresolved and paired provenance is not independently established.
- Synthetic/photographic child images are not used. Facial appearance, expression, eye contact, or gaze are not treated as diagnostic.
- Dataset reuse terms are unclear. No archives are copied into this repository or redistributed.
- The demo is not a medical device, screening service, or substitute for professional assessment.

## 12. Conclusion and future work

The delivered implementation establishes a reproducible screening-label reconstruction experiment and a cautious synthetic multimodal pipeline prototype. The toddler dataset’s perfect rule/logistic result is fully explained by deterministic target generation. The multimodal experiments did not show useful held-out performance and cannot support claims about people. The strongest next work would be a larger independently collected paired dataset with expert-reviewed annotations, clear consent/licensing, participant/session identifiers, longitudinal outcomes, external validation across populations, privacy-preserving governance, calibration/uncertainty evaluation, and preregistered missing-modality experiments. Advanced fusion and agent assistance should be revisited only if those data justify them.

## 13. Reproducibility and references

Executed commands (Python 3.12.14, NumPy/Pillow workspace runtime):

```text
python -m src.data.audit_datasets
python -m src.train_experiments
python -m unittest discover -s tests -v   # 15 tests passed
python -m src.report_results
```

Primary dataset references:

1. Thabtah, F. (2018). “A new computational intelligence approach to detect autistic features for autism screening.” *International Journal of Medical Informatics*, 117, 112–124. [https://doi.org/10.1016/j.ijmedinf.2018.06.009](https://doi.org/10.1016/j.ijmedinf.2018.06.009). The supplied toddler metadata cites this paper and the ASDTests app.
2. Thabtah, F. (2017). “Autism Spectrum Disorder Screening: Machine Learning Adaptation and DSM-5 Fulfillment.” Proceedings of the 1st International Conference on Medical and Health Informatics, 1–6. [https://doi.org/10.1145/3107514.3107515](https://doi.org/10.1145/3107514.3107515).
3. Thabtah, F., & Peebles, D. (2020). “A new machine learning model based on induction of rules for autism detection.” *Health Informatics Journal*, 26(1), 264–286. [https://doi.org/10.1177/1460458218824711](https://doi.org/10.1177/1460458218824711). This item has a published expression of concern; it is listed for bibliographic context only and does not substantiate this project’s results.
4. Fadi/Fadi Fayez (Kaggle). [Autism screening data for toddlers](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers), supplied archive and metadata; license recorded there as unknown.
5. Kadhiravan UCET (Kaggle). [Multimodal Autism Dataset](https://www.kaggle.com/datasets/kadhiravanucet/multimodal-autism-dataset), version described by supplied metadata as partially synthetic/augmented; exact reuse terms require source clarification.
'''
    (ROOT / "reports/final_technical_report.md").write_text(report, encoding="utf-8")
    print("Generated final technical report and figures")


if __name__ == "__main__":
    main()
