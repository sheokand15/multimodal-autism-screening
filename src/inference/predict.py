"""Research-only inference interfaces for the saved tabular and multimodal models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features.extract import features_from_files, score_rule
from src.fusion import early_concat, late_average

ROOT = Path(__file__).resolve().parents[2]


def _read_artifact(name):
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def _probabilities(model, vector):
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or len(vector) != len(model["feature_names"]):
        raise ValueError("Feature vector does not match the fitted training schema")
    mean = np.asarray(model["scaler"]["mean"], dtype=float)
    scale = np.asarray(model["scaler"]["scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    z = (vector - mean) / scale
    scores = np.r_[1.0, z] @ weights
    scores -= scores.max()
    probs = np.exp(scores); probs /= probs.sum()
    return {str(label): float(prob) for label, prob in zip(model["classes"], probs)}


def predict_questionnaire(responses):
    """Return the transparent threshold and a separately named learned prediction."""
    responses = list(responses)
    rule_code = score_rule(responses)
    binary = [int(value) for value in responses]
    score = sum(binary)
    rule = "Yes" if rule_code else "No"
    model = _read_artifact("dataset1_logistic.json")
    prediction_probs = _probabilities(model, [float(value) for value in binary])
    learned_prediction = max(prediction_probs, key=prediction_probs.get)
    return {"screening_rule": {"score": score, "threshold": "> 3", "screening_trait_output": rule},
            "learned_model": {"model": "logistic_regression", "screening_trait_prediction": learned_prediction,
                              "uncalibrated_class_probabilities": prediction_probs},
            "limitation": "Reproduces the supplied questionnaire screening-trait label; this is not a clinical diagnosis or independent autism detection."}


def predict_multimodal_files(audio_path=None, motion_path=None, physiology_path=None):
    """Predict from verified uploaded files. Images are excluded due unresolved mappings."""
    if not any((audio_path, motion_path, physiology_path)):
        raise ValueError("Provide at least one supported file: audio WAV, motion JSON, physiology CSV")
    features = features_from_files(audio_path, motion_path, physiology_path)
    models = _read_artifact("dataset2_models.json")
    branch_predictions = {}
    branch_probs = []
    for modality, (vector, names) in features.items():
        model = models[modality]
        if list(names) != model["feature_names"]:
            raise ValueError(f"Feature schema mismatch for {modality}")
        probs = _probabilities(model, vector)
        branch_predictions[modality] = {"prediction": max(probs, key=probs.get), "class_probabilities": probs}
        branch_probs.append(probs)
    classes = models["late_fusion"]["classes"]
    averaged_array = late_average([np.asarray([[row[label] for label in classes]], dtype=float) for row in branch_probs])[0]
    averaged = {label: float(prob) for label, prob in zip(classes, averaged_array)}
    result = {"available_modalities": list(features), "branch_predictions": branch_predictions,
              "selected_fusion_prediction": {"method": "mean probabilities over the available modality branches", "prediction": max(averaged, key=averaged.get), "class_probabilities": averaged},
              "limitation": "Research-only prototype trained on provider-described synthetic samples; modality alignment and class labels are not clinically validated. Not a diagnosis."}
    if set(features) == {"audio", "motion", "physiology"}:
        expected = models["early_fusion"]["feature_names"]
        actual = [f"{modality}__{name}" for modality in ("audio", "motion", "physiology") for name in features[modality][1]]
        if actual == expected:
            blocks = {modality: (features[modality][0][None, :], features[modality][1]) for modality in ("audio", "motion", "physiology")}
            vector, _ = early_concat(blocks)
            vector = vector[0]
            probs = _probabilities(models["early_fusion"], vector)
            result["early_fusion_prediction"] = {"prediction": max(probs, key=probs.get), "class_probabilities": probs}
    return result
