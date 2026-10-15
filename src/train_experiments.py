"""Train leakage-aware baselines for both dataset tracks (NumPy only)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation.metrics import accuracy_wilson_ci, binary_curves, classification_metrics
from src.features.extract import multimodal_features, questionnaire_design, questionnaire_rows, score_rule
from src.models.classical import DecisionTree, LogisticRegression
from src.preprocessing import Standardizer
from src.data.splitting import stratified_folds
from src.fusion import early_concat, late_average

ROOT = Path(__file__).resolve().parents[1]


def load_manifest(name):
    with (ROOT / "data/manifests" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def indices_for(manifest, split):
    return [i for i, row in enumerate(manifest) if row["split"] == split]


def choose_macro_f1(results):
    return max(results, key=lambda row: (row["validation"]["macro_f1"], row["validation"]["balanced_accuracy"], -row["regularization"] if "regularization" in row else -row["max_depth"]))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def train_track_a(registry):
    manifest = load_manifest("track_a_manifest.csv")
    raw = questionnaire_rows()
    targets = np.asarray([row["label"] for row in manifest])
    train, val, test = [indices_for(manifest, split) for split in ("train", "validation", "test")]
    x, feature_names, categories = questionnaire_design(raw, train)
    scaler = Standardizer().fit(x[train])
    xs = scaler.transform(x)
    y_train, y_val, y_test = targets[train], targets[val], targets[test]
    classes = np.unique(targets)
    results = []
    for reg in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(regularization=reg).fit(xs[train], y_train, validation_data=(xs[val], y_val))
        val_result = classification_metrics(y_val, model.predict(xs[val]), classes)
        results.append({"regularization": reg, "validation": val_result, "model": model})
    selected = choose_macro_f1(results)
    logistic = selected["model"]
    pred = logistic.predict(xs[test])
    logistic_result = classification_metrics(y_test, pred, classes)
    logistic_result["validation_selected_regularization"] = selected["regularization"]
    logistic_result["test_accuracy_95pct_wilson_ci"] = accuracy_wilson_ci(y_test, pred)
    positive_column = list(logistic.classes_).index("Yes")
    logistic_curve = binary_curves(y_test, logistic.predict_proba(xs[test])[:, positive_column])
    logistic_artifact = logistic.to_dict(scaler, feature_names)
    logistic_artifact.update({"categories": categories, "target": "Class/ASD Traits ", "excluded_predictors": ["Case_No", "Qchat-10-Score", "Class/ASD Traits "]})
    save_json(ROOT / "artifacts/dataset1_logistic.json", logistic_artifact)
    coefficients = logistic.weights_[1:, :]
    with (ROOT / "artifacts/dataset1_feature_coefficients.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["feature", "coefficient_No", "coefficient_Yes", "absolute_max_coefficient"])
        for name, coef in sorted(zip(feature_names, coefficients), key=lambda pair: -float(np.max(np.abs(pair[1])))):
            writer.writerow([name, float(coef[0]), float(coef[1]), float(np.max(np.abs(coef)))])

    trees = []
    for depth in (3, 5, 10):
        tree = DecisionTree(max_depth=depth, min_leaf=2).fit(x[train], y_train)
        val_result = classification_metrics(y_val, tree.predict(x[val]), classes)
        trees.append({"max_depth": depth, "validation": val_result, "model": tree})
    selected_tree = choose_macro_f1(trees)
    tree = selected_tree["model"]
    tree_result = classification_metrics(y_test, tree.predict(x[test]), classes)
    tree_result["validation_selected_max_depth"] = selected_tree["max_depth"]
    save_json(ROOT / "artifacts/dataset1_tree.json", tree.to_dict(feature_names))

    rule_pred = np.asarray(["Yes" if score_rule([int(raw[i][f"A{k}"]) for k in range(1, 11)]) else "No" for i in test])
    rule_result = classification_metrics(y_test, rule_pred, classes)
    rule_result["test_accuracy_95pct_wilson_ci"] = accuracy_wilson_ci(y_test, rule_pred)
    rule_curve = binary_curves(y_test, [float(score_rule([int(raw[i][f"A{k}"]) for k in range(1, 11)])) for i in test])
    majority = np.repeat(max(set(y_train), key=list(y_train).count), len(y_test))
    majority_result = classification_metrics(y_test, majority, classes)
    model_results = {"questionnaire_scoring_rule": rule_result, "logistic_regression": logistic_result, "decision_tree": tree_result, "train_majority": majority_result}
    save_json(ROOT / "artifacts/dataset1_results.json", model_results)
    save_json(ROOT / "artifacts/dataset1_curves.json", {"questionnaire_scoring_rule": rule_curve, "logistic_regression": logistic_curve})
    # Development-only repeated stratified group-CV. The held-out test rows are never used here.
    development = sorted(train + val)
    development_records = [manifest[i] for i in development]
    cv_scores = {"logistic_regression": [], "decision_tree": []}
    for repeat_seed in (42, 43):
        fold_by_id = stratified_folds(development_records, "label", "sample_id", folds=5, seed=repeat_seed)
        for fold in range(5):
            fold_train = [i for i in development if fold_by_id[manifest[i]["sample_id"]] != fold]
            fold_valid = [i for i in development if fold_by_id[manifest[i]["sample_id"]] == fold]
            fold_scaler = Standardizer().fit(x[fold_train]); fold_x = fold_scaler.transform(x)
            fold_model = LogisticRegression(regularization=selected["regularization"], epochs=1600).fit(fold_x[fold_train], targets[fold_train])
            cv_scores["logistic_regression"].append(classification_metrics(targets[fold_valid], fold_model.predict(fold_x[fold_valid]), classes))
            fold_tree = DecisionTree(max_depth=selected_tree["max_depth"], min_leaf=2).fit(x[fold_train], targets[fold_train])
            cv_scores["decision_tree"].append(classification_metrics(targets[fold_valid], fold_tree.predict(x[fold_valid]), classes))
    cv_results = {name: {metric: {"mean": float(np.mean([row[metric] for row in values])), "std": float(np.std([row[metric] for row in values], ddof=1))}
                         for metric in ("accuracy", "balanced_accuracy", "macro_f1")} for name, values in cv_scores.items()}
    save_json(ROOT / "artifacts/dataset1_cv_results.json", {"repeats": 2, "folds_per_repeat": 5, "development_n": len(development), "models": cv_results,
                                                             "selection_note": "Fixed model settings selected using the saved validation partition; CV is a development stability diagnostic, not an independent final test."})
    for name, result in model_results.items(): registry.append({"track": "A", "experiment": name, "selection": "fixed rule" if name in ("questionnaire_scoring_rule", "train_majority") else "validation macro-F1", "test_n": result["n"], "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"], "macro_f1": result["macro_f1"], "interpretation": "screening-rule label reconstruction only; not clinical diagnosis"})
    return model_results


def train_track_b(registry):
    manifest = load_manifest("track_b_manifest.csv")
    labels = np.asarray([row["label"] for row in manifest])
    sample_ids = [row["sample_id"] for row in manifest]
    splits = {split: indices_for(manifest, split) for split in ("train", "validation", "test")}
    classes = np.unique(labels)
    features_by_modality, names_by_modality = {}, {}
    for index, sample_id in enumerate(sample_ids):
        parts = multimodal_features(sample_id)
        for modality, (values, names) in parts.items():
            features_by_modality.setdefault(modality, []).append(values)
            names_by_modality[modality] = names
    arrays = {modality: np.stack(values) for modality, values in features_by_modality.items()}
    model_state, branch_probabilities, branch_outputs = {}, {}, {}
    branch_validation = {}
    selected_regularization = {}
    for modality, x in arrays.items():
        train, val, test = (splits[split] for split in ("train", "validation", "test"))
        scaler = Standardizer().fit(x[train])
        z = scaler.transform(x)
        candidates = []
        for reg in (0.1, 1.0, 10.0):
            model = LogisticRegression(regularization=reg, epochs=2200).fit(z[train], labels[train], validation_data=(z[val], labels[val]))
            val_result = classification_metrics(labels[val], model.predict(z[val]), classes)
            candidates.append({"regularization": reg, "validation": val_result, "model": model, "scaler": scaler})
        selected = choose_macro_f1(candidates)
        model, scaler = selected["model"], selected["scaler"]
        selected_regularization[modality] = selected["regularization"]
        probabilities = model.predict_proba(scaler.transform(x))
        branch_probabilities[modality] = probabilities
        branch_outputs[modality] = classification_metrics(labels[test], model.classes_[np.argmax(probabilities[test], axis=1)], classes)
        branch_outputs[modality]["validation_selected_regularization"] = selected["regularization"]
        branch_validation[modality] = selected["validation"]
        model_state[modality] = model.to_dict(scaler, names_by_modality[modality])

    # Early fusion concatenates hand-engineered features; intermediate deep fusion is unsupported at n=70 train.
    modalities = list(arrays)
    early_x, early_names = early_concat({modality: (arrays[modality], names_by_modality[modality]) for modality in modalities})
    train, val, test = (splits[split] for split in ("train", "validation", "test"))
    early_scaler = Standardizer().fit(early_x[train]); early_z = early_scaler.transform(early_x)
    early_candidates = []
    for reg in (0.1, 1.0, 10.0):
        model = LogisticRegression(regularization=reg, epochs=2200).fit(early_z[train], labels[train], validation_data=(early_z[val], labels[val]))
        early_candidates.append({"regularization": reg, "validation": classification_metrics(labels[val], model.predict(early_z[val]), classes), "model": model})
    early_selected = choose_macro_f1(early_candidates)
    early_model = early_selected["model"]
    selected_regularization["early_fusion"] = early_selected["regularization"]
    early_result = classification_metrics(labels[test], early_model.predict(early_z[test]), classes)
    early_result["validation_selected_regularization"] = early_selected["regularization"]
    model_state["early_fusion"] = early_model.to_dict(early_scaler, early_names)

    # Late fusion is a validation-chosen unweighted mean of all available non-image branch probabilities.
    late_val = late_average([branch_probabilities[modality][val] for modality in modalities])
    late_result_val = classification_metrics(labels[val], classes[np.argmax(late_val, axis=1)], classes)
    late_test = late_average([branch_probabilities[modality][test] for modality in modalities])
    late_result = classification_metrics(labels[test], classes[np.argmax(late_test, axis=1)], classes)
    missing_modality = {}
    for omitted in modalities:
        remaining = [modality for modality in modalities if modality != omitted]
        probabilities = late_average([branch_probabilities[modality][test] for modality in remaining])
        missing_modality[f"without_{omitted}"] = classification_metrics(labels[test], classes[np.argmax(probabilities, axis=1)], classes)

    # Development-only repeated stratified group-CV; locked test rows are never accessed here.
    development_indices = sorted(splits["train"] + splits["validation"])
    development_records = [manifest[i] for i in development_indices]
    cv_scores = {name: [] for name in modalities + ["early_fusion", "late_fusion"]}
    for repeat_seed in (42, 43):
        fold_by_id = stratified_folds(development_records, "label", "sample_id", folds=5, seed=repeat_seed)
        for fold in range(5):
            fold_train = [i for i in development_indices if fold_by_id[sample_ids[i]] != fold]
            fold_valid = [i for i in development_indices if fold_by_id[sample_ids[i]] == fold]
            fold_probs = {}
            for modality in modalities:
                x = arrays[modality]
                scaler = Standardizer().fit(x[fold_train]); z = scaler.transform(x)
                model = LogisticRegression(regularization=selected_regularization[modality], epochs=1000).fit(z[fold_train], labels[fold_train])
                fold_probs[modality] = model.predict_proba(z[fold_valid])
                cv_scores[modality].append(classification_metrics(labels[fold_valid], model.classes_[np.argmax(fold_probs[modality], axis=1)], classes))
            scaler = Standardizer().fit(early_x[fold_train]); z = scaler.transform(early_x)
            model = LogisticRegression(regularization=selected_regularization["early_fusion"], epochs=1000).fit(z[fold_train], labels[fold_train])
            cv_scores["early_fusion"].append(classification_metrics(labels[fold_valid], model.predict(z[fold_valid]), classes))
            combined = late_average([fold_probs[modality] for modality in modalities])
            cv_scores["late_fusion"].append(classification_metrics(labels[fold_valid], classes[np.argmax(combined, axis=1)], classes))
    development_cv = {name: {metric: {"mean": float(np.mean([row[metric] for row in rows])), "std": float(np.std([row[metric] for row in rows], ddof=1))}
                             for metric in ("accuracy", "balanced_accuracy", "macro_f1")} for name, rows in cv_scores.items()}
    model_state["late_fusion"] = {"modalities": modalities, "method": "unweighted_mean_of_branch_probabilities", "classes": classes.tolist()}
    out = {"unimodal": branch_outputs, "early_fusion": early_result, "late_fusion": late_result,
           "missing_modality_robustness": missing_modality,
           "development_repeated_stratified_group_cv": {"repeats": 2, "folds_per_repeat": 5, "development_n": len(development_indices), "models": development_cv,
                                                         "selection_note": "Fixed regularization selected using the saved validation partition; CV is a development stability diagnostic, not an independent final test."},
           "validation": {"unimodal": {m: branch_validation[m] for m in modalities}, "late_fusion": late_result_val},
           "split_sizes": {key: len(value) for key, value in splits.items()}, "classes": classes.tolist(), "modalities": modalities,
           "caveat": "synthetic file-level benchmark only; no clinical or real-participant generalization"}
    save_json(ROOT / "artifacts/dataset2_models.json", model_state)
    save_json(ROOT / "artifacts/dataset2_results.json", out)
    with (ROOT / "artifacts/dataset2_test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "true_label", "late_fusion_prediction"])
        for index, probability in zip(test, late_test):
            writer.writerow([sample_ids[index], labels[index], classes[int(np.argmax(probability))]])
    for modality, result in branch_outputs.items(): registry.append({"track": "B", "experiment": f"unimodal_{modality}", "selection": "validation macro-F1", "test_n": result["n"], "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"], "macro_f1": result["macro_f1"], "interpretation": "synthetic file-level benchmark only"})
    for name, result in (("early_fusion", early_result), ("late_fusion", late_result)):
        registry.append({"track": "B", "experiment": name, "selection": "validation macro-F1", "test_n": result["n"], "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"], "macro_f1": result["macro_f1"], "interpretation": "synthetic file-level benchmark only"})
    return out


def main():
    registry = []
    results_a = train_track_a(registry)
    results_b = train_track_b(registry)
    with (ROOT / "artifacts/experiment_registry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["track", "experiment", "selection", "test_n", "accuracy", "balanced_accuracy", "macro_f1", "interpretation"])
        writer.writeheader(); writer.writerows(registry)
    print(json.dumps({"track_a": results_a, "track_b": results_b}, indent=2))


if __name__ == "__main__":
    main()
