from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
import io
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_ROOT, PROJECT_ROOT
from src.data.splitting import stratified_assignments, stratified_folds
from src.features.extract import multimodal_features, questionnaire_design, questionnaire_rows, score_rule
from src.fusion import early_concat, late_average
from src.inference.predict import predict_multimodal_files, predict_questionnaire
from src.models.classical import LogisticRegression
from src.preprocessing import Standardizer


class QuestionnaireTests(unittest.TestCase):
    def test_scoring_rule_and_validation(self):
        self.assertEqual(score_rule([0] * 10), 0)
        self.assertEqual(score_rule([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 1)
        with self.assertRaises(ValueError): score_rule([1] * 9)
        with self.assertRaises(ValueError): score_rule([1] * 9 + [2])

    def test_rule_matches_all_source_targets_and_score(self):
        rows = questionnaire_rows()
        self.assertEqual(len(rows), 1054)
        target = next(key for key in rows[0] if key.strip() == "Class/ASD Traits")
        for row in rows:
            items = [int(row[f"A{i}"]) for i in range(1, 11)]
            self.assertEqual(sum(items), int(row["Qchat-10-Score"]))
            self.assertEqual(score_rule(items), int(row[target].strip().lower() == "yes"))

    def test_predictor_schema_excludes_score_target_and_id(self):
        rows = questionnaire_rows()
        matrix, names, _ = questionnaire_design(rows, range(750))
        self.assertEqual(matrix.shape, (1054, 10))
        self.assertEqual(names, [f"A{i}" for i in range(1, 11)])
        self.assertFalse(any("Score" in name or "Class" in name or "Case" in name for name in names))

    def test_questionnaire_inference_output(self):
        result = predict_questionnaire([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result["screening_rule"]["score"], 4)
        self.assertEqual(result["screening_rule"]["screening_trait_output"], "Yes")
        self.assertIn("learned_model", result)
        self.assertAlmostEqual(sum(result["learned_model"]["uncalibrated_class_probabilities"].values()), 1.0)
        with self.assertRaises(ValueError): predict_questionnaire([0.5] * 10)


class SplitAndPreprocessingTests(unittest.TestCase):
    def _manifest(self, name):
        with (PROJECT_ROOT / "data/manifests" / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_saved_splits_are_disjoint_and_cover_all_samples(self):
        for name in ("track_a_manifest.csv", "track_b_manifest.csv"):
            rows = self._manifest(name)
            ids = [row["sample_id"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            sets = [{row["sample_id"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")]
            self.assertEqual(sets[0] & sets[1], set())
            self.assertEqual(sets[0] & sets[2], set())
            self.assertEqual(sets[1] & sets[2], set())
            self.assertEqual(set.union(*sets), set(ids))

    def test_image_mapping_stays_unresolved(self):
        rows = self._manifest("track_b_manifest.csv")
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(not row["image_path"] for row in rows))
        self.assertTrue(all("image" not in row["available_modalities"] for row in rows))

    def test_split_assignments_are_deterministic_and_reject_duplicate_groups(self):
        records = [{"sample_id": str(i), "label": "a" if i < 20 else "b"} for i in range(40)]
        first = stratified_assignments(records, "label", "sample_id", seed=42)
        second = stratified_assignments(records, "label", "sample_id", seed=42)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError): stratified_assignments(records + [records[0]], "label", "sample_id")

    def test_group_folds_cover_each_sample_once_per_repeat(self):
        records = [{"sample_id": str(i), "label": "a" if i < 20 else "b"} for i in range(40)]
        folds = stratified_folds(records, "label", "sample_id", folds=5, seed=42)
        self.assertEqual(set(folds), {row["sample_id"] for row in records})
        self.assertEqual(set(folds.values()), set(range(5)))
        for label in ("a", "b"):
            counts = [sum(folds[row["sample_id"]] == fold for row in records if row["label"] == label) for fold in range(5)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_file_decode_and_nonimage_sample_id_mapping(self):
        with zipfile.ZipFile(DATA_ROOT / "multimodal-autism-dataset-zip.zip") as archive:
            with Image.open(io.BytesIO(archive.read("images/Child (1).jpg"))) as image:
                self.assertGreater(image.width, 0)
            self.assertIn("voice/child_001.wav", archive.namelist())
            self.assertIn("motion/child_001.json", archive.namelist())
            self.assertIn("physio/child_001.csv", archive.namelist())

    def test_transform_uses_only_training_statistics(self):
        x_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler = Standardizer().fit(x_train)
        self.assertEqual(scaler.mean_.tolist(), [2.0, 3.0])
        transformed = scaler.transform(np.array([[100.0, 200.0]]))
        self.assertGreater(float(transformed[0, 0]), 50)

    def test_model_predictions_are_reproducible(self):
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        y = np.array(["a", "b", "b", "c"])
        first = LogisticRegression(regularization=0.1, epochs=100).fit(x, y).predict_proba(x)
        second = LogisticRegression(regularization=0.1, epochs=100).fit(x, y).predict_proba(x)
        np.testing.assert_allclose(first, second)


class MultimodalTests(unittest.TestCase):
    def test_fusion_helpers_validate_shapes_and_average_probabilities(self):
        blocks = {"audio": (np.ones((2, 1)), ["a"]), "motion": (np.ones((2, 2)), ["m1", "m2"]), "physiology": (np.ones((2, 1)), ["p"])}
        joined, names = early_concat(blocks)
        self.assertEqual(joined.shape, (2, 4))
        self.assertEqual(len(names), 4)
        averaged = late_average([np.array([[.8, .2]]), np.array([[.2, .8]])])
        np.testing.assert_allclose(averaged, [[.5, .5]])
        with self.assertRaises(ValueError): early_concat({"audio": blocks["audio"]})
        with self.assertRaises(ValueError): late_average([])

    def test_feature_shapes_and_repeatability(self):
        first = multimodal_features("child_001")
        again = multimodal_features("child_001")
        self.assertEqual({key: len(value[0]) for key, value in first.items()}, {"audio": 10, "motion": 63, "physiology": 21})
        for modality in first:
            np.testing.assert_allclose(first[modality][0], again[modality][0])

    def test_fusion_artifact_compatibility(self):
        models = json.loads((PROJECT_ROOT / "artifacts/dataset2_models.json").read_text(encoding="utf-8"))
        self.assertEqual(models["late_fusion"]["modalities"], ["audio", "motion", "physiology"])
        self.assertEqual(len(models["early_fusion"]["feature_names"]), 94)
        self.assertEqual(len(models["early_fusion"]["weights"]), 95)

    def test_missing_modality_handling_and_output_shape(self):
        with self.assertRaises(ValueError): predict_multimodal_files()
        with zipfile.ZipFile(DATA_ROOT / "multimodal-autism-dataset-zip.zip") as archive, tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            names = ["voice/child_001.wav", "motion/child_001.json", "physio/child_001.csv"]
            paths = {}
            for name in names:
                path = temp / Path(name).name
                path.write_bytes(archive.read(name)); paths[name.split("/")[0]] = path
            result = predict_multimodal_files(paths["voice"], paths["motion"], paths["physio"])
            self.assertEqual(set(result["available_modalities"]), {"audio", "motion", "physiology"})
            self.assertIn("selected_fusion_prediction", result)
            self.assertIn("early_fusion_prediction", result)
            probs = result["selected_fusion_prediction"]["class_probabilities"]
            self.assertAlmostEqual(sum(probs.values()), 1.0)
            partial = predict_multimodal_files(audio_path=paths["voice"])
            self.assertEqual(partial["available_modalities"], ["audio"])


if __name__ == "__main__":
    unittest.main()
