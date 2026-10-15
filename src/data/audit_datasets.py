"""Read-only forensic audit for the supplied autism screening archives."""
from __future__ import annotations

import csv
import array
import hashlib
import io
import json
import math
import statistics
import sys
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from src.config import DATA_ROOT, PROJECT_ROOT
from src.data.splitting import stratified_assignments

ROOT = PROJECT_ROOT
TODDLER = DATA_ROOT / "autism-screening-for-toddlers-data.zip"
MULTI = DATA_ROOT / "multimodal-autism-dataset-zip.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_rows(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig")))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    report = ["# Data Quality Report\n\nAutomated read-only audit. Input archives were not modified.\n"]

    with zipfile.ZipFile(TODDLER) as archive:
        bad_member = archive.testzip()
        if bad_member is not None: raise ValueError(f"ZIP CRC failure: {bad_member}")
        name = "Toddler Autism dataset July 2018.csv"
        rows = csv_rows(archive, name)
        columns = list(rows[0])
        target = next(column for column in columns if column.strip() == "Class/ASD Traits")
        items = [f"A{i}" for i in range(1, 11)]
        scores = [int(row["Qchat-10-Score"]) for row in rows]
        item_sums = [sum(int(row[column]) for column in items) for row in rows]
        labels = [row[target].strip() for row in rows]
        missing = sum(not value.strip() for row in rows for value in row.values())
        unique = len({tuple(row[column] for column in columns) for row in rows})
        ages = [int(row["Age_Mons"]) for row in rows]
        invalid_items = sum(row[column] not in ("0", "1") for row in rows for column in items)
        threshold_labels = ["Yes" if score > 3 else "No" for score in scores]
        report.extend([
            "\n## Dataset 1 — toddler questionnaire\n",
            f"- File: `{name}`; rows: {len(rows)}; columns: {len(columns)}.\n",
            f"- Actual target header: `{target}` (trailing whitespace preserved in source).\n",
            f"- Missing cells: {missing}; duplicate rows: {len(rows)-unique}; unique Case_No values: {len({row['Case_No'] for row in rows})}.\n",
            f"- Labels: {dict(Counter(labels))}.\n",
            f"- Age_Mons range: {min(ages)}–{max(ages)}.\n",
            f"- Qchat-10-Score distribution: {dict(Counter(scores))}.\n",
            f"- Item-positive counts: {dict((column, sum(int(row[column]) for row in rows)) for column in items)}.\n",
            f"- A1–A10 response values outside 0/1: {invalid_items}.\n",
            f"- `Qchat-10-Score == sum(A1..A10)`: {sum(a == b for a, b in zip(scores, item_sums))}/{len(rows)}.\n",
            f"- Target equals (`score > 3`): {sum(a.lower() == b.lower() for a, b in zip(labels, threshold_labels))}/{len(rows)}.\n",
        ])
        if len(rows) != 1054 or any(a != b for a, b in zip(scores, item_sums)) or labels != threshold_labels:
            raise ValueError("Unexpected toddler dataset contract; inspect before proceeding")

        # Fixed-seed-derived ordering, stratified per target class, no identifier as feature.
        assignments = stratified_assignments([{"sample_id": row["Case_No"], "label": labels[i]} for i, row in enumerate(rows)], "label", "sample_id")
        manifest = [{"sample_id": row["Case_No"], "label": labels[i], "split": assignments[row["Case_No"]]} for i, row in enumerate(rows)]
        write_csv(ROOT / "data/manifests/track_a_manifest.csv", manifest, ["sample_id", "label", "split"])
        report.append("\n### Dataset 1 split feasibility\n")
        for split in ("train", "validation", "test"):
            report.append(f"- {split}: {sum(row['split'] == split for row in manifest)} records; class counts {dict(Counter(row['label'] for row in manifest if row['split'] == split))}.\n")
        report.append("\nCategorical distributions:\n")
        for column in ("Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who completed the test"):
            report.append(f"- `{column}`: {dict(Counter(row[column].strip() for row in rows))}\n")
        report.append("\nThe score and target are excluded from ML predictors. Case_No is an identifier and excluded. No participant linkage is supplied.\n")

    with zipfile.ZipFile(MULTI) as archive:
        bad_member = archive.testzip()
        if bad_member is not None: raise ValueError(f"ZIP CRC failure: {bad_member}")
        index = csv_rows(archive, "autism_dataset_index.csv")
        names = set(archive.namelist())
        labels = Counter(row["label"] for row in index)

        image_info, image_hashes, image_errors, image_phashes = [], defaultdict(list), [], []
        for name in sorted(path for path in names if path.startswith("images/")):
            data = archive.read(name)
            image_hashes[digest(data)].append(name)
            try:
                image = Image.open(io.BytesIO(data))
                image.verify()
                image = Image.open(io.BytesIO(data))
                image_info.append((name, image.size, image.mode, image.format))
                thumb = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = list(thumb.tobytes())
                image_phashes.append((name, tuple(pixels[y * 9 + x + 1] > pixels[y * 9 + x] for y in range(8) for x in range(8))))
            except Exception as error:
                image_errors.append((name, str(error)))

        audio_info, audio_hashes, audio_errors, audio_waveforms = [], defaultdict(list), [], []
        for name in sorted(path for path in names if path.startswith("voice/")):
            data = archive.read(name)
            audio_hashes[digest(data)].append(name)
            try:
                with wave.open(io.BytesIO(data), "rb") as audio:
                    frames, rate = audio.getnframes(), audio.getframerate()
                    channels, width = audio.getnchannels(), audio.getsampwidth()
                    raw = audio.readframes(frames)
                    samples = array.array("h")
                    if width == 2:
                        samples.frombytes(raw)
                        if sys.byteorder != "little": samples.byteswap()
                    peak = max((abs(value) for value in samples), default=0)
                    rms = (sum(int(value) ** 2 for value in samples) / len(samples)) ** .5 if samples else 0
                    silence = sum(abs(value) < 33 for value in samples) / len(samples) if samples else 1
                    clipped = sum(abs(value) >= 32760 for value in samples) / len(samples) if samples else 0
                    audio_info.append((name, rate, channels, width, frames, frames / rate if rate else 0, peak, rms, silence, clipped))
                    audio_waveforms.append(np.asarray(samples[::16], dtype=float))
            except Exception as error:
                audio_errors.append((name, str(error)))

        motion_info, motion_hashes, motion_errors, motion_values = [], defaultdict(list), [], []
        for name in sorted(path for path in names if path.startswith("motion/")):
            data = archive.read(name)
            motion_hashes[digest(data)].append(name)
            try:
                content = json.loads(data)
                lengths = {key: len(value) for key, value in content.items() if isinstance(value, list)}
                values = [number for series in content.values() if isinstance(series, list) for point in series if isinstance(point, list) for number in point if isinstance(number, (int, float))]
                motion_values.extend(float(value) for value in values)
                motion_info.append((name, tuple(sorted(content)), lengths, len(values), min(values) if values else None, max(values) if values else None))
            except Exception as error:
                motion_errors.append((name, str(error)))

        physio_info, physio_hashes, physio_errors, temperature_sequences = [], defaultdict(list), [], []
        time_intervals, nonmonotonic_time = [], 0
        for name in sorted(path for path in names if path.startswith("physio/")):
            data = archive.read(name)
            physio_hashes[digest(data)].append(name)
            try:
                measurements = list(csv.DictReader(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig")))
                columns = list(measurements[0]) if measurements else []
                values = {column: [float(row[column]) for row in measurements if row[column].strip()] for column in columns}
                constants = [column for column, vals in values.items() if len(set(vals)) <= 1]
                temperature_sequences.append(tuple(values.get("TEMP", [])))
                times = values.get("Time", [])
                deltas = [right - left for left, right in zip(times, times[1:])]
                if any(delta <= 0 for delta in deltas): nonmonotonic_time += 1
                time_intervals.extend(deltas)
                physio_info.append((name, len(measurements), columns, constants, values))
            except Exception as error:
                physio_errors.append((name, str(error)))

        # Deterministic within-class hash ordering; group is the index sample_id.
        split_by_id = stratified_assignments(index, "label", "sample_id")
        manifest = []
        for row in index:
            resolved = {modality: row[field] in names for modality, field in (("image", "image"), ("audio", "voice"), ("motion", "motion"), ("physiology", "physio"))}
            manifest.append({
                "sample_id": row["sample_id"], "label": row["label"],
                "split": split_by_id[row["sample_id"]],
                "image_path": row["image"] if resolved["image"] else "",
                "audio_path": row["voice"] if resolved["audio"] else "",
                "motion_path": row["motion"] if resolved["motion"] else "",
                "physiology_path": row["physio"] if resolved["physiology"] else "",
                "available_modalities": ";".join(modality for modality, available in resolved.items() if available),
                "mapping_status": "partial_image_unresolved" if not resolved["image"] else "paths_resolve_provenance_unverified",
                "exclusion_reason": "image mapping unresolved; cross-modal provenance not independently verified" if not resolved["image"] else "",
            })
        write_csv(ROOT / "data/manifests/track_b_manifest.csv", manifest, ["sample_id", "label", "split", "image_path", "audio_path", "motion_path", "physiology_path", "available_modalities", "mapping_status", "exclusion_reason"])

        def duplicate_groups(groups):
            return sum(len(paths) > 1 for paths in groups.values())

        near_image_pairs = 0
        for i, (_, left_hash) in enumerate(image_phashes):
            for _, right_hash in image_phashes[i + 1:]:
                if sum(a != b for a, b in zip(left_hash, right_hash)) <= 4:
                    near_image_pairs += 1
        audio_matrix = np.stack(audio_waveforms)
        audio_matrix -= audio_matrix.mean(axis=1, keepdims=True)
        audio_norms = np.linalg.norm(audio_matrix, axis=1, keepdims=True)
        audio_matrix /= np.where(audio_norms == 0, 1, audio_norms)
        audio_correlations = audio_matrix @ audio_matrix.T
        upper = audio_correlations[np.triu_indices(len(audio_info), k=1)]
        suspicious_audio_pairs = int(np.sum(np.abs(upper) >= .95))
        max_abs_audio_correlation = float(np.max(np.abs(upper)))

        labels = Counter(row["label"] for row in index)
        image_sizes = Counter(row[1] for row in image_info)
        physiology_ranges = {column: (min(min(item[4][column]) for item in physio_info), max(max(item[4][column]) for item in physio_info)) for column in ("HR", "GSR", "TEMP")}
        report.extend([
            "\n## Dataset 2 — multimodal synthetic dataset\n",
            f"- Index rows: {len(index)}; labels: {dict(labels)}; unique sample IDs: {len({row['sample_id'] for row in index})}.\n",
            f"- Exact indexed paths resolving: image {sum(bool(row['image_path']) for row in manifest)}/{len(manifest)}, audio {sum(bool(row['audio_path']) for row in manifest)}/{len(manifest)}, motion {sum(bool(row['motion_path']) for row in manifest)}/{len(manifest)}, physiology {sum(bool(row['physiology_path']) for row in manifest)}/{len(manifest)}.\n",
            f"- Images: decoded {len(image_info)}/100; width range {min(size[0] for size in image_sizes)}–{max(size[0] for size in image_sizes)}, height range {min(size[1] for size in image_sizes)}–{max(size[1] for size in image_sizes)}; RGB {sum(row[2] == 'RGB' for row in image_info)}/100; exact duplicate groups {duplicate_groups(image_hashes)}; dHash near-duplicate candidate pairs (Hamming distance ≤4): {near_image_pairs}; decode errors {len(image_errors)}.\n",
            f"- WAV: decoded {len(audio_info)}/100; rates {dict(Counter(row[1] for row in audio_info))}; channels {dict(Counter(row[2] for row in audio_info))}; durations min/median/max {min(row[5] for row in audio_info):.3f}/{statistics.median(row[5] for row in audio_info):.3f}/{max(row[5] for row in audio_info):.3f}s; silence fraction median {statistics.median(row[8] for row in audio_info):.5f}; clipped fraction max {max(row[9] for row in audio_info):.5f}; exact duplicate groups {duplicate_groups(audio_hashes)}; max absolute downsampled waveform correlation {max_abs_audio_correlation:.3f}; pairs at |r|≥0.95: {suspicious_audio_pairs}; errors {len(audio_errors)}.\n",
            f"- Motion JSON: decoded {len(motion_info)}/100; schemas {dict(Counter(row[1] for row in motion_info))}; sequence lengths {dict(Counter(tuple(sorted(row[2].values())) for row in motion_info))}; numeric range {min(motion_values):.4f} to {max(motion_values):.4f}; nonfinite values {sum(not math.isfinite(value) for value in motion_values)}; duplicate groups {duplicate_groups(motion_hashes)}; errors {len(motion_errors)}.\n",
            f"- Physiology CSV: decoded {len(physio_info)}/100; row counts {dict(Counter(row[1] for row in physio_info))}; column schemas {dict(Counter(tuple(row[2]) for row in physio_info))}; constant column counts {dict(Counter(column for row in physio_info for column in row[3]))}; duplicate groups {duplicate_groups(physio_hashes)}; errors {len(physio_errors)}.\n",
            f"- Physiology time ordering: {len(physio_info)-nonmonotonic_time}/100 files strictly increase; interval min/median/max {min(time_intervals):.6f}/{statistics.median(time_intervals):.6f}/{max(time_intervals):.6f} seconds.\n",
            f"- Raw value ranges as stored (units undocumented): {physiology_ranges}.\n",
            "\n### Interpretation and mapping\n",
            "- Both archives passed `ZipFile.testzip()` (no CRC errors).\n",
            "- Exact sample-ID references resolve audio, motion and physiology paths; they do not independently verify participant provenance or synchronization.\n",
            "- All image paths from the index are unresolved: index references `images/child_NNN.png`, while actual files are `images/Child (n).jpg`. No filename-position matching was performed. Image paths remain blank in the manifest.\n",
            "- Dataset metadata explicitly describes generated/synthetic or augmented content. Labels are simulated dataset labels and cannot establish clinical associations.\n",
            f"- Each physiology CSV has `Time`, `HR`, `GSR`, `TEMP` and 100 records from 0 to 10 seconds. The entire TEMP series is identical across all samples ({len(set(temperature_sequences))} unique sequences). Units and medical meanings are not inferred.\n",
            "- Split feasibility: one indexed record per sample ID, with 19–33 examples per class. A stratified file-level 70/15/15 partition is feasible, but cannot be described as participant-level clinical generalization.\n",
            "- Dataset 2 manifest uses a fixed seed-derived stratified 70/15/15 split by sample_id; exact assignments are saved.\n",
            *[f"- {split}: {sum(row['split'] == split for row in manifest)} rows; class counts {dict(Counter(row['label'] for row in manifest if row['split'] == split))}.\n" for split in ("train", "validation", "test")],
        ])

    (ROOT / "reports/data_quality_report.md").write_text("".join(report), encoding="utf-8")
    (ROOT / "reports/label_audit.md").write_text("""# Label Audit

## Toddler questionnaire dataset

The actual target header is `Class/ASD Traits ` (trailing space), with `Yes` (728) and `No` (326). Supplied metadata documents a positive label for Q-CHAT-10 score greater than 3 and says the class is assigned automatically by the screening application. In the archive, `Qchat-10-Score` equals the sum of A1–A10 for all 1,054 records, and target equals `score > 3` for all 1,054. Score and target are mechanically linked.

The target represents the dataset’s screening-trait label, not a confirmed clinical diagnosis. The score is excluded as a predictor; A1–A10 may assess reconstruction of that same rule, with the rule itself reported as the transparent baseline.

## Multimodal dataset

The index contains 100 labels: typical 25, mild_asd 23, moderate_asd 19, severe_asd 33. Supplied metadata describes a partially synthetic and augmented dataset and explicitly disclaims real individuals and clinical use. Treat these as simulated class labels. File-level correlations may reflect the generation recipe, duplicates, templates, or class-specific artifacts; they do not establish behavioral or physiological biomarkers.
""", encoding="utf-8")
    (ROOT / "reports/assumptions_and_limitations.md").write_text("""# Assumptions and Limitations

- Dataset 1 screening labels are an exact deterministic threshold on the supplied score/items; high accuracy is label reconstruction, not independent autism detection.
- Dataset 1 unique Case_No values do not prove distinct participants. There are no supplied linkage fields; records are provisionally treated as independent for splitting.
- Dataset 2 is described by its provider as synthetic/augmented. Class definitions and generation procedures are not clinically validated in supplied materials.
- Dataset 2 exact paths link WAV, motion JSON and physiology CSV to index sample IDs; they do not authenticate shared participant identity or timing.
- Dataset 2 image mapping is unresolved. No images are assigned to samples or used for modeling.
- Dataset licensing/reuse terms are unclear in both metadata files. Input archives remain outside this repository.
- Numeric modality checks are structural, not medical validation. No undocumented units or clinical meanings are inferred.
""", encoding="utf-8")
    print("Wrote data-quality report and track manifests")


if __name__ == "__main__":
    main()
