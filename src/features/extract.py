"""Feature extraction for the supplied research datasets; images are unmapped."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import wave
import zipfile
from pathlib import Path

import numpy as np
from src.config import DATA_ROOT

TODDLER_ZIP = DATA_ROOT / "autism-screening-for-toddlers-data.zip"
MULTI_ZIP = DATA_ROOT / "multimodal-autism-dataset-zip.zip"


def questionnaire_rows():
    with zipfile.ZipFile(TODDLER_ZIP) as archive:
        return list(csv.DictReader(io.TextIOWrapper(archive.open("Toddler Autism dataset July 2018.csv"), encoding="utf-8-sig")))


def questionnaire_design(rows, fit_indices):
    """Return only the ten declared questionnaire items; demographics are unnecessary."""
    del fit_indices  # No vocabulary or transform is learned from data for binary item columns.
    names = [f"A{i}" for i in range(1, 11)]
    matrix = [[float(row[name]) for name in names] for row in rows]
    return np.asarray(matrix, dtype=float), names, {}


def score_rule(responses):
    """Reference screen output: positive trait label when sum(A1..A10) > 3."""
    if len(responses) != 10 or any(value not in (0, 1, False, True) for value in responses):
        raise ValueError("Provide exactly ten binary A1–A10 responses")
    return int(sum(responses) > 3)


def _wav_features(data):
    with wave.open(io.BytesIO(data), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM WAV")
        rate = stream.getframerate()
        raw = stream.readframes(stream.getnframes())
    signal = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if len(signal) == 0:
        raise ValueError("Empty WAV")
    frame = max(64, int(rate * 0.025))
    hop = max(32, int(rate * 0.010))
    frames = np.stack([signal[i:i + frame] for i in range(0, max(1, len(signal) - frame + 1), hop) if len(signal[i:i + frame]) == frame])
    if frames.size == 0:
        frames = signal[:min(len(signal), frame)][None, :]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    window = np.hanning(frames.shape[1])
    spectra = np.abs(np.fft.rfft(frames * window, n=512, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(512, 1 / rate)
    centroid = (spectra * freqs).sum(axis=1) / spectra.sum(axis=1)
    prob = spectra / spectra.sum(axis=1, keepdims=True)
    entropy = -(prob * np.log(prob)).sum(axis=1)
    zero_cross = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
    return [float(np.mean(rms)), float(np.std(rms)), float(np.max(rms)), float(np.mean(centroid)), float(np.mean(entropy)), float(np.mean(zero_cross)), float(np.mean(np.abs(signal) < 1e-4)), float(np.mean(np.abs(signal) >= 0.995)), float(rate), float(len(signal) / rate)]


def _series_features(values, time):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if len(values) != len(time) or not len(values):
        raise ValueError("Invalid time series")
    diff = np.diff(values)
    centered_time = time - time.mean()
    slope = float(np.dot(centered_time, values - values.mean()) / np.dot(centered_time, centered_time)) if np.dot(centered_time, centered_time) else 0.0
    return [float(values.mean()), float(values.std()), float(values.min()), float(values.max()), float(np.mean(np.abs(diff))) if len(diff) else 0.0, float(diff.std()) if len(diff) else 0.0, slope]


def _motion_features(data):
    motion = json.loads(data)
    features, names = [], []
    for joint in ("head", "left_hand", "right_hand", "torso"):
        sequence = np.asarray(motion[joint], dtype=float)
        if sequence.ndim != 2 or sequence.shape[1] != 3 or not len(sequence):
            raise ValueError(f"Unexpected sequence shape for {joint}: {sequence.shape}")
        for axis, axis_name in enumerate(("x", "y", "z")):
            values = sequence[:, axis]
            diffs = np.diff(values)
            features.extend([float(values.mean()), float(values.std()), float(values.min()), float(values.max()), float(np.mean(np.abs(diffs))) if len(diffs) else 0.0])
            names.extend([f"{joint}_{axis_name}_{stat}" for stat in ("mean", "std", "min", "max", "mean_abs_diff")])
    features.extend([float(motion["frame_rate"]), float(motion["duration_sec"]), float(bool(motion["stimming_detected"]))])
    names.extend(["frame_rate", "duration_sec", "stimming_detected"])
    return np.asarray(features), names


def _physiology_features(data):
    rows = list(csv.DictReader(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig")))
    times = [float(row["Time"]) for row in rows]
    features, names = [], []
    for column in ("HR", "GSR", "TEMP"):
        features.extend(_series_features([float(row[column]) for row in rows], times))
        names.extend([f"{column}_{stat}" for stat in ("mean", "std", "min", "max", "mean_abs_diff", "diff_std", "slope")])
    return np.asarray(features), names


def features_from_files(audio_path=None, motion_path=None, physiology_path=None):
    """Extract supported modalities from user-supplied files; unresolved images are not accepted."""
    features = {}
    if audio_path is not None:
        features["audio"] = (np.asarray(_wav_features(Path(audio_path).read_bytes())), ["rms_mean", "rms_std", "rms_max", "spectral_centroid", "spectral_entropy", "zero_crossing_rate", "silence_fraction", "clipping_fraction", "sample_rate", "duration_sec"])
    if motion_path is not None:
        features["motion"] = _motion_features(Path(motion_path).read_bytes())
    if physiology_path is not None:
        features["physiology"] = _physiology_features(Path(physiology_path).read_bytes())
    return features


def multimodal_features(sample_id):
    """Read only ID-addressed audio, motion and physiology; never guesses image links."""
    with zipfile.ZipFile(MULTI_ZIP) as archive:
        voice = _wav_features(archive.read(f"voice/{sample_id}.wav"))
        motion = _motion_features(archive.read(f"motion/{sample_id}.json"))
        physio = _physiology_features(archive.read(f"physio/{sample_id}.csv"))
        return {"audio": (np.asarray(voice), ["rms_mean", "rms_std", "rms_max", "spectral_centroid", "spectral_entropy", "zero_crossing_rate", "silence_fraction", "clipping_fraction", "sample_rate", "duration_sec"]),
                "motion": motion, "physiology": physio}
