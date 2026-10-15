"""Reproducible per-class train/validation/test assignments."""
from __future__ import annotations
import hashlib
from src.config import RANDOM_SEED, TRAIN_FRACTION, VALIDATION_FRACTION


def stratified_assignments(records, label_key, id_key, seed=RANDOM_SEED):
    identifiers = [row[id_key] for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Split groups must be unique; aggregate repeated records before splitting")
    result = {}
    for label in sorted({row[label_key] for row in records}):
        group = sorted((row for row in records if row[label_key] == label), key=lambda row: hashlib.sha256(f"{seed}:{row[id_key]}".encode()).digest())
        n_train, n_validation = round(TRAIN_FRACTION * len(group)), round(VALIDATION_FRACTION * len(group))
        for position, row in enumerate(group):
            result[row[id_key]] = "train" if position < n_train else "validation" if position < n_train + n_validation else "test"
    return result


def stratified_folds(records, label_key, id_key, folds=5, seed=RANDOM_SEED):
    """Return deterministic stratified fold IDs; every group appears in one fold."""
    identifiers = [row[id_key] for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Fold groups must be unique")
    result = {}
    for label in sorted({row[label_key] for row in records}):
        group = sorted((row for row in records if row[label_key] == label), key=lambda row: hashlib.sha256(f"{seed}:{row[id_key]}".encode()).digest())
        for position, row in enumerate(group):
            result[row[id_key]] = position % folds
    return result
