"""Parameter-light early and late fusion helpers."""
from __future__ import annotations
import numpy as np

MODALITY_ORDER = ("audio", "motion", "physiology")


def early_concat(blocks):
    available = [name for name in MODALITY_ORDER if name in blocks]
    if available != list(MODALITY_ORDER):
        raise ValueError("Early fusion requires audio, motion, and physiology blocks")
    row_counts = {np.asarray(blocks[name][0]).shape[0] for name in available}
    if len(row_counts) != 1: raise ValueError("Feature blocks must have equal sample counts")
    matrix = np.column_stack([np.asarray(blocks[name][0], dtype=float) for name in available])
    names = [f"{name}__{feature}" for name in available for feature in blocks[name][1]]
    if matrix.shape[1] != len(names): raise ValueError("Feature names do not match concatenated dimensions")
    return matrix, names


def late_average(probabilities):
    """Equal-weight average over available modality probability matrices."""
    if isinstance(probabilities, dict): probabilities = list(probabilities.values())
    if not probabilities: raise ValueError("Late fusion needs at least one modality branch")
    arrays = [np.asarray(value, dtype=float) for value in probabilities]
    if any(value.ndim != 2 for value in arrays) or len({value.shape for value in arrays}) != 1:
        raise ValueError("Branch probability matrices must have matching 2D shapes")
    result = np.mean(arrays, axis=0)
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0): raise ValueError("Invalid branch probabilities")
    return result / totals
