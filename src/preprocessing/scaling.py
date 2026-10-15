"""Feature standardization fitted only on the supplied training partition."""
from __future__ import annotations
import numpy as np


class Standardizer:
    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def to_dict(self):
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}
