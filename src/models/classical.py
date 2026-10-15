"""Minimal deterministic models to keep the project runnable without scikit-learn."""
from __future__ import annotations

import numpy as np
from src.preprocessing import Standardizer


def softmax(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


class LogisticRegression:
    """Batch gradient descent logistic model (binary or multinomial)."""
    def __init__(self, regularization=0.1, learning_rate=0.15, epochs=1600):
        self.regularization = float(regularization)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)

    def fit(self, x, y, validation_data=None):
        x, y = np.asarray(x, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.weights_ = np.zeros((x.shape[1] + 1, len(self.classes_)))
        xb = np.column_stack([np.ones(len(x)), x])
        yi = np.searchsorted(self.classes_, y)
        target = np.eye(len(self.classes_))[yi]
        history = []

        def loss(features, labels):
            augmented = np.column_stack([np.ones(len(features)), features])
            indices = np.searchsorted(self.classes_, labels)
            probabilities = softmax(augmented @ self.weights_)
            cross_entropy = -np.log(np.maximum(probabilities[np.arange(len(labels)), indices], 1e-15)).mean()
            return float(cross_entropy + .5 * self.regularization * np.sum(self.weights_[1:] ** 2))

        history.append({"epoch": 0, "train_loss": loss(x, y), "validation_loss": loss(*validation_data) if validation_data else None})
        for epoch in range(1, self.epochs + 1):
            probs = softmax(xb @ self.weights_)
            grad = xb.T @ (probs - target) / len(x)
            grad[1:] += self.regularization * self.weights_[1:]
            self.weights_ -= self.learning_rate * grad
            if epoch % 50 == 0 or epoch == self.epochs:
                history.append({"epoch": epoch, "train_loss": loss(x, y), "validation_loss": loss(*validation_data) if validation_data else None})
        self.training_history_ = history
        return self

    def predict_proba(self, x):
        xb = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
        return softmax(xb @ self.weights_)

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def to_dict(self, scaler, feature_names):
        return {"classes": self.classes_.tolist(), "weights": self.weights_.tolist(), "regularization": self.regularization,
                "feature_names": list(feature_names), "scaler": scaler.to_dict(), "training_history": self.training_history_}


class DecisionTree:
    """Small CART classifier with Gini splits, numeric thresholds and no pruning beyond depth/min-leaf."""
    def __init__(self, max_depth=5, min_leaf=2):
        self.max_depth, self.min_leaf = int(max_depth), int(min_leaf)

    @staticmethod
    def _impurity(y):
        if len(y) == 0: return 0.0
        counts = np.bincount(y)
        p = counts[counts > 0] / len(y)
        return float(1 - np.sum(p * p))

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.n_features_ = x.shape[1]
        self.tree_ = self._grow(np.asarray(x, float), yi, 0)
        return self

    def _grow(self, x, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = {"counts": counts.tolist(), "prediction": int(np.argmax(counts))}
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_leaf * 2:
            return node
        base = self._impurity(y); best = None
        for feature in range(x.shape[1]):
            unique = np.unique(x[:, feature])
            if len(unique) < 2: continue
            thresholds = (unique[:-1] + unique[1:]) / 2
            for threshold in thresholds:
                left = x[:, feature] <= threshold
                nl, nr = int(left.sum()), int((~left).sum())
                if min(nl, nr) < self.min_leaf: continue
                gain = base - (nl * self._impurity(y[left]) + nr * self._impurity(y[~left])) / len(y)
                if best is None or gain > best[0]: best = (gain, feature, float(threshold), left)
        if best is None or best[0] <= 1e-12: return node
        _, feature, threshold, left = best
        node.update(feature=int(feature), threshold=threshold,
                    left=self._grow(x[left], y[left], depth + 1), right=self._grow(x[~left], y[~left], depth + 1))
        return node

    def predict(self, x):
        out=[]
        for row in np.asarray(x, float):
            node=self.tree_
            while "feature" in node:
                node=node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            out.append(self.classes_[node["prediction"]])
        return np.asarray(out)

    def to_dict(self, feature_names):
        return {"classes": self.classes_.tolist(), "tree": self.tree_, "feature_names": list(feature_names), "max_depth": self.max_depth}
