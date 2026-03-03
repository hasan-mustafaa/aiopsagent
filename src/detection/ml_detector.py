"""Isolation Forest detector for multivariate anomalies.

Per-service unsupervised training on warm-up data. Normalizes scores for
ensemble combination with statistical detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.detection.feature_extractor import FeatureVector


@dataclass
class MLDetectionResult:
    """Result of a single Isolation Forest anomaly check.

    Attributes:
        service:        Service that was scored.
        is_anomaly:     True if the model considers this observation anomalous.
        anomaly_score:  Normalized to [0, 1]; higher = more anomalous.
        raw_if_score:   Raw sklearn decision_function output (negative = anomalous).
        feature_vector: The input FeatureVector that was scored.
        model_trained:  False during warm-up (score is unreliable).
    """

    service: str
    is_anomaly: bool
    anomaly_score: float
    raw_if_score: float
    feature_vector: FeatureVector | None = None
    model_trained: bool = False


# Minimum samples to train a meaningful Isolation Forest
MIN_TRAINING_SAMPLES: int = 10


class MLAnomalyDetector:
    """Per-service Isolation Forest detector with warm-up handling.

    Each service gets its own trained model so that normal baselines are
    learnt independently. Untrained services return neutral results.
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100) -> None:
        """Initialize IF hyperparameters. Models are created per-service at train time."""
        self._contamination = contamination
        self._n_estimators = n_estimators
        # service → trained IsolationForest
        self._models: dict[str, IsolationForest] = {}

    def train(self, features: np.ndarray, service: str) -> None:
        """Train an Isolation Forest for a service on normal operational data.

        Raises ValueError if features has fewer than MIN_TRAINING_SAMPLES rows.
        """
        if len(features) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_TRAINING_SAMPLES} samples to train, "
                f"got {len(features)}"
            )

        model = IsolationForest(
            contamination=self._contamination,
            n_estimators=self._n_estimators,
            random_state=42,
        )
        model.fit(features)
        self._models[service] = model

    def detect(self, feature_vector: FeatureVector) -> MLDetectionResult:
        """Score a FeatureVector. Returns neutral result if model isn't trained yet."""
        service = feature_vector.service
        model = self._models.get(service)

        if model is None:
            return MLDetectionResult(
                service=service,
                is_anomaly=False,
                anomaly_score=0.0,
                raw_if_score=0.0,
                feature_vector=feature_vector,
                model_trained=False,
            )

        X = feature_vector.features.reshape(1, -1)
        raw_score = float(model.decision_function(X)[0])
        prediction = int(model.predict(X)[0])  # 1 = inlier, -1 = outlier

        return MLDetectionResult(
            service=service,
            is_anomaly=prediction == -1,
            anomaly_score=self._normalise_score(raw_score),
            raw_if_score=raw_score,
            feature_vector=feature_vector,
            model_trained=True,
        )

    def detect_batch(
        self, feature_vectors: list[FeatureVector]
    ) -> list[MLDetectionResult]:
        """Score multiple FeatureVectors. All must belong to the same service."""
        return [self.detect(fv) for fv in feature_vectors]

    def is_trained(self, service: str) -> bool:
        """Check if a trained model exists for this service."""
        return service in self._models

    def get_model(self, service: str) -> Optional[IsolationForest]:
        """Return the raw sklearn model for a service, or None if untrained."""
        return self._models.get(service)

    def _normalise_score(self, raw_score: float) -> float:
        """Map IF decision_function output to [0, 1].

        sklearn IF returns negative scores for anomalies, positive for inliers.
        We negate and pass through a sigmoid so that anomalies → high scores.
        """
        # Sigmoid of negated score: anomalies (negative raw) → high output
        return 1.0 / (1.0 + np.exp(raw_score))
