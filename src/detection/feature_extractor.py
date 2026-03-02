"""Sliding-window feature extraction for anomaly detection.

Aggregates ParsedLog and MetricSnapshot streams into fixed-size numerical
feature vectors (one per service per window). These vectors feed both the
StatisticalDetector and MLAnomalyDetector. Also handles ensemble score
fusion from the two detector outputs.

Feature vector layout (15 dimensions):
  [0-5]   Log-derived:  counts, error rate, template diversity
  [6-14]  Metric-derived: CPU, memory, latency, error rate, throughput stats
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.detection.log_parser import ParsedLog
from src.simulator.metrics_generator import MetricSnapshot

# Ordered feature names — indices must match the numpy array in FeatureVector.
FEATURE_NAMES: list[str] = [
    # Log-derived
    "log_count_total",
    "log_count_error",
    "log_count_warning",
    "error_rate",
    "unique_templates",
    "top_template_frequency",
    # Metric-derived
    "cpu_percent_mean",
    "cpu_percent_std",
    "memory_mb_mean",
    "memory_mb_std",
    "latency_ms_mean",
    "latency_ms_p99",
    "metric_error_rate_mean",
    "request_rate_mean",
    "active_connections_mean",
]


@dataclass
class FeatureVector:
    """Fixed-size feature vector for one service over one time window.

    Attributes:
        service:        Service this vector describes.
        window_start:   Start of the aggregation window.
        window_end:     End of the aggregation window.
        features:       Numpy array of shape (FEATURE_DIM,).
        feature_names:  Ordered names matching features array indices.
        ensemble_score: Combined anomaly score from detectors (0-1).
        is_anomaly:     Final anomaly verdict after ensemble scoring.
    """

    service: str
    window_start: datetime
    window_end: datetime
    features: np.ndarray
    feature_names: list[str]
    ensemble_score: float = 0.0
    is_anomaly: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "service": self.service,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "features": self.features.tolist(),
            "feature_names": self.feature_names,
            "ensemble_score": self.ensemble_score,
            "is_anomaly": self.is_anomaly,
            "metadata": self.metadata,
        }


class FeatureExtractor:
    """Sliding-window feature aggregation with ensemble score fusion.

    Buffers ParsedLog and MetricSnapshot objects per service, then builds
    a 15-dimensional feature vector on demand. Also computes the weighted
    ensemble score from statistical and ML detector outputs.
    """

    FEATURE_DIM: int = 15  # Must match len(FEATURE_NAMES)
    ANOMALY_THRESHOLD: float = 0.5  # Ensemble score above this → anomaly

    def __init__(self, detection_config: dict[str, Any]) -> None:
        """Initialize window buffers and ensemble weights from config."""
        self._window_seconds: int = detection_config.get("window_size_seconds", 60)
        weights = detection_config.get("ensemble_weights", {})
        self._stat_weight: float = weights.get("statistical", 0.4)
        self._ml_weight: float = weights.get("ml", 0.6)

        # Per-service sliding window buffers
        self._log_buffer: dict[str, list[ParsedLog]] = defaultdict(list)
        self._metric_buffer: dict[str, list[MetricSnapshot]] = defaultdict(list)

    def ingest_log(self, parsed_log: ParsedLog) -> None:
        """Buffer a parsed log entry for its service's current window."""
        self._log_buffer[parsed_log.service].append(parsed_log)

    def ingest_metric(self, snapshot: MetricSnapshot) -> None:
        """Buffer a metric snapshot for its service's current window."""
        self._metric_buffer[snapshot.service].append(snapshot)

    def extract(self, service: str) -> FeatureVector | None:
        """Build a feature vector from the current window for a service.

        Returns None if no logs or metrics have been ingested yet.
        """
        logs = self._log_buffer.get(service, [])
        metrics = self._metric_buffer.get(service, [])

        if not logs and not metrics:
            return None

        log_feats = self._build_log_features(service)
        metric_feats = self._build_metric_features(service)

        # Assemble feature array in FEATURE_NAMES order
        features = np.array(
            [log_feats.get(name, 0.0) for name in FEATURE_NAMES[:6]]
            + [metric_feats.get(name, 0.0) for name in FEATURE_NAMES[6:]],
            dtype=np.float64,
        )

        now = datetime.now()
        window_start = now - timedelta(seconds=self._window_seconds)

        return FeatureVector(
            service=service,
            window_start=window_start,
            window_end=now,
            features=features,
            feature_names=list(FEATURE_NAMES),
        )

    def flush_window(self, service: str) -> None:
        """Discard buffered data older than window_size_seconds for a service."""
        cutoff = datetime.now() - timedelta(seconds=self._window_seconds)

        if service in self._log_buffer:
            self._log_buffer[service] = [
                log for log in self._log_buffer[service]
                if log.original.timestamp >= cutoff
            ]
        if service in self._metric_buffer:
            self._metric_buffer[service] = [
                m for m in self._metric_buffer[service]
                if m.timestamp >= cutoff
            ]

    def compute_ensemble_score(
        self,
        stat_score: float,
        ml_score: float,
    ) -> tuple[float, bool]:
        """Weighted average of statistical and ML anomaly scores.

        Returns (ensemble_score, is_anomaly) where is_anomaly is True
        when the score exceeds ANOMALY_THRESHOLD.
        """
        score = self._stat_weight * stat_score + self._ml_weight * ml_score
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        return score, score >= self.ANOMALY_THRESHOLD

    def _build_log_features(self, service: str) -> dict[str, float]:
        """Compute log-derived features from the current window buffer."""
        logs = self._log_buffer.get(service, [])

        total = len(logs)
        if total == 0:
            return {name: 0.0 for name in FEATURE_NAMES[:6]}

        error_count = sum(1 for log in logs if log.level == "ERROR")
        warning_count = sum(1 for log in logs if log.level == "WARNING")

        # Template diversity
        cluster_ids = [log.cluster_id for log in logs]
        unique_templates = len(set(cluster_ids))
        cluster_counts = Counter(cluster_ids)
        top_template_freq = max(cluster_counts.values()) / total

        return {
            "log_count_total": float(total),
            "log_count_error": float(error_count),
            "log_count_warning": float(warning_count),
            "error_rate": error_count / total,
            "unique_templates": float(unique_templates),
            "top_template_frequency": top_template_freq,
        }

    def _build_metric_features(self, service: str) -> dict[str, float]:
        """Compute metric-derived features from the current window buffer."""
        metrics = self._metric_buffer.get(service, [])

        if not metrics:
            return {name: 0.0 for name in FEATURE_NAMES[6:]}

        cpu = np.array([m.cpu_percent for m in metrics])
        mem = np.array([m.memory_mb for m in metrics])
        lat = np.array([m.latency_ms for m in metrics])
        err = np.array([m.error_rate for m in metrics])
        req = np.array([m.request_rate for m in metrics])
        conn = np.array([float(m.active_connections) for m in metrics])

        return {
            "cpu_percent_mean": float(np.mean(cpu)),
            "cpu_percent_std": float(np.std(cpu)) if len(cpu) > 1 else 0.0,
            "memory_mb_mean": float(np.mean(mem)),
            "memory_mb_std": float(np.std(mem)) if len(mem) > 1 else 0.0,
            "latency_ms_mean": float(np.mean(lat)),
            "latency_ms_p99": float(np.percentile(lat, 99)) if len(lat) > 1 else float(lat[0]),
            "metric_error_rate_mean": float(np.mean(err)),
            "request_rate_mean": float(np.mean(req)),
            "active_connections_mean": float(np.mean(conn)),
        }
