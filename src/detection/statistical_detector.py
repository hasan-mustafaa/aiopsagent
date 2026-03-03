"""Z-score anomaly detection on metric time series.

Fast, unsupervised detection via rolling windows. Combines with ML detector
for ensemble scoring.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulator.metrics_generator import MetricSnapshot


@dataclass
class StatDetectionResult:
    """Result of a single statistical anomaly check.

    Attributes:
        service:           Service that was checked.
        is_anomaly:        True if any metric exceeded the z-score threshold.
        anomaly_score:     Normalized score in [0, 1] (sigmoid of max |z|).
        triggered_metrics: Names of metrics that exceeded the threshold.
        z_scores:          Per-metric z-scores for diagnostics.
        snapshot:          The MetricSnapshot that was evaluated.
    """

    service: str
    is_anomaly: bool
    anomaly_score: float
    triggered_metrics: list[str]
    z_scores: dict[str, float]
    snapshot: MetricSnapshot | None = None


class StatisticalDetector:
    """Rolling z-score detector with per-service, per-metric windows.

    Returns a neutral score during cold start (< MIN_WINDOW_SAMPLES).
    After warm-up, flags metrics where |z| > z_score_threshold.
    """

    MONITORED_METRICS: tuple[str, ...] = (
        "cpu_percent",
        "memory_mb",
        "latency_ms",
        "error_rate",
        "active_connections",
    )

    MIN_WINDOW_SAMPLES: int = 5  # Minimum samples before producing real scores

    def __init__(self, detection_config: dict[str, Any]) -> None:
        """Initialize rolling windows and threshold from config."""
        self._z_threshold: float = detection_config.get("z_score_threshold", 3.0)
        self._window_size: int = detection_config.get("window_size_seconds", 60)

        # service → metric_name → deque of recent values
        self._windows: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: {m: deque(maxlen=self._window_size) for m in self.MONITORED_METRICS}
        )

    def update(self, snapshot: MetricSnapshot) -> StatDetectionResult:
        """Ingest a MetricSnapshot, compute z-scores, return anomaly verdict.

        During cold start (< MIN_WINDOW_SAMPLES), returns neutral score ~0.
        """
        service = snapshot.service
        windows = self._windows[service]

        # Extract current values from the snapshot
        values = self._extract_metrics(snapshot)

        # Compute z-scores against current window state (before adding new values)
        z_scores: dict[str, float] = {}
        for metric in self.MONITORED_METRICS:
            window = windows[metric]
            if len(window) < self.MIN_WINDOW_SAMPLES:
                z_scores[metric] = 0.0
            else:
                mean = float(np.mean(window))
                std = float(np.std(window))
                z_scores[metric] = self._compute_z_score(values[metric], mean, std)

        # Append new values to windows after scoring
        for metric in self.MONITORED_METRICS:
            windows[metric].append(values[metric])

        # Determine anomaly status
        triggered = [
            m for m in self.MONITORED_METRICS
            if abs(z_scores[m]) > self._z_threshold
        ]
        max_abs_z = max(abs(z) for z in z_scores.values())
        score = self._normalise_score(max_abs_z)

        return StatDetectionResult(
            service=service,
            is_anomaly=len(triggered) > 0,
            anomaly_score=score,
            triggered_metrics=triggered,
            z_scores=z_scores,
            snapshot=snapshot,
        )

    def get_rolling_stats(
        self, service: str, metric: str
    ) -> tuple[float, float] | None:
        """Return (mean, std) for a service's metric window, or None if empty."""
        if service not in self._windows:
            return None
        window = self._windows[service].get(metric)
        if window is None or len(window) == 0:
            return None
        return float(np.mean(window)), float(np.std(window))

    def reset(self, service: str | None = None) -> None:
        """Clear rolling windows for one or all services."""
        if service is not None:
            self._windows.pop(service, None)
        else:
            self._windows.clear()

    def _extract_metrics(self, snapshot: MetricSnapshot) -> dict[str, float]:
        """Pull monitored metric values from a snapshot."""
        return {
            "cpu_percent": snapshot.cpu_percent,
            "memory_mb": snapshot.memory_mb,
            "latency_ms": snapshot.latency_ms,
            "error_rate": snapshot.error_rate,
            "active_connections": float(snapshot.active_connections),
        }

    def _compute_z_score(self, value: float, mean: float, std: float) -> float:
        """Compute z-score. If std ≈ 0, any deviation from mean returns ±10.0."""
        if std < 1e-10:
            # Window is constant — any change is significant
            if abs(value - mean) < 1e-10:
                return 0.0
            return 10.0 if value > mean else -10.0
        return (value - mean) / std

    def _normalise_score(self, max_abs_z: float) -> float:
        """Map max |z-score| to [0, 1] via sigmoid centered at the threshold.

        At z = threshold → score ≈ 0.5. At z >> threshold → score → 1.0.
        At z = 0 → score ≈ 0.05 (near zero).
        """
        return 1.0 / (1.0 + math.exp(-(max_abs_z - self._z_threshold)))
