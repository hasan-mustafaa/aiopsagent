"""Time-series metrics generator for FCT services.

Produces realistic metrics (cpu, memory, latency, errors, requests, connections)
modelled as Gaussian random walks. FaultInjector calls apply_fault_profile() to
shift baselines during active faults.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator


# Fault multipliers applied to default baselines (not current values)
# to maintain proportional anomaly magnitude across services.

_FAULT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "crash": {
        # Process dead: resource counters collapse to near-zero; observed
        # latency spikes because upstream callers block until timeout.
        "cpu_mean": 0.05,     "cpu_std": 0.5,
        "memory_mean": 0.1,   "memory_std": 0.5,
        "latency_mean": 50.0, "latency_std": 5.0,
        "error_rate_mean": 80.0, "error_rate_std": 10.0,
        "request_rate_mean": 0.0, "request_rate_std": 0.0,
        "connections_mean": 0.0,  "connections_std": 0.0,
    },
    "latency_spike": {
        # Process healthy but saturated: 15× latency, connections back up,
        # timeout errors climb proportionally.
        "cpu_mean": 1.2,      "cpu_std": 1.5,
        "memory_mean": 1.0,   "memory_std": 1.0,
        "latency_mean": 15.0, "latency_std": 3.0,
        "error_rate_mean": 10.0, "error_rate_std": 3.0,
        "request_rate_mean": 0.6, "request_rate_std": 0.8,
        "connections_mean": 1.5,  "connections_std": 2.0,
    },
    "connection_failure": {
        # Listener down: request rate collapses, callers get ECONNREFUSED
        # and time out, driving error_rate sharply up.
        "cpu_mean": 0.3,      "cpu_std": 0.5,
        "memory_mean": 1.0,   "memory_std": 1.0,
        "latency_mean": 20.0, "latency_std": 4.0,
        "error_rate_mean": 50.0, "error_rate_std": 8.0,
        "request_rate_mean": 0.1, "request_rate_std": 0.2,
        "connections_mean": 0.05, "connections_std": 0.1,
    },
    "memory_leak": {
        # GC pressure from unbounded heap growth: elevated CPU and latency
        # variance, with occasional stop-the-world pause errors.
        "cpu_mean": 1.4,      "cpu_std": 1.5,
        "memory_mean": 3.5,   "memory_std": 3.0,
        "latency_mean": 2.0,  "latency_std": 2.0,
        "error_rate_mean": 3.0,  "error_rate_std": 2.0,
        "request_rate_mean": 0.9, "request_rate_std": 0.9,
        "connections_mean": 1.0,  "connections_std": 1.0,
    },
    "oom": {
        # OOM killer active: memory pegged at ceiling, CPU thrashing,
        # throughput near zero as the process fights for pages.
        "cpu_mean": 2.0,      "cpu_std": 0.5,
        "memory_mean": 5.0,   "memory_std": 0.2,
        "latency_mean": 5.0,  "latency_std": 2.0,
        "error_rate_mean": 30.0, "error_rate_std": 5.0,
        "request_rate_mean": 0.1, "request_rate_std": 0.1,
        "connections_mean": 0.2,  "connections_std": 0.1,
    },
}


@dataclass
class MetricSnapshot:
    """A single point-in-time metrics snapshot for one service."""

    timestamp: datetime
    service: str
    cpu_percent: float
    memory_mb: float
    latency_ms: float
    error_rate: float
    request_rate: float
    active_connections: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "timestamp":          self.timestamp.isoformat(),
            "service":            self.service,
            "cpu_percent":        self.cpu_percent,
            "memory_mb":          self.memory_mb,
            "latency_ms":         self.latency_ms,
            "error_rate":         self.error_rate,
            "request_rate":       self.request_rate,
            "active_connections": self.active_connections,
        }


@dataclass
class ServiceBaseline:
    """
    Normal operating baseline for one service.

    All metrics are modelled as Gaussian: N(mean, std).
    FaultInjector temporarily replaces these with fault profiles.
    """

    cpu_mean: float
    cpu_std: float
    memory_mean: float
    memory_std: float
    latency_mean: float
    latency_std: float
    error_rate_mean: float
    error_rate_std: float
    request_rate_mean: float
    request_rate_std: float
    connections_mean: float
    connections_std: float


class MetricsGenerator:
    """
    Generates a continuous stream of MetricSnapshot objects for all services.

    Maintains per-service baselines and applies random walk noise to simulate
    realistic metric fluctuation. Baselines can be temporarily overridden by
    the FaultInjector to produce anomalous readings.
    """

    def __init__(self, simulator_config: dict[str, Any]) -> None:
        """
        Initialise the generator from the [simulator] section of config.yaml.

        Args:
            simulator_config: Dict containing 'services' list and
                              'metrics_interval_seconds'.
        """
        self._services: list[str] = [s["name"] for s in simulator_config["services"]]

        # _default_baselines is the immutable source of truth — never mutated.
        # _active_baselines is the live working copy; fault profiles write here,
        # restore_baseline() resets individual entries back to default.
        self._default_baselines: dict[str, ServiceBaseline] = self._build_default_baselines()
        self._active_baselines: dict[str, ServiceBaseline] = copy.copy(self._default_baselines)

        self._latest: dict[str, MetricSnapshot | None] = {s: None for s in self._services}

    def generate(self) -> Generator[MetricSnapshot, None, None]:
        """
        Yield one MetricSnapshot per service per interval tick.

        Yields:
            MetricSnapshot instances in service definition order.
        """
        while True:
            now = datetime.now()
            for service in self._services:
                b = self._active_baselines[service]
                snapshot = MetricSnapshot(
                    timestamp=now,
                    service=service,
                    # Physical upper bound — OS won't report > 100%.

                    cpu_percent=min(100.0, self._sample_metric(b.cpu_mean, b.cpu_std)),
                    memory_mb=self._sample_metric(b.memory_mean, b.memory_std, min_val=1.0),
                    latency_ms=self._sample_metric(b.latency_mean, b.latency_std),
                    error_rate=self._sample_metric(b.error_rate_mean, b.error_rate_std),
                    request_rate=self._sample_metric(b.request_rate_mean, b.request_rate_std),
                    active_connections=int(
                        self._sample_metric(b.connections_mean, b.connections_std)
                    ),
                )
                self._latest[service] = snapshot
                yield snapshot

    def apply_fault_profile(self, service: str, fault_type: str) -> None:
        """
        Override the baseline for a service to simulate anomalous metrics.

        Called by FaultInjector when a fault is activated. The overridden
        baseline persists until `restore_baseline()` is called.

        Args:
            service:    Name of the affected service.
            fault_type: One of: 'crash', 'latency_spike', 'connection_failure',
                        'memory_leak', 'oom'.
        """
        d = self._default_baselines[service]  # always multiply from default, not from current
        m = _FAULT_MULTIPLIERS[fault_type]
        self._active_baselines[service] = ServiceBaseline(
            cpu_mean=d.cpu_mean * m["cpu_mean"],
            cpu_std=d.cpu_std                 * m["cpu_std"],
            memory_mean=d.memory_mean         * m["memory_mean"],
            memory_std=d.memory_std           * m["memory_std"],
            latency_mean=d.latency_mean       * m["latency_mean"],
            latency_std=d.latency_std         * m["latency_std"],
            error_rate_mean=d.error_rate_mean * m["error_rate_mean"],
            error_rate_std=d.error_rate_std   * m["error_rate_std"],
            request_rate_mean=d.request_rate_mean * m["request_rate_mean"],
            request_rate_std=d.request_rate_std   * m["request_rate_std"],
            connections_mean=d.connections_mean   * m["connections_mean"],
            connections_std=d.connections_std     * m["connections_std"],
        )

    def restore_baseline(self, service: str) -> None:
        """
        Restore a service's metrics baseline to its normal operating values.

        Args:
            service: Name of the service to restore.
        """
        self._active_baselines[service] = self._default_baselines[service]

    def get_latest_snapshot(self, service: str) -> MetricSnapshot | None:
        """
        Return the most recently generated snapshot for a service.

        Args:
            service: Service name.

        Returns:
            Latest MetricSnapshot, or None if no snapshot has been generated yet.
        """
        return self._latest.get(service)

    def _sample_metric(self, mean: float, std: float, min_val: float = 0.0) -> float:
        """
        Sample a metric value from N(mean, std), clipped at min_val.

        Args:
            mean:    Distribution mean.
            std:     Distribution standard deviation.
            min_val: Lower bound for the sampled value.

        Returns:
            Sampled metric value.
        """
        return max(min_val, random.gauss(mean, std))

    def _build_default_baselines(self) -> dict[str, ServiceBaseline]:
        """
        Construct hardcoded normal baselines for each FCT service.

        Returns:
            Dict mapping service name to its ServiceBaseline.
        """
        # Values reflect each service's role in the FCT stack:
        #   transaction-validator  — upstream orchestrator, high throughput, moderate CPU
        #   fraud-check-service    — runs ML models, higher CPU and memory
        #   document-processor     — OCR workloads, highest memory, higher latency
        #   title-search-service   — cached leaf, lowest latency and CPU, highest request rate
        return {
            "transaction-validator": ServiceBaseline(
                cpu_mean=35.0,          cpu_std=8.0,
                memory_mean=512.0,      memory_std=50.0,
                latency_mean=80.0,      latency_std=15.0,
                error_rate_mean=0.02,   error_rate_std=0.005,
                request_rate_mean=150.0, request_rate_std=20.0,
                connections_mean=50.0,  connections_std=10.0,
            ),
            "fraud-check-service": ServiceBaseline(
                cpu_mean=55.0,          cpu_std=12.0,
                memory_mean=1024.0,     memory_std=100.0,
                latency_mean=120.0,     latency_std=25.0,
                error_rate_mean=0.01,   error_rate_std=0.003,
                request_rate_mean=100.0, request_rate_std=15.0,
                connections_mean=30.0,  connections_std=8.0,
            ),
            "document-processor": ServiceBaseline(
                cpu_mean=45.0,          cpu_std=15.0,
                memory_mean=1536.0,     memory_std=200.0,
                latency_mean=200.0,     latency_std=50.0,
                error_rate_mean=0.015,  error_rate_std=0.004,
                request_rate_mean=80.0, request_rate_std=12.0,
                connections_mean=25.0,  connections_std=7.0,
            ),
            "title-search-service": ServiceBaseline(
                cpu_mean=20.0,          cpu_std=5.0,
                memory_mean=768.0,      memory_std=80.0,
                latency_mean=15.0,      latency_std=5.0,
                error_rate_mean=0.005,  error_rate_std=0.002,
                request_rate_mean=200.0, request_rate_std=30.0,
                connections_mean=80.0,  connections_std=15.0,
            ),
        }
