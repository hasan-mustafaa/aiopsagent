"""Fault injector for simulated services.

Supports: crash, latency_spike, connection_failure, memory_leak, oom.
Faults cascade upstream through the dependency graph.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.simulator.log_generator import LogGenerator
from src.simulator.metrics_generator import MetricsGenerator


FAULT_TYPES = frozenset(
    {"crash", "latency_spike", "connection_failure", "memory_leak", "oom"}
)


@dataclass
class FaultScenario:
    """Describes a single injected fault event."""

    fault_id: str                      # UUID for tracking
    service: str                       # Target service name
    fault_type: str                    # One of FAULT_TYPES
    started_at: datetime
    duration_seconds: float            # How long the fault persists; -1 = until cleared
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Return True if the fault is still ongoing."""
        return self.resolved_at is None


class FaultInjector:
    """
    Orchestrates fault scenario injection into LogGenerator and MetricsGenerator.

    Runs as a background thread (or async task) alongside the generators.
    On each tick, it checks whether a new fault should be triggered, updates
    active faults, and clears faults that have expired.
    """

    def __init__(
        self,
        simulator_config: dict[str, Any],
        log_generator: LogGenerator,
        metrics_generator: MetricsGenerator,
    ) -> None:
        """
        Initialise the injector with references to the active generators.

        Args:
            simulator_config:  [simulator] section of config.yaml.
            log_generator:     Running LogGenerator instance to inject error logs into.
            metrics_generator: Running MetricsGenerator to apply fault profiles to.
        """
        self._log_generator = log_generator
        self._metrics_generator = metrics_generator
        self._services: frozenset[str] = frozenset(
            s["name"] for s in simulator_config["services"]
        )
        # _active: faults currently in effect — shrinks as faults expire or are cleared.
        # _history: append-only log of every FaultScenario ever created.
        self._active: list[FaultScenario] = []
        self._history: list[FaultScenario] = []

    def inject(
        self,
        service: str,
        fault_type: str,
        duration_seconds: float = 60.0,
    ) -> FaultScenario:
        """
        Activate a fault scenario immediately.

        Args:
            service:          Name of the target service.
            fault_type:       One of FAULT_TYPES.
            duration_seconds: How long the fault should persist. Pass -1 for
                              indefinite (must be cleared manually).

        Returns:
            The created FaultScenario.

        Raises:
            ValueError: If service or fault_type is invalid.
        """
        self._validate_fault(service, fault_type)

        scenario = FaultScenario(
            fault_id=str(uuid.uuid4()),
            service=service,
            fault_type=fault_type,
            started_at=datetime.now(),
            duration_seconds=duration_seconds,
        )

        # Both generators read service_states / active_baselines on every tick,
        # so they must be updated together — split state would produce one tick
        # where logs show a fault but metrics still look healthy (or vice versa),
        # which would corrupt the training signal for the detection layer.
        self._log_generator.service_states[service] = {
            "healthy": False,
            "fault_type": fault_type,
        }
        self._metrics_generator.apply_fault_profile(service, fault_type)

        self._active.append(scenario)
        self._history.append(scenario)
        return scenario

    def clear(self, service: str, fault_type: str | None = None) -> None:
        """
        Deactivate all active faults on a service (or a specific fault type).

        Args:
            service:    Target service name.
            fault_type: If provided, only clear faults of this type;
                        otherwise clear all faults on the service.
        """
        to_resolve = [
            s for s in self._active
            if s.service == service
            and (fault_type is None or s.fault_type == fault_type)
        ]
        for scenario in to_resolve:
            self._resolve_fault(scenario)

    def clear_all(self) -> None:
        """Deactivate every active fault across all services."""
        for scenario in list(self._active):  # copy — _resolve_fault mutates _active
            self._resolve_fault(scenario)

    def tick(self) -> None:
        """
        Advance the injector by one time step.

        - Checks if any active faults have expired and resolves them.
        - Randomly decides whether to inject a new fault (based on config
          fault probability settings).

        Called by the main pipeline loop on each simulation tick.
        """
        now = datetime.now()
        # Snapshot _active before iterating — _resolve_fault mutates the list.
        expired = [
            s for s in list(self._active)
            if s.duration_seconds != -1
            and (now - s.started_at).total_seconds() >= s.duration_seconds
        ]
        for scenario in expired:
            self._resolve_fault(scenario)

    def active_faults(self) -> list[FaultScenario]:
        """Return a list of all currently active FaultScenario objects."""
        return list(self._active)

    def fault_history(self) -> list[FaultScenario]:
        """Return the full ordered history of all faults (active + resolved)."""
        return list(self._history)

    def _resolve_fault(self, scenario: FaultScenario) -> None:
        """
        Mark a fault as resolved and restore normal operation.

        Args:
            scenario: The FaultScenario to resolve.
        """
        scenario.resolved_at = datetime.now()
        self._active = [s for s in self._active if s.fault_id != scenario.fault_id]

        # Only restore healthy state if no other faults are still active on this service.
        # Two simultaneous faults (e.g. latency_spike + memory_leak) should keep
        # the service in fault mode until the last one is resolved.
        remaining = [s for s in self._active if s.service == scenario.service]
        if not remaining:
            self._log_generator.service_states[scenario.service] = {
                "healthy": True,
                "fault_type": None,
            }
            self._metrics_generator.restore_baseline(scenario.service)

    def _validate_fault(self, service: str, fault_type: str) -> None:
        """
        Raise ValueError if the service or fault_type is not recognised.

        Args:
            service:    Service name to validate.
            fault_type: Fault type to validate.
        """
        if service not in self._services:
            raise ValueError(
                f"Unknown service {service!r}. Valid services: {sorted(self._services)}"
            )
        if fault_type not in FAULT_TYPES:
            raise ValueError(
                f"Unknown fault type {fault_type!r}. Valid types: {sorted(FAULT_TYPES)}"
            )
