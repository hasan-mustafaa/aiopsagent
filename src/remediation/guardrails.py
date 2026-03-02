"""
Structured log generator for simulated FCT microservices.

Generates realistic, structured log entries for each service defined in
config.yaml. Each log entry contains: timestamp, level, service, message.

Log levels and their approximate emission rates during normal operation:
  - INFO:     80%
  - WARNING:  15%
  - ERROR:     5%

Services simulated:
  - transaction-validator  (upstream orchestrator)
  - fraud-check-service    (depends on title-search-service)
  - document-processor     (depends on title-search-service)
  - title-search-service   (leaf dependency, no upstream deps)

The generator checks a shared service_states dict each tick to determine
whether to emit healthy or fault-mode logs, including cascade errors for
services whose dependencies are faulted.

# ──────────────────────────────────────────────────────────────
# REMOVED FOR MVP (add back if time permits / mention in ASSUMPTIONS.md):
#
# - trace_id (UUID4 per request):
#     Enables correlating a single request across all four services.
#     Detection layer works on aggregate counts, not individual traces,
#     so not needed for the demo. Would be valuable for distributed
#     tracing visualization in a production system.
#
# - metadata dict on LogEntry:
#     Extra structured fields (e.g. latency_ms, property_id, risk_score)
#     attached to each log. Useful for richer analysis but the anomaly
#     detector only needs timestamp, service, level, and message text.
#
# - DEBUG and CRITICAL log levels:
#     Three levels (INFO, WARN, ERROR) are sufficient to demonstrate
#     anomaly detection. A production system would use all five.
#
# - JSON serialization (to_dict / to_json):
#     The MVP passes LogEntry objects directly between modules in memory.
#     In production you'd serialize to JSON for log shipping (e.g. to
#     Elasticsearch or CloudWatch).
#
# - Generator/yield pattern:
#     Replaced with a simpler generate_tick() that returns a list.
#     A real system would use async generators or a message queue.
# ──────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class LogEntry:
    """A single structured log record emitted by a simulated service."""

    timestamp: datetime
    level: str          # INFO | WARNING | ERROR
    service: str        # e.g. "transaction-validator"
    message: str        # Human-readable log message


class LogGenerator:
    """
    Produces a batch of log entries for all services on each tick.

    Checks the shared service_states dict to determine healthy vs fault
    output. Also checks the dependency graph so that when a downstream
    service is faulted, upstream services emit cascade error messages.
    """

    def __init__(self, simulator_config: dict[str, Any], service_states: dict) -> None:
        """
        Args:
            simulator_config: The 'simulator' section from config.yaml
                              containing 'services' list with names and dependencies.
            service_states:   Shared mutable dict tracking each service's health.
                              Format: {"service-name": {"healthy": True, "fault_type": None}}
        """
        raise NotImplementedError

    def generate_tick(self) -> list[LogEntry]:
        """
        Produce one batch of log entries (one or more per service).

        For each service:
          1. Check if the service itself is faulted → use fault templates
          2. Check if any dependency is faulted → mix in cascade templates
          3. Otherwise → use healthy templates with normal level weights

        Returns:
            List of LogEntry objects for this tick.
        """
        raise NotImplementedError

    def _select_log_level(self, fault_active: bool) -> str:
        """
        Pick a log level using weighted random selection.

        Normal weights:  INFO 80%, WARNING 15%, ERROR 5%
        Fault weights:   INFO 10%, WARNING 20%, ERROR 70%

        Args:
            fault_active: Whether this service or its dependencies are faulted.

        Returns:
            One of "INFO", "WARNING", "ERROR"
        """
        raise NotImplementedError

    def _pick_template(self, service: str, level: str, fault_type: str | None) -> str:
        """
        Select a random message template for the given service, level, and state.

        Templates are defined per service, per state (healthy / fault_type),
        per level. Placeholders like {tx_id}, {pid}, {latency} are filled
        with random values by _fill_template().

        Args:
            service:    Service name.
            level:      Log level.
            fault_type: Active fault type, or None for healthy operation.

        Returns:
            A message template string with placeholders.
        """
        raise NotImplementedError

    def _fill_template(self, template: str) -> str:
        """
        Replace placeholders in a template with random realistic values.

        Handles: {tx_id}, {pid}, {latency}, {duration}, {score}, {n}, {count}
        Unused placeholders in a given template are harmless — .format()
        only replaces what's present if you use .format_map() with a defaultdict.

        Args:
            template: Message string with {placeholder} tokens.

        Returns:
            Filled message string.
        """
        raise NotImplementedError

    def _get_faulted_dependencies(self, service: str) -> list[str]:
        """
        Check if any of this service's dependencies are currently faulted.

        Uses the dependency graph from config.yaml.

        Args:
            service: The service to check dependencies for.

        Returns:
            List of dependency service names that are currently faulted.
            Empty list if all dependencies are healthy.
        """
        raise NotImplementedError