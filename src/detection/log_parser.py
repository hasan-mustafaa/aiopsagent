"""Online log parsing via the Drain algorithm.

Clusters raw logs into templates. Extracts features for anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from src.simulator.log_generator import LogEntry


@dataclass
class ParsedLog:
    """Log record enriched with Drain template and extracted parameters.

    Attributes:
        original:    Raw LogEntry (timestamp, level, service, message).
        template:    Generalized pattern with <*> wildcards.
        cluster_id:  Global cluster ID (assigned by LogParser, stable across resets).
        parameters:  Extracted variable values (e.g., ["192.168.1.1", "5000"]).
    """

    original: LogEntry
    template: str
    cluster_id: int
    parameters: list[str] = field(default_factory=list)

    @property
    def service(self) -> str:
        """Service name that emitted this log."""
        return self.original.service

    @property
    def level(self) -> str:
        """Log level from the original entry (INFO | WARNING | ERROR)."""
        return self.original.level


def _make_miner_config() -> TemplateMinerConfig:
    """Create Drain config for microservice logs."""
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.4  # Lower threshold for diverse log payloads
    cfg.drain_depth = 4     # Suitable for <10K distinct templates
    cfg.parametrize_numeric_tokens = True  # Treat numbers as variables
    return cfg


class LogParser:
    """Per-service log clustering with global cluster ID mapping.

    Maintains independent Drain instances per service. Local cluster IDs (which
    restart at 1 per service) are mapped to globally unique IDs for downstream
    feature extraction. Templates may evolve as Drain sees more logs.
    """

    def __init__(self, detection_config: dict[str, Any]) -> None:
        """Initialize parser with lazy miner instantiation per service."""
        self._detection_config = detection_config
        # service → TemplateMiner (created lazily on first log for that service)
        self._miners: dict[str, TemplateMiner] = {}
        # (service, local_drain_cluster_id) → globally unique cluster_id
        self._local_to_global: dict[tuple[str, int], int] = {}
        # global_cluster_id → most-recent template string
        self._global_templates: dict[int, str] = {}
        # service → set of global cluster IDs seen for that service
        self._service_global_ids: dict[str, set[int]] = {}
        self._next_global_id: int = 1

    def parse(self, log_entry: LogEntry) -> ParsedLog:
        """Parse a log message: route to service miner, extract template and params.

        Returns ParsedLog with global cluster ID, template, and extracted parameters.
        """
        miner = self._get_miner(log_entry.service)
        result = miner.add_log_message(log_entry.message)

        local_id: int = result["cluster_id"]
        template: str = result["template_mined"]

        # Assign a globally unique ID the first time this (service, local) pair appears.
        key = (log_entry.service, local_id)
        if key not in self._local_to_global:
            global_id = self._next_global_id
            self._next_global_id += 1
            self._local_to_global[key] = global_id
            self._service_global_ids.setdefault(log_entry.service, set()).add(global_id)
        global_id = self._local_to_global[key]

        # Keep the template map in sync (template may evolve as drain learns).
        self._global_templates[global_id] = template

        parameters: list[str] = miner.get_parameter_list(template, log_entry.message)

        return ParsedLog(
            original=log_entry,
            template=template,
            cluster_id=global_id,
            parameters=parameters,
        )

    def parse_batch(self, log_entries: list[LogEntry]) -> list[ParsedLog]:
        """Parse multiple log entries in order."""
        return [self.parse(entry) for entry in log_entries]

    def get_templates(self, service: str | None = None) -> dict[int, str]:
        """Query current templates: dict of {cluster_id: template_string}.

        If service is None, returns templates from all services.
        Returns empty dict if service has no logs yet.
        """
        if service is not None:
            miner = self._miners.get(service)
            if miner is None:
                return {}
            return {
                self._local_to_global[(service, local_id)]: cluster.get_template()
                for local_id, cluster in miner.drain.id_to_cluster.items()
                if (service, local_id) in self._local_to_global
            }

        # No filter — aggregate across all services.
        templates: dict[int, str] = {}
        for svc, miner in self._miners.items():
            for local_id, cluster in miner.drain.id_to_cluster.items():
                key = (svc, local_id)
                if key in self._local_to_global:
                    global_id = self._local_to_global[key]
                    templates[global_id] = cluster.get_template()
        return templates

    def reset(self, service: str | None = None) -> None:
        """Clear Drain state and cluster ID mappings for one or all services.

        Used primarily for test isolation. In production, avoid calling this to
        preserve template history.
        """
        if service is not None:
            services_to_reset = [service] if service in self._miners else []
        else:
            services_to_reset = list(self._miners.keys())

        for svc in services_to_reset:
            # Remove global ID mappings for this service's clusters.
            for local_id in list(self._miners[svc].drain.id_to_cluster.keys()):
                key = (svc, local_id)
                global_id = self._local_to_global.pop(key, None)
                if global_id is not None:
                    self._global_templates.pop(global_id, None)
            self._service_global_ids.pop(svc, None)
            # Replace the miner with a fresh instance.
            self._miners[svc] = TemplateMiner(config=_make_miner_config())

    def _get_miner(self, service: str) -> TemplateMiner:
        """Get or create Drain miner for this service (lazy instantiation)."""
        if service not in self._miners:
            self._miners[service] = TemplateMiner(config=_make_miner_config())
        return self._miners[service]
