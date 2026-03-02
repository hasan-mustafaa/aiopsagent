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

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# ── Message templates ─────────────────────────────────────────────────────────
# Structure: service → fault_state → level → [template_string, ...]
#
# States: "healthy" | FAULT_TYPES | "cascade"
# "cascade" applies when the service itself is healthy but a dependency is
# faulted. Templates in this state may use {dep}, which generate_tick()
# substitutes with the actual dependency name before calling _fill_template().
# Empty state/level lists fall back to "healthy" in _pick_template().

_TEMPLATES: dict[str, dict[str, dict[str, list[str]]]] = {

    "transaction-validator": {
        "healthy": {
            "INFO": [
                "Validated transaction {tx_id} in {latency}ms",
                "Transaction {tx_id} passed AML check, score={score}",
                "Processed {n} transactions in last window",
                "Routing transaction {tx_id} to downstream services",
            ],
            "WARNING": [
                "Transaction {tx_id} took {latency}ms — approaching SLA threshold",
                "Unusual pattern in transaction {tx_id}, score={score}",
                "Queue depth at {n} items, consider scaling",
            ],
            "ERROR": [
                "Failed to route transaction {tx_id}: downstream timeout",
                "Transaction {tx_id} rejected after {n} retries",
            ],
        },
        "crash": {
            "INFO": [
                "Service recovering; processed {n} transactions post-restart",
            ],
            "WARNING": [
                "Restarting: {n} in-flight transactions dropped",
                "Health check failed, pid={pid}",
            ],
            "ERROR": [
                "Service crashed: unhandled exception in transaction routing",
                "Fatal error in transaction-validator (pid={pid}): SIGABRT",
                "Transaction {tx_id} lost: service unavailable during crash recovery",
            ],
        },
        "latency_spike": {
            "INFO": [
                "Transaction {tx_id} completed but slow: {latency}ms",
            ],
            "WARNING": [
                "P99 latency at {latency}ms — SLA breach imminent",
                "Transaction {tx_id} retried {n} times due to slow response",
            ],
            "ERROR": [
                "Transaction {tx_id} timed out after {latency}ms",
                "SLA violation: {n} transactions exceeded {latency}ms threshold",
            ],
        },
        "connection_failure": {
            "INFO": [
                "Reconnection attempt {n} for downstream fraud-check-service",
            ],
            "WARNING": [
                "Connection pool exhausted: {count} active, {n} waiting",
                "Intermittent connection loss to document-processor",
            ],
            "ERROR": [
                "Connection refused by fraud-check-service: {n} retries exhausted",
                "Network partition detected: cannot reach downstream services",
            ],
        },
        "memory_leak": {
            "INFO": [
                "Memory usage at {n}MB, within acceptable range",
            ],
            "WARNING": [
                "Heap size growing: {n}MB allocated, {count}MB retained",
                "GC pressure: {n} collections in last minute",
            ],
            "ERROR": [
                "Memory usage critical: {n}MB — approaching OOM threshold",
                "Possible memory leak in transaction cache: finalizer queue depth {n}",
            ],
        },
        "oom": {
            "INFO": [],
            "WARNING": [
                "Pre-OOM warning: only {n}MB heap remaining",
            ],
            "ERROR": [
                "OutOfMemoryError: transaction-validator killed by OOM killer",
                "JVM heap exhausted at {n}MB — process terminating",
                "OOM in transaction buffer: {count} transactions lost",
            ],
        },
        "cascade": {
            "INFO": [
                "Dependency {dep} degraded, falling back to cached data",
            ],
            "WARNING": [
                "Upstream call to {dep} slow: {latency}ms",
                "{dep} returned partial results, {n} transactions pending",
            ],
            "ERROR": [
                "Dependency {dep} unavailable: transaction {tx_id} failed",
                "Cascade failure: {dep} not responding after {n} retries",
            ],
        },
    },

    "fraud-check-service": {
        "healthy": {
            "INFO": [
                "Fraud check passed for transaction {tx_id}, score={score}",
                "Screened {n} transactions, {count} flagged for review",
                "ML model inference completed in {duration}ms, score={score}",
                "AML watchlist lookup returned clean for tx {tx_id}",
            ],
            "WARNING": [
                "High-risk transaction {tx_id} queued for manual review, score={score}",
                "Model confidence low ({score}): flagging for human review",
                "Fraud check latency {latency}ms — exceeding target",
            ],
            "ERROR": [
                "Fraud model failed to score transaction {tx_id}: missing features",
                "Watchlist service timeout for transaction {tx_id}",
            ],
        },
        "crash": {
            "INFO": [
                "Fraud-check-service restarted, warming up model cache",
            ],
            "WARNING": [
                "Service restart: {n} in-flight fraud checks dropped",
            ],
            "ERROR": [
                "fraud-check-service crashed: segfault in ML scoring pipeline",
                "Process {pid} killed: unhandled exception in fraud scorer",
                "Fraud check unavailable: service restart required",
            ],
        },
        "latency_spike": {
            "INFO": [
                "Fraud check for {tx_id} completed in {latency}ms",
            ],
            "WARNING": [
                "Model inference slow: {latency}ms for transaction {tx_id}",
            ],
            "ERROR": [
                "Fraud check timeout ({latency}ms) for transaction {tx_id}",
                "Batch scoring delayed: {n} transactions waiting in queue",
            ],
        },
        "connection_failure": {
            "INFO": [
                "Retrying connection to title-search-service (attempt {n})",
            ],
            "WARNING": [
                "title-search-service unreachable, using stale cache for {n} lookups",
            ],
            "ERROR": [
                "Cannot connect to title-search-service: connection refused",
                "Fraud check failed for {tx_id}: title lookup dependency unavailable",
            ],
        },
        "memory_leak": {
            "INFO": [
                "Model cache size: {n}MB",
            ],
            "WARNING": [
                "Model cache growing unboundedly: {n}MB and increasing",
                "Feature store not evicting: {count}MB retained",
            ],
            "ERROR": [
                "Memory exhaustion in fraud model cache: {n}MB",
                "OOM risk: fraud-check-service heap at {n}MB",
            ],
        },
        "oom": {
            "INFO": [],
            "WARNING": [
                "Low memory warning in fraud scorer: {n}MB remaining",
            ],
            "ERROR": [
                "OutOfMemoryError: fraud-check-service terminated by OOM killer",
                "Heap exhausted during batch fraud scoring: {count} transactions lost",
            ],
        },
        "cascade": {
            "INFO": [
                "title-search-service degraded, fraud check using cached title data",
            ],
            "WARNING": [
                "Dependency {dep} slow ({latency}ms): fraud scoring delayed for {n} transactions",
                "Cache miss rate high due to {dep} outage",
            ],
            "ERROR": [
                "Cannot complete fraud check: {dep} unavailable for transaction {tx_id}",
                "Cascade from {dep}: {n} fraud checks queued without title data",
            ],
        },
    },

    "document-processor": {
        "healthy": {
            "INFO": [
                "Document {tx_id} processed successfully in {duration}ms",
                "OCR completed for document batch: {n} pages scanned",
                "Extracted {n} entities from document {tx_id}",
                "Document validation passed: checksum OK for {tx_id}",
            ],
            "WARNING": [
                "Document {tx_id} low OCR confidence ({score}): flagged for review",
                "Processing queue depth {n} — may delay SLA",
                "Large document {tx_id}: {n}MB exceeds recommended size",
            ],
            "ERROR": [
                "Failed to parse document {tx_id}: unsupported format",
                "Document corruption detected in {tx_id}: checksum mismatch",
            ],
        },
        "crash": {
            "INFO": [
                "document-processor restarted, resuming queue from offset {n}",
            ],
            "WARNING": [
                "Crash recovery: {n} documents re-queued from dead-letter queue",
            ],
            "ERROR": [
                "document-processor crashed: segfault in OCR engine (pid={pid})",
                "Fatal error processing document {tx_id}: worker process killed",
            ],
        },
        "latency_spike": {
            "INFO": [
                "Document {tx_id} processed in {latency}ms",
            ],
            "WARNING": [
                "Slow document processing: {latency}ms for {tx_id}",
                "Processing pipeline backed up: {n} documents waiting",
            ],
            "ERROR": [
                "Document {tx_id} processing timed out after {latency}ms",
            ],
        },
        "connection_failure": {
            "INFO": [
                "Reconnecting to title-search-service for document {tx_id}",
            ],
            "WARNING": [
                "Title lookup failed for document {tx_id}, retrying ({n}/{count})",
            ],
            "ERROR": [
                "Cannot retrieve title data: title-search-service connection refused",
                "Document processing halted: storage backend unreachable",
            ],
        },
        "memory_leak": {
            "INFO": [
                "OCR engine memory: {n}MB",
            ],
            "WARNING": [
                "OCR memory growing: {n}MB ({count}MB increase since last check)",
                "Image buffer not being released: {n}MB retained",
            ],
            "ERROR": [
                "Memory limit reached in document processor: {n}MB",
                "OCR worker killed due to memory overrun on document {tx_id}",
            ],
        },
        "oom": {
            "INFO": [],
            "WARNING": [
                "Low memory in document processor: {n}MB remaining",
            ],
            "ERROR": [
                "OOM: document-processor killed while processing {n} documents",
                "Heap exhausted: document batch {tx_id} cannot be processed",
            ],
        },
        "cascade": {
            "INFO": [
                "Dependency {dep} slow, document processing using cached title data",
            ],
            "WARNING": [
                "Title lookup via {dep} taking {latency}ms, document {tx_id} delayed",
            ],
            "ERROR": [
                "Document {tx_id} failed: {dep} unreachable for title verification",
                "Cascade failure: {n} documents stalled waiting for {dep}",
            ],
        },
    },

    "title-search-service": {
        "healthy": {
            "INFO": [
                "Title search completed for property {tx_id} in {duration}ms",
                "Index lookup returned {n} results for query {tx_id}",
                "Cache hit for title query {tx_id} (ratio={score})",
                "Batch title lookup: {n} queries processed in {duration}ms",
            ],
            "WARNING": [
                "Title index rebuild in progress: {n}% complete",
                "Cache eviction rate high: {count} evictions in last minute",
                "Slow index scan for property {tx_id}: {latency}ms",
            ],
            "ERROR": [
                "Title record not found for {tx_id}: index may be stale",
                "Search index corrupted for partition {n}: rebuild required",
            ],
        },
        "crash": {
            "INFO": [
                "title-search-service restarting, rebuilding in-memory index",
            ],
            "WARNING": [
                "Service crash: index state lost, {n} cached entries invalidated",
            ],
            "ERROR": [
                "title-search-service crashed: panic in search index (pid={pid})",
                "Unhandled exception in title lookup for query {tx_id}",
                "Process {pid} terminated unexpectedly — service unavailable",
            ],
        },
        "latency_spike": {
            "INFO": [
                "Title search for {tx_id} completed in {latency}ms",
            ],
            "WARNING": [
                "Index scan slow: {latency}ms for {n}-result query",
                "Full-text search degraded: {latency}ms P99",
            ],
            "ERROR": [
                "Title search timed out ({latency}ms) for property {tx_id}",
                "Search index overloaded: {n} queries in queue, {latency}ms wait time",
            ],
        },
        "connection_failure": {
            "INFO": [
                "Reconnecting to backing store (attempt {n})",
            ],
            "WARNING": [
                "Database connection intermittent: {count} of {n} queries failed",
            ],
            "ERROR": [
                "Cannot connect to title index database: connection refused",
                "All database connections exhausted: {n} pending queries dropped",
            ],
        },
        "memory_leak": {
            "INFO": [
                "Search index memory: {n}MB",
            ],
            "WARNING": [
                "In-memory index growing: {n}MB, expected {count}MB",
                "Search result cache not evicting: {n}MB retained",
            ],
            "ERROR": [
                "Memory limit exceeded in title index: {n}MB",
                "Search worker OOM-killed: index partition {n} unavailable",
            ],
        },
        "oom": {
            "INFO": [],
            "WARNING": [
                "Low memory: title-search-service has {n}MB remaining",
            ],
            "ERROR": [
                "OOM: title-search-service killed during index rebuild",
                "Heap exhausted: {n} pending title searches dropped",
            ],
        },
        # title-search-service is a leaf node — it has no dependencies,
        # so cascade never fires for it. Empty lists fall back to healthy.
        "cascade": {
            "INFO": [],
            "WARNING": [],
            "ERROR": [],
        },
    },
}


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
        services_cfg = simulator_config["services"]
        self._services: list[str] = [s["name"] for s in services_cfg]
        self._deps: dict[str, list[str]] = {
            s["name"]: s.get("dependencies", []) for s in services_cfg
        }
        # Held by reference — FaultInjector mutates this in-place, so changes
        # are visible on the next generate_tick() without any explicit sync.
        self.service_states = service_states

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
        now = datetime.now()
        entries: list[LogEntry] = []

        for service in self._services:
            state = self.service_states.get(service, {"healthy": True, "fault_type": None})
            own_fault: str | None = None if state.get("healthy", True) else state.get("fault_type")
            faulted_deps = self._get_faulted_dependencies(service)

            fault_active = own_fault is not None or bool(faulted_deps)
            level = self._select_log_level(fault_active)

            if own_fault is not None:
                template = self._pick_template(service, level, own_fault)
            elif faulted_deps:
                # Substitute {dep} before _fill_template so the defaultdict
                # fallback doesn't silently swallow it as an empty string.
                template = self._pick_template(service, level, "cascade")
                template = template.replace("{dep}", faulted_deps[0])
            else:
                template = self._pick_template(service, level, None)

            message = self._fill_template(template)
            entries.append(LogEntry(timestamp=now, level=level, service=service, message=message))

        return entries

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
        levels = ["INFO", "WARNING", "ERROR"]
        weights = [10, 20, 70] if fault_active else [80, 15, 5]
        return random.choices(levels, weights=weights, k=1)[0]

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
        state_key = fault_type if fault_type is not None else "healthy"
        service_tmpl = _TEMPLATES.get(service, {})

        # Primary lookup: templates for this fault state + level.
        candidates = service_tmpl.get(state_key, {}).get(level, [])

        # Fallback: some fault states have no INFO-level templates (e.g. "oom").
        # Use the healthy templates for that level instead.
        if not candidates:
            candidates = service_tmpl.get("healthy", {}).get(level, [])

        if not candidates:
            return f"{service} reported a {level.lower()} event"

        return random.choice(candidates)

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
        values: dict[str, Any] = {
            "tx_id":    f"TXN-{random.randint(10_000, 99_999)}",
            "pid":      random.randint(1_000, 65_535),
            "latency":  random.randint(50, 8_000),    # milliseconds
            "duration": random.randint(10, 500),       # milliseconds
            "score":    round(random.uniform(0.0, 1.0), 3),
            "n":        random.randint(1, 100),
            "count":    random.randint(1, 500),
        }
        # defaultdict(str) silently returns "" for unknown keys, so any
        # placeholder not in `values` degrades gracefully instead of raising.
        return template.format_map(defaultdict(str, values))

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
        return [
            dep
            for dep in self._deps.get(service, [])
            if not self.service_states.get(dep, {}).get("healthy", True)
        ]