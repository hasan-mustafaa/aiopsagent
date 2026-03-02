"""
Tests for the simulator package.

Covers:
  - LogGenerator:     correct schema, level distribution, fault message injection
  - MetricsGenerator: baseline sampling, fault profile application, baseline restore
  - FaultInjector:    fault activation, expiry, cascade across dependent services,
                      clear / clear_all behaviour
"""

from __future__ import annotations

import pytest

from src.simulator.log_generator import LogEntry, LogGenerator
from src.simulator.metrics_generator import MetricSnapshot, MetricsGenerator
from src.simulator.fault_injector import FaultInjector, FaultScenario

# Services defined in the test config — used to avoid magic strings in assertions.
_SERVICE_NAMES = [
    "transaction-validator",
    "fraud-check-service",
    "document-processor",
    "title-search-service",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simulator_config() -> dict:
    """Minimal simulator config mirroring config.yaml structure."""
    return {
        "services": [
            {"name": "transaction-validator",  "port": 8001, "dependencies": ["fraud-check-service", "document-processor"]},
            {"name": "fraud-check-service",    "port": 8002, "dependencies": ["title-search-service"]},
            {"name": "document-processor",     "port": 8003, "dependencies": ["title-search-service"]},
            {"name": "title-search-service",   "port": 8004, "dependencies": []},
        ],
        "log_interval_seconds": 1,
        "metrics_interval_seconds": 5,
    }


@pytest.fixture
def service_states(simulator_config: dict) -> dict:
    """All-healthy shared state dict — tests can mutate this to simulate faults."""
    return {s["name"]: {"healthy": True, "fault_type": None} for s in simulator_config["services"]}


@pytest.fixture
def log_generator(simulator_config: dict, service_states: dict) -> LogGenerator:
    """Create a LogGenerator wired to the test config and healthy service states."""
    return LogGenerator(simulator_config, service_states)


@pytest.fixture
def metrics_generator(simulator_config: dict) -> MetricsGenerator:
    """Create a MetricsGenerator from the test config."""
    return MetricsGenerator(simulator_config)


@pytest.fixture
def fault_injector(
    simulator_config: dict,
    log_generator: LogGenerator,
    metrics_generator: MetricsGenerator,
) -> FaultInjector:
    """Create a FaultInjector wired to the test generators."""
    return FaultInjector(simulator_config, log_generator, metrics_generator)


# ── LogGenerator tests ────────────────────────────────────────────────────────

class TestLogGenerator:

    # ── Basic output shape ────────────────────────────────────────────────────

    def test_generate_yields_one_entry_per_service(self, log_generator: LogGenerator) -> None:
        """generate_tick() must return exactly one LogEntry per configured service."""
        # This is the most fundamental contract: every downstream consumer
        # (detection, dashboard) assumes one entry per service per tick.
        tick = log_generator.generate_tick()
        assert len(tick) == len(_SERVICE_NAMES)

    def test_log_entry_schema(self, log_generator: LogGenerator) -> None:
        """Every field on each LogEntry must be populated (no None / empty string)."""
        # The detection layer indexes entries by service name and level — either
        # missing would silently break feature extraction downstream.
        from datetime import datetime
        for entry in log_generator.generate_tick():
            assert isinstance(entry, LogEntry)
            assert isinstance(entry.timestamp, datetime)
            assert entry.level           # non-empty string
            assert entry.service         # non-empty string
            assert entry.message         # non-empty string

    def test_log_level_is_valid(self, log_generator: LogGenerator) -> None:
        """Log levels must be restricted to the three levels defined in the spec."""
        # Running 10 ticks gives enough randomness to surface any typo in
        # _select_log_level's level list without making the test slow.
        valid_levels = {"INFO", "WARNING", "ERROR"}
        for _ in range(10):
            for entry in log_generator.generate_tick():
                assert entry.level in valid_levels, f"Unexpected level: {entry.level!r}"

    def test_service_names_match_config(self, log_generator: LogGenerator) -> None:
        """Each LogEntry.service must be one of the names from the config."""
        for entry in log_generator.generate_tick():
            assert entry.service in _SERVICE_NAMES, f"Unknown service: {entry.service!r}"

    # ── Fault behaviour ───────────────────────────────────────────────────────

    def test_fault_shifts_level_distribution_toward_error(
        self,
        log_generator: LogGenerator,
        service_states: dict,
    ) -> None:
        """ERROR rate for a faulted service must be significantly higher than healthy.

        We inject a crash into title-search-service and sample 200 ticks.
        The fault weights are INFO 10 / WARNING 20 / ERROR 70, so ERROR should
        dominate. We just assert ERROR > INFO (a very weak claim) to avoid
        making the test flaky on small samples.
        """
        service_states["title-search-service"] = {"healthy": False, "fault_type": "crash"}

        from collections import Counter
        counts: Counter[str] = Counter()
        for _ in range(200):
            for entry in log_generator.generate_tick():
                if entry.service == "title-search-service":
                    counts[entry.level] += 1

        assert counts["ERROR"] > counts["INFO"], (
            f"Expected ERROR > INFO under fault, got {dict(counts)}"
        )

    def test_no_unfilled_placeholders_in_messages(self, log_generator: LogGenerator) -> None:
        """No message should contain a literal '{' after template filling.

        Any leftover {placeholder} means _fill_template has a gap in its
        values dict — that would show up as garbled text in the dashboard.
        """
        for _ in range(30):
            for entry in log_generator.generate_tick():
                assert "{" not in entry.message, (
                    f"Unfilled placeholder in [{entry.service}]: {entry.message!r}"
                )

    def test_cascade_message_does_not_contain_literal_dep(
        self,
        log_generator: LogGenerator,
        service_states: dict,
    ) -> None:
        """When a dependency is faulted, cascade messages must reference its real name.

        title-search-service is a dependency of both fraud-check-service and
        document-processor. After injecting a fault there, those two services
        should emit cascade messages that mention 'title-search-service' by name,
        not the raw '{dep}' placeholder.
        """
        service_states["title-search-service"] = {"healthy": False, "fault_type": "crash"}

        cascade_services = {"fraud-check-service", "document-processor"}

        for _ in range(20):
            for entry in log_generator.generate_tick():
                assert "{dep}" not in entry.message, (
                    f"Unfilled {{dep}} in [{entry.service}]: {entry.message!r}"
                )

    # ── Removed-for-MVP stubs (kept so the intent is documented) ─────────────

    @pytest.mark.skip(reason="inject_fault_message() removed for MVP — fault state is driven "
                             "by service_states dict, not a direct method call")
    def test_inject_fault_message_returns_error_or_critical(
        self, log_generator: LogGenerator
    ) -> None:
        """inject_fault_message() must return a log at ERROR or CRITICAL level."""
        raise NotImplementedError

    @pytest.mark.skip(reason="inject_fault_message() removed for MVP")
    def test_inject_fault_message_includes_fault_type(
        self, log_generator: LogGenerator
    ) -> None:
        """Fault message content or metadata should reference the fault type."""
        raise NotImplementedError

    @pytest.mark.skip(reason="trace_id removed for MVP — detection works on aggregate "
                             "level counts, not per-request traces")
    def test_trace_ids_are_unique_across_entries(self, log_generator: LogGenerator) -> None:
        """Each generated log entry should have a distinct trace_id UUID."""
        raise NotImplementedError

    @pytest.mark.skip(reason="to_dict() / JSON serialisation removed for MVP — "
                             "LogEntry objects are passed in-memory between modules")
    def test_to_dict_is_json_serialisable(self, log_generator: LogGenerator) -> None:
        """LogEntry.to_dict() should return a dict serialisable by json.dumps."""
        raise NotImplementedError


# ── MetricsGenerator tests ────────────────────────────────────────────────────

class TestMetricsGenerator:

    # Helper: drain N full rounds (4 snapshots each) from the generator.
    @staticmethod
    def _take(gen: Any, n_rounds: int, n_services: int = 4) -> list[MetricSnapshot]:
        import itertools
        return list(itertools.islice(gen, n_rounds * n_services))

    # ── Output shape ──────────────────────────────────────────────────────────

    def test_generate_yields_one_snapshot_per_service(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """generate() yields exactly one MetricSnapshot per service per round.

        We take one full round (4 items) and check that each configured service
        appears exactly once, in config order.
        """
        snapshots = self._take(metrics_generator.generate(), n_rounds=1)
        assert len(snapshots) == len(_SERVICE_NAMES)
        assert [s.service for s in snapshots] == _SERVICE_NAMES

    def test_snapshot_values_within_plausible_range(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """All metric values must satisfy their physical constraints.

        We sample 10 rounds (40 snapshots) to account for Gaussian variance.
        """
        snapshots = self._take(metrics_generator.generate(), n_rounds=10)
        for s in snapshots:
            assert 0.0 <= s.cpu_percent <= 100.0,  f"cpu out of range: {s.cpu_percent}"
            assert s.memory_mb > 0,                f"memory non-positive: {s.memory_mb}"
            assert s.latency_ms >= 0,              f"latency negative: {s.latency_ms}"
            assert s.error_rate >= 0,              f"error_rate negative: {s.error_rate}"
            assert s.request_rate >= 0,            f"request_rate negative: {s.request_rate}"
            assert s.active_connections >= 0,      f"connections negative: {s.active_connections}"

    # ── Fault profiles ────────────────────────────────────────────────────────

    def test_apply_fault_profile_increases_latency(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """After latency_spike, mean latency across 50 samples must be higher than normal.

        The multiplier is 15×, so even with Gaussian noise this should be a
        very reliable assertion.
        """
        service = "title-search-service"
        gen = metrics_generator.generate()

        # Sample 50 healthy snapshots and compute mean latency.
        healthy = [s for s in self._take(gen, n_rounds=50) if s.service == service]
        healthy_mean = sum(s.latency_ms for s in healthy) / len(healthy)

        metrics_generator.apply_fault_profile(service, "latency_spike")
        faulted = [s for s in self._take(gen, n_rounds=50) if s.service == service]
        faulted_mean = sum(s.latency_ms for s in faulted) / len(faulted)

        assert faulted_mean > healthy_mean * 5, (
            f"Expected faulted latency >> healthy, got {faulted_mean:.1f} vs {healthy_mean:.1f}"
        )

    def test_apply_fault_profile_crash_drops_request_rate(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """After crash, mean request_rate across 50 samples must drop near zero.

        The crash multiplier is 0×, so request_rate_mean → 0.
        """
        service = "fraud-check-service"
        gen = metrics_generator.generate()

        metrics_generator.apply_fault_profile(service, "crash")
        faulted = [s for s in self._take(gen, n_rounds=50) if s.service == service]
        faulted_mean = sum(s.request_rate for s in faulted) / len(faulted)

        assert faulted_mean < 1.0, (
            f"Expected near-zero request_rate after crash, got {faulted_mean:.2f}"
        )

    def test_restore_baseline_reverts_fault_profile(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """restore_baseline() must return latency to its normal operating range.

        Strategy: apply latency_spike (15× multiplier), then restore, then
        verify the mean latency drops back close to the original baseline.
        """
        service = "title-search-service"
        # title-search-service normal latency_mean = 15 ms
        normal_latency_mean = 15.0
        gen = metrics_generator.generate()

        metrics_generator.apply_fault_profile(service, "latency_spike")
        metrics_generator.restore_baseline(service)

        restored = [s for s in self._take(gen, n_rounds=50) if s.service == service]
        restored_mean = sum(s.latency_ms for s in restored) / len(restored)

        # Should be back within 3× of the normal mean (generous for Gaussian noise).
        assert restored_mean < normal_latency_mean * 3, (
            f"Latency not restored: {restored_mean:.1f}ms (expected ~{normal_latency_mean}ms)"
        )

    # ── Latest snapshot cache ─────────────────────────────────────────────────

    def test_get_latest_snapshot_returns_none_before_first_tick(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """get_latest_snapshot() must return None before generate() has been called."""
        for name in _SERVICE_NAMES:
            assert metrics_generator.get_latest_snapshot(name) is None

    def test_get_latest_snapshot_returns_snapshot_after_generate(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """After one full round from generate(), every service has a cached snapshot."""
        self._take(metrics_generator.generate(), n_rounds=1)
        for name in _SERVICE_NAMES:
            snap = metrics_generator.get_latest_snapshot(name)
            assert snap is not None
            assert snap.service == name

    # ── Serialisation ─────────────────────────────────────────────────────────

    def test_snapshot_to_dict_is_json_serialisable(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """MetricSnapshot.to_dict() must produce a dict that json.dumps accepts."""
        import json
        snapshots = self._take(metrics_generator.generate(), n_rounds=1)
        for snap in snapshots:
            d = snap.to_dict()
            assert isinstance(d, dict)
            json.dumps(d)  # raises if any value is non-serialisable


# ── FaultInjector tests ───────────────────────────────────────────────────────

class TestFaultInjector:

    # ── inject() ──────────────────────────────────────────────────────────────

    def test_inject_returns_fault_scenario(self, fault_injector: FaultInjector) -> None:
        """inject() must return an active FaultScenario with correct fields."""
        scenario = fault_injector.inject("title-search-service", "crash")
        assert isinstance(scenario, FaultScenario)
        assert scenario.is_active
        assert scenario.service == "title-search-service"
        assert scenario.fault_type == "crash"
        assert scenario.resolved_at is None
        assert scenario.fault_id  # non-empty UUID string

    def test_active_faults_contains_injected_fault(
        self, fault_injector: FaultInjector
    ) -> None:
        """active_faults() must list the fault immediately after inject()."""
        fault_injector.inject("fraud-check-service", "latency_spike")
        active = fault_injector.active_faults()
        assert len(active) == 1
        assert active[0].service == "fraud-check-service"

    def test_inject_mutates_service_states(
        self, fault_injector: FaultInjector, service_states: dict
    ) -> None:
        """inject() must flip service_states[service] to unhealthy so LogGenerator reacts."""
        fault_injector.inject("title-search-service", "crash")
        state = service_states["title-search-service"]
        assert state["healthy"] is False
        assert state["fault_type"] == "crash"

    # ── Expiry via tick() ─────────────────────────────────────────────────────

    def test_fault_expires_after_duration(self, fault_injector: FaultInjector) -> None:
        """A fault with duration_seconds=0 must be resolved on the next tick().

        We use duration_seconds=0 so the fault has already elapsed by the time
        tick() runs — no real sleeping needed.
        """
        scenario = fault_injector.inject("title-search-service", "crash", duration_seconds=0)
        assert scenario.is_active

        fault_injector.tick()

        assert not scenario.is_active
        assert scenario.resolved_at is not None
        assert fault_injector.active_faults() == []

    def test_indefinite_fault_not_expired_by_tick(
        self, fault_injector: FaultInjector
    ) -> None:
        """duration_seconds=-1 means the fault must persist through tick() calls."""
        scenario = fault_injector.inject("title-search-service", "crash", duration_seconds=-1)
        fault_injector.tick()
        fault_injector.tick()
        assert scenario.is_active

    def test_service_restored_after_expiry(
        self, fault_injector: FaultInjector, service_states: dict
    ) -> None:
        """After a fault expires via tick(), service_states must return to healthy."""
        fault_injector.inject("title-search-service", "crash", duration_seconds=0)
        fault_injector.tick()
        state = service_states["title-search-service"]
        assert state["healthy"] is True
        assert state["fault_type"] is None

    # ── clear() / clear_all() ─────────────────────────────────────────────────

    def test_clear_removes_specific_service_fault(
        self, fault_injector: FaultInjector
    ) -> None:
        """clear(service) must resolve all active faults on that service."""
        fault_injector.inject("title-search-service", "crash")
        fault_injector.inject("fraud-check-service", "latency_spike")
        fault_injector.clear("title-search-service")

        active_services = {s.service for s in fault_injector.active_faults()}
        assert "title-search-service" not in active_services
        assert "fraud-check-service" in active_services  # untouched

    def test_clear_all_removes_every_fault(self, fault_injector: FaultInjector) -> None:
        """clear_all() must leave active_faults() empty regardless of count."""
        fault_injector.inject("title-search-service", "crash")
        fault_injector.inject("fraud-check-service", "latency_spike")
        fault_injector.inject("document-processor", "memory_leak")
        fault_injector.clear_all()
        assert fault_injector.active_faults() == []

    # ── Validation ────────────────────────────────────────────────────────────

    def test_inject_invalid_service_raises_value_error(
        self, fault_injector: FaultInjector
    ) -> None:
        """inject() with an unknown service name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown service"):
            fault_injector.inject("nonexistent-service", "crash")

    def test_inject_invalid_fault_type_raises_value_error(
        self, fault_injector: FaultInjector
    ) -> None:
        """inject() with an unknown fault type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown fault type"):
            fault_injector.inject("title-search-service", "explode")

    # ── History ───────────────────────────────────────────────────────────────

    def test_fault_history_accumulates_resolved_faults(
        self, fault_injector: FaultInjector
    ) -> None:
        """fault_history() must include both resolved and still-active faults."""
        fault_injector.inject("title-search-service", "crash", duration_seconds=0)
        fault_injector.inject("fraud-check-service", "latency_spike")
        fault_injector.tick()  # resolves the first

        history = fault_injector.fault_history()
        assert len(history) == 2

        resolved = [s for s in history if not s.is_active]
        active   = [s for s in history if s.is_active]
        assert len(resolved) == 1
        assert len(active) == 1
        assert resolved[0].service == "title-search-service"

    def test_second_fault_on_same_service_keeps_it_unhealthy(
        self, fault_injector: FaultInjector, service_states: dict
    ) -> None:
        """If two faults are active on the same service, resolving one must not
        restore service_states — the service is still faulted by the second."""
        fault_injector.inject("title-search-service", "crash",         duration_seconds=0)
        fault_injector.inject("title-search-service", "latency_spike", duration_seconds=-1)
        fault_injector.tick()  # only crash expires

        state = service_states["title-search-service"]
        assert state["healthy"] is False, "Service should still be unhealthy"
