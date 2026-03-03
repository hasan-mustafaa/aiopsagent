"""Remediation executor for the simulated service mesh.

Translates actions into simulator operations. All actions work on in-memory state:
- restart_service: clears faults, restores baseline
- scale_service: records replica change
- rollback_service: clears faults
- alert_on_call: logs alert
- no_action: no-op decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agent.action_planner import (
    Action,
    AlertAction,
    NoAction,
    RestartAction,
    RollbackAction,
    ScaleAction,
)
from src.remediation.guardrails import Guardrails

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single remediation action execution.

    Attributes:
        action:               The Action that was executed.
        success:              True if the action completed without error.
        blocked_by_guardrail: True if Guardrails prevented execution.
        message:              Human-readable outcome description.
        post_metrics:         MetricSnapshot dict after the action (agent feedback).
        post_logs:            Recent log entries after the action (agent feedback).
        executed_at:          Timestamp of execution.
        duration_ms:          Simulated wall-clock time for the action.
    """

    action: Action
    success: bool
    blocked_by_guardrail: bool
    message: str
    post_metrics: dict[str, Any] = field(default_factory=dict)
    post_logs: list[dict[str, Any]] = field(default_factory=list)
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for LLM observation injection."""
        return {
            "action": self.action.model_dump(),
            "success": self.success,
            "blocked_by_guardrail": self.blocked_by_guardrail,
            "message": self.message,
            "post_metrics": self.post_metrics,
            "post_logs": self.post_logs,
            "executed_at": self.executed_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


class Executor:
    """Executes remediation actions against the simulated service mesh.

    Checks each action against Guardrails before execution and records all
    attempts (successful or blocked) in an ordered execution log.

    simulator_state is duck-typed — expected to optionally expose:
      fault_injector:    FaultInjector (for restart/rollback)
      metrics_generator: MetricsGenerator (for post-action snapshots)
    May be None in tests/dry-run mode.
    """

    # Maps Action type → handler method name
    _DISPATCH: dict[type, str] = {
        RestartAction: "_execute_restart",
        ScaleAction: "_execute_scale",
        RollbackAction: "_execute_rollback",
        AlertAction: "_execute_alert",
        NoAction: "_execute_no_action",
    }

    def __init__(
        self,
        remediation_config: dict[str, Any],
        guardrails: Guardrails,
        simulator_state: Any | None = None,
    ) -> None:
        """Initialize from the [remediation] section of config.yaml."""
        self._config = remediation_config
        self._guardrails = guardrails
        self._simulator = simulator_state
        self._log: list[ExecutionResult] = []

    def execute(self, action: Action) -> ExecutionResult:
        """Validate and execute a single remediation action.

        Checks guardrails first. If blocked, returns a failed result without
        touching simulator state. Records every attempt in the execution log.

        Args:
            action: A validated Action object from ActionPlanner.

        Returns:
            ExecutionResult describing the outcome.
        """
        allowed, reason = self._guardrails.check(action)
        if not allowed:
            result = ExecutionResult(
                action=action, success=False, blocked_by_guardrail=True,
                message=f"Blocked by guardrails: {reason}",
            )
            self._log.append(result)
            logger.warning("Action blocked: %s — %s", action.action, reason)
            return result

        handler_name = self._DISPATCH.get(type(action))
        if handler_name is None:
            result = ExecutionResult(
                action=action, success=False, blocked_by_guardrail=False,
                message=f"Unknown action type: {type(action).__name__}",
            )
            self._log.append(result)
            return result

        result = getattr(self, handler_name)(action)
        self._guardrails.record_execution(action, result.success)
        self._log.append(result)
        logger.info(
            "Executed %s on %s — %s",
            action.action, getattr(action, "target", "n/a"), result.message,
        )
        return result

    def execution_log(self) -> list[ExecutionResult]:
        """Return the full ordered list of all execution attempts."""
        return list(self._log)

    def _execute_restart(self, action: RestartAction) -> ExecutionResult:
        """Clear active faults and restore the service metrics baseline."""
        service = action.target
        try:
            if self._simulator is not None:
                fi = getattr(self._simulator, "fault_injector", None)
                if fi is not None:
                    fi.clear(service)
        except Exception as exc:
            logger.error("Restart failed for %s: %s", service, exc)
            return ExecutionResult(
                action=action, success=False, blocked_by_guardrail=False,
                message=f"Restart failed: {exc}",
            )

        post = self._collect_post_action_state(service)
        return ExecutionResult(
            action=action, success=True, blocked_by_guardrail=False,
            message=f"Service '{service}' restarted — faults cleared, baseline restored.",
            post_metrics=post.get("metrics", {}),
            post_logs=post.get("logs", []),
            duration_ms=2000.0,
        )

    def _execute_scale(self, action: ScaleAction) -> ExecutionResult:
        """Record a replica count change. No physical resources in simulation."""
        service = action.target
        post = self._collect_post_action_state(service)
        return ExecutionResult(
            action=action, success=True, blocked_by_guardrail=False,
            message=f"Service '{service}' scaled to {action.replicas} replica(s).",
            post_metrics=post.get("metrics", {}),
            duration_ms=500.0,
        )

    def _execute_rollback(self, action: RollbackAction) -> ExecutionResult:
        """Clear faults and restore baseline — simulates rollback to last good state."""
        service = action.target
        try:
            if self._simulator is not None:
                fi = getattr(self._simulator, "fault_injector", None)
                if fi is not None:
                    fi.clear(service)
        except Exception as exc:
            logger.error("Rollback failed for %s: %s", service, exc)
            return ExecutionResult(
                action=action, success=False, blocked_by_guardrail=False,
                message=f"Rollback failed: {exc}",
            )

        post = self._collect_post_action_state(service)
        return ExecutionResult(
            action=action, success=True, blocked_by_guardrail=False,
            message=f"Service '{service}' rolled back — baseline restored.",
            post_metrics=post.get("metrics", {}),
            post_logs=post.get("logs", []),
            duration_ms=3000.0,
        )

    def _execute_alert(self, action: AlertAction) -> ExecutionResult:
        """Log the on-call alert. No external paging system in simulation."""
        logger.warning(
            "[ALERT %s] %s — %s", action.severity, action.target, action.message
        )
        return ExecutionResult(
            action=action, success=True, blocked_by_guardrail=False,
            message=f"On-call alerted ({action.severity}): {action.message}",
            duration_ms=100.0,
        )

    def _execute_no_action(self, action: NoAction) -> ExecutionResult:
        """Record the no-action decision without modifying any state."""
        return ExecutionResult(
            action=action, success=True, blocked_by_guardrail=False,
            message=f"No action taken: {action.reason}",
            duration_ms=0.0,
        )

    def _collect_post_action_state(self, service: str) -> dict[str, Any]:
        """Gather current metrics for a service post-execution.

        Returns dict with 'metrics' and 'logs' keys. Both empty if no simulator.
        """
        result: dict[str, Any] = {"metrics": {}, "logs": []}
        if self._simulator is None:
            return result

        mg = getattr(self._simulator, "metrics_generator", None)
        if mg is not None:
            snapshot = mg.get_latest_snapshot(service)
            if snapshot is not None:
                result["metrics"] = snapshot.to_dict()

        return result
