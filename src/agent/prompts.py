"""LLM prompt templates for the LogSentry ReAct agent.

All prompts are assembled as functions so dynamic context (service topology,
anomaly details, recent logs) can be injected at call time rather than scattered
across the codebase. Supports both OpenAI and Anthropic message formats.
"""

from __future__ import annotations

import json
from typing import Any

# Service topology injected into the system prompt
FCT_SERVICE_TOPOLOGY: str = """
Services and their upstream dependencies:
  transaction-validator (port 8001)
    └── fraud-check-service (port 8002)
          └── title-search-service (port 8004)
    └── document-processor (port 8003)
          └── title-search-service (port 8004)
  title-search-service (port 8004) — leaf service, no dependencies
"""

# Available remediation actions (injected into prompts)
AVAILABLE_ACTIONS: str = """
Available actions (respond with exactly one JSON block per action):
  {"action": "restart_service",  "target": "<service-name>", "reason": "<str>"}
  {"action": "scale_service",    "target": "<service-name>", "replicas": <int>, "reason": "<str>"}
  {"action": "rollback_service", "target": "<service-name>", "reason": "<str>"}
  {"action": "alert_on_call",    "target": "<service-name>", "severity": "P1|P2|P3", "message": "<str>"}
  {"action": "no_action",        "reason": "<str>"}
"""


def build_system_prompt() -> str:
    """Build the static system prompt that establishes the agent's role and context.

    Sent once as the 'system' role at the start of every ReAct session.
    """
    return f"""You are LogSentry, an autonomous AIOps agent for the FCT (Financial Compliance Toolkit) microservice platform.

Your job is to diagnose anomalies detected by the monitoring system and execute targeted remediation actions.
You follow the ReAct loop: Observe → Think → Act → Observe → ... until the incident is resolved or you reach the step limit.

When reasoning, consider:
  - Whether the flagged service is the root cause or a downstream victim
  - The likely fault type (crash, latency_spike, memory_leak, oom, connection_failure)
  - The least disruptive remediation that restores normal operation

Service topology:
{FCT_SERVICE_TOPOLOGY}
{AVAILABLE_ACTIONS}
Rules:
  - Emit exactly one action per Act step as a raw JSON block with no surrounding text.
  - Prefer targeted restarts over broad escalations unless fault severity warrants it.
  - If uncertain about root cause, emit alert_on_call rather than guessing.
  - Do not repeat the same action twice in one session.
"""


def build_observe_prompt(
    service: str,
    anomaly_score: float,
    triggered_metrics: list[str],
    recent_logs: list[dict[str, Any]],
    metric_snapshot: dict[str, Any],
) -> str:
    """Build the initial Observation prompt describing the detected anomaly.

    Args:
        service:           Name of the anomalous service.
        anomaly_score:     Ensemble anomaly score (0–1).
        triggered_metrics: Metric names that exceeded thresholds.
        recent_logs:       Last N log entries for the service (as dicts).
        metric_snapshot:   Latest MetricSnapshot serialized to dict.

    Returns:
        Formatted observation string for the first user/human turn.
    """
    triggered_str = ", ".join(triggered_metrics) if triggered_metrics else "none"
    metrics_str = json.dumps(metric_snapshot, indent=2)

    # Show last 5 log lines to keep prompt concise
    log_lines = ""
    for entry in recent_logs[-5:]:
        level = entry.get("level", "?")
        message = entry.get("message", "")
        log_lines += f"  [{level}] {message}\n"
    if not log_lines:
        log_lines = "  (no recent logs)\n"

    return (
        f"OBSERVATION:\n"
        f"Service '{service}' has triggered an anomaly alert.\n"
        f"  Anomaly score : {anomaly_score:.3f}  (threshold 0.5)\n"
        f"  Triggered     : {triggered_str}\n\n"
        f"Latest metrics:\n{metrics_str}\n\n"
        f"Recent logs:\n{log_lines}\n"
        f"Begin your analysis."
    )


def build_think_prompt(step: int, max_steps: int) -> str:
    """Build the instruction prompt that asks the LLM to emit a Thought.

    Args:
        step:      Current reasoning step (1-indexed).
        max_steps: Maximum steps allowed before the session is forcibly closed.

    Returns:
        Formatted prompt requesting a Thought from the LLM.
    """
    return (
        f"Step {step}/{max_steps} — THINK:\n"
        f"Reason about the root cause. Is this service the origin of the fault, "
        f"or a victim of an upstream/downstream failure? "
        f"Write your thought, then I will ask for an action."
    )


def build_action_prompt() -> str:
    """Build the prompt instructing the LLM to emit a structured JSON action.

    Returns:
        Formatted prompt requesting a single JSON action block.
    """
    return (
        "ACT:\n"
        "Based on your thought, choose one action from the available actions "
        "in your system prompt. Respond with a single valid JSON block and nothing else."
    )


def build_observation_from_action(
    action: dict[str, Any],
    execution_result: dict[str, Any],
) -> str:
    """Build the Observation turn that follows an executed action.

    Injects the executor result so the LLM can ground its next Thought
    in real post-action data.

    Args:
        action:           The action dict that was executed.
        execution_result: Result dict returned by the executor.

    Returns:
        Formatted observation string.
    """
    action_str = json.dumps(action)
    result_str = json.dumps(execution_result, indent=2)
    return (
        f"OBSERVATION (after action):\n"
        f"Executed : {action_str}\n"
        f"Result   :\n{result_str}"
    )


def build_rca_report_prompt(
    service: str,
    reasoning_trace: list[dict[str, Any]],
) -> str:
    """Build the final prompt requesting a structured RCA report.

    The report is returned as JSON and stored in the incident record.

    Args:
        service:         Primary affected service.
        reasoning_trace: Full list of Thought/Action/Observation dicts from the run.

    Returns:
        Formatted prompt requesting a JSON RCA report.
    """
    trace_str = json.dumps(reasoning_trace, indent=2)
    return (
        f"Generate a structured RCA report for this incident.\n\n"
        f"Affected service: {service}\n"
        f"Reasoning trace:\n{trace_str}\n\n"
        f"Respond with a single JSON object:\n"
        f"{{\n"
        f'  "root_cause_service": "<service-name>",\n'
        f'  "fault_type": "<crash|latency_spike|memory_leak|oom|connection_failure|unknown>",\n'
        f'  "confidence": <0.0-1.0>,\n'
        f'  "summary": "<1-2 sentence description>",\n'
        f'  "actions_taken": ["<action1>", ...],\n'
        f'  "resolved": <true|false>\n'
        f"}}"
    )


def format_messages_openai(
    system: str,
    turns: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Format a system prompt + turns into OpenAI ChatCompletion format.

    Args:
        system: System prompt string.
        turns:  List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        Message list ready for openai.chat.completions.create().
    """
    return [{"role": "system", "content": system}, *turns]


def format_messages_anthropic(
    system: str,
    turns: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Format a system prompt + turns into Anthropic Messages format.

    Args:
        system: System prompt string.
        turns:  List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        Tuple of (system_string, messages_list) for anthropic.messages.create().
    """
    return system, turns
