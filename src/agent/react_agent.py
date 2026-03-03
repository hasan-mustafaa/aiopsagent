"""ReAct agent — LLM reasoning loop for root cause analysis and remediation.

Implements Observe → Think → Act → Observe → ... until the anomaly resolves,
the agent emits no_action, or max_reasoning_steps is reached. After the loop,
generates a structured RCA report via a final LLM call.

Supports OpenAI and Anthropic as LLM providers (set in config.yaml → agent).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agent.action_planner import Action, ActionParseError, ActionPlanner
from src.agent.prompts import (
    build_action_prompt,
    build_observation_from_action,
    build_observe_prompt,
    build_rca_report_prompt,
    build_system_prompt,
    build_think_prompt,
    format_messages_anthropic,
    format_messages_openai,
)
from src.detection.feature_extractor import FeatureVector

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """A single Thought / Action / Observation triplet from one ReAct iteration."""

    step_number: int
    thought: str
    action: Action | None
    observation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentContext:
    """All context passed to the agent when an anomaly is detected.

    Assembled by main.py from the detection layer outputs.
    """

    service: str
    anomaly_score: float
    triggered_metrics: list[str]
    feature_vector: FeatureVector
    recent_logs: list[dict[str, Any]]
    metric_snapshot: dict[str, Any]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentResult:
    """Full output of a completed ReAct agent run.

    Attributes:
        context:         The AgentContext that triggered this run.
        reasoning_trace: Ordered ReasoningStep list (the full chain).
        actions_taken:   All Action objects that were executed.
        rca_report:      Structured RCA dict from the final LLM call.
        resolved:        True if anomaly score dropped below threshold.
        escalated:       True if max steps reached without resolution.
    """

    context: AgentContext
    reasoning_trace: list[ReasoningStep]
    actions_taken: list[Action]
    rca_report: dict[str, Any]
    resolved: bool
    escalated: bool
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReActAgent:
    """LLM-powered ReAct agent for AIOps root cause analysis and remediation.

    Orchestrates the Observe → Think → Act loop, parsing LLM output with
    ActionPlanner and dispatching actions to the executor.
    """

    def __init__(self, agent_config: dict[str, Any]) -> None:
        """Initialize from the [agent] section of config.yaml."""
        self._provider: str = agent_config.get("llm_provider", "openai")
        self._model: str = agent_config.get("model", "gpt-4o-mini")
        self._max_steps: int = agent_config.get("max_reasoning_steps", 5)
        self._confidence_threshold: float = agent_config.get("confidence_threshold", 0.7)

        self._planner = ActionPlanner()
        self._client: Any = None  # Lazy-initialized on first LLM call
        self._system_prompt: str = build_system_prompt()

    def run(
        self,
        context: AgentContext,
        executor: Any | None = None,
    ) -> AgentResult:
        """Execute the full ReAct loop for a detected anomaly.

        Args:
            context:  AgentContext assembled from the detection layer.
            executor: Optional Executor to dispatch actions. If None, dry-run mode.

        Returns:
            AgentResult with reasoning trace, actions taken, and RCA report.
        """
        messages = self._build_initial_messages(context)
        trace: list[ReasoningStep] = []
        actions_taken: list[Action] = []
        resolved = False

        for step_num in range(1, self._max_steps + 1):
            # Ask LLM to Think
            messages.append({"role": "user", "content": build_think_prompt(step_num, self._max_steps)})
            thought = self._call_llm(messages)
            messages.append({"role": "assistant", "content": thought})

            # Ask LLM to Act
            messages.append({"role": "user", "content": build_action_prompt()})
            action_text = self._call_llm(messages)
            messages.append({"role": "assistant", "content": action_text})

            # Parse action from LLM output
            action = None
            observation = ""
            try:
                actions = self._planner.parse(action_text)
                action = actions[0] if actions else None
            except ActionParseError as exc:
                logger.warning("Failed to parse action at step %d: %s", step_num, exc)
                observation = f"ERROR: Could not parse action — {exc}. Please try again with valid JSON."

            if action is not None:
                # Check for no_action — agent decided no intervention needed
                if action.action == "no_action":  # type: ignore[union-attr]
                    observation = "Agent decided no action is needed."
                elif executor is not None:
                    # Execute the action and capture result
                    exec_result = executor.execute(action)
                    observation = build_observation_from_action(
                        action.model_dump(),
                        exec_result.to_dict() if hasattr(exec_result, "to_dict") else {"result": str(exec_result)},
                    )
                    actions_taken.append(action)
                else:
                    # Dry-run mode
                    observation = build_observation_from_action(
                        action.model_dump(),
                        {"success": True, "message": "Dry-run — action logged but not executed."},
                    )
                    actions_taken.append(action)

            step = ReasoningStep(
                step_number=step_num,
                thought=thought,
                action=action,
                observation=observation,
            )
            trace.append(step)

            # Feed observation back for next iteration
            if observation:
                messages.append({"role": "user", "content": observation})

            # Check termination conditions
            if action is not None and action.action == "no_action":  # type: ignore[union-attr]
                break
            if self._check_resolved(context.service):
                resolved = True
                break

        escalated = not resolved and len(trace) >= self._max_steps
        rca_report = self._generate_rca_report(context, trace)

        return AgentResult(
            context=context,
            reasoning_trace=trace,
            actions_taken=actions_taken,
            rca_report=rca_report,
            resolved=resolved,
            escalated=escalated,
        )

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the configured LLM and return response text.

        Lazily initializes the LLM client on first call.

        Raises:
            RuntimeError: If the LLM call fails.
        """
        if self._client is None:
            self._client = self._init_llm_client()

        try:
            if self._provider == "openai":
                formatted = format_messages_openai(self._system_prompt, messages)
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=formatted,
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""

            elif self._provider == "anthropic":
                system, turns = format_messages_anthropic(self._system_prompt, messages)
                response = self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=turns,
                    max_tokens=1024,
                    temperature=0.2,
                )
                return response.content[0].text

            else:
                raise RuntimeError(f"Unsupported LLM provider: {self._provider}")

        except Exception as exc:
            raise RuntimeError(f"LLM call failed ({self._provider}/{self._model}): {exc}") from exc

    def _build_initial_messages(self, context: AgentContext) -> list[dict[str, str]]:
        """Construct the initial message list with the first Observation turn."""
        observe = build_observe_prompt(
            service=context.service,
            anomaly_score=context.anomaly_score,
            triggered_metrics=context.triggered_metrics,
            recent_logs=context.recent_logs,
            metric_snapshot=context.metric_snapshot,
        )
        return [{"role": "user", "content": observe}]

    def _append_step(
        self,
        messages: list[dict[str, str]],
        step: ReasoningStep,
    ) -> list[dict[str, str]]:
        """Append a completed ReasoningStep to the message history."""
        messages.append({"role": "assistant", "content": step.thought})
        if step.observation:
            messages.append({"role": "user", "content": step.observation})
        return messages

    def _generate_rca_report(
        self,
        context: AgentContext,
        trace: list[ReasoningStep],
    ) -> dict[str, Any]:
        """Make a final LLM call to produce a structured JSON RCA report.

        Falls back to a default report if parsing fails.
        """
        # Build a serializable reasoning trace
        trace_dicts = [
            {
                "step": s.step_number,
                "thought": s.thought,
                "action": s.action.model_dump() if s.action else None,
                "observation": s.observation,
            }
            for s in trace
        ]

        rca_prompt = build_rca_report_prompt(context.service, trace_dicts)
        messages = self._build_initial_messages(context)
        messages.append({"role": "user", "content": rca_prompt})

        try:
            response = self._call_llm(messages)
            # Try to extract JSON from the response
            planner = ActionPlanner()
            blocks = planner._extract_json_blocks(response)
            if blocks:
                return blocks[0]
        except Exception as exc:
            logger.warning("RCA report generation failed: %s", exc)

        # Fallback report
        return {
            "root_cause_service": context.service,
            "fault_type": "unknown",
            "confidence": 0.0,
            "summary": "RCA report generation failed — see reasoning trace.",
            "actions_taken": [s.action.model_dump() for s in trace if s.action],
            "resolved": False,
        }

    def _check_resolved(self, service: str) -> bool:
        """Check if the anomaly has resolved. Always returns False without live detection.

        In production, this would query the detection layer for current anomaly score.
        """
        # The detection layer isn't wired in directly — resolution is determined
        # by the orchestrator (main.py) re-checking after action execution.
        return False

    def _init_llm_client(self) -> Any:
        """Initialize the LLM client based on llm_provider config.

        Raises:
            ValueError: If provider is not 'openai' or 'anthropic'.
            RuntimeError: If the required API key env var is not set.
        """
        if self._provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set")
            import openai
            return openai.OpenAI(api_key=api_key)

        else:
            raise ValueError(
                f"Unknown LLM provider '{self._provider}'. Must be 'openai' or 'anthropic'."
            )
