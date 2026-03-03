"""Entry point for the LogSentry Agent pipeline.

Orchestrates the full system lifecycle:
  1. Load configuration from config/config.yaml.
  2. Start the simulator (metrics generator + fault injector).
  3. Warm up the detection pipeline (fill stat windows, train ML models).
  4. Run the main loop: detect anomalies → invoke ReAct agent → remediate.

Run with:
    python -m src.main
    python -m src.main --dry-run
    python -m src.main --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

_STATE_FILE = Path("data/dashboard_state.json")

from src.agent.react_agent import AgentContext, ReActAgent
from src.detection.feature_extractor import FeatureExtractor
from src.detection.log_parser import LogParser
from src.detection.ml_detector import MIN_TRAINING_SAMPLES, MLAnomalyDetector
from src.detection.statistical_detector import StatisticalDetector
from src.remediation.executor import Executor
from src.remediation.guardrails import Guardrails
from src.simulator.fault_injector import FaultInjector
from src.simulator.log_generator import LogGenerator
from src.simulator.metrics_generator import MetricsGenerator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="LogSentry Agent — AIOps pipeline")
    parser.add_argument(
        "--config", type=Path, default=Path("config/config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log actions without executing remediation",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip launching the Streamlit dashboard",
    )
    return parser.parse_args()


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


_MAX_LOG_BUFFER = 200  # keep last N log entries for the dashboard


@dataclass
class SimulatorState:
    """Container passed to Executor so it can manipulate simulator state."""

    fault_injector: FaultInjector | None = None
    metrics_generator: MetricsGenerator | None = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Wires all components and runs the main detection → agent → remediation loop."""

    def __init__(self, config: dict, dry_run: bool = False) -> None:
        self._config = config
        self._dry_run = dry_run
        self._running = False

        services = [s["name"] for s in config["simulator"]["services"]]
        self._services = services
        self._tick_interval = config["simulator"]["metrics_interval_seconds"]

        # Simulator
        self._metrics_gen = MetricsGenerator(config["simulator"])
        self._metrics_iter = self._metrics_gen.generate()
        self._log_gen = LogGenerator(config["simulator"], {
            s: {"healthy": True, "fault_type": None} for s in services
        })
        self._fault_injector = FaultInjector(
            config["simulator"], self._log_gen, self._metrics_gen,  # type: ignore[arg-type]
        )

        # Detection
        self._log_parser = LogParser(config["detection"])
        self._feature_extractor = FeatureExtractor(config["detection"])
        self._stat_detector = StatisticalDetector(config["detection"])
        if_conf = config["detection"].get("isolation_forest", {})
        self._ml_detector = MLAnomalyDetector(
            contamination=if_conf.get("contamination", 0.1),
            n_estimators=if_conf.get("n_estimators", 100),
        )

        # Agent + Remediation
        self._agent = ReActAgent(config["agent"])
        self._guardrails = Guardrails(config["remediation"])
        self._sim_state = SimulatorState(
            fault_injector=self._fault_injector,
            metrics_generator=self._metrics_gen,
        )
        self._executor = Executor(
            config["remediation"], self._guardrails,
            simulator_state=self._sim_state if not dry_run else None,
        )

        # Threshold from ensemble scoring (FeatureExtractor.ANOMALY_THRESHOLD = 0.5)
        self._anomaly_threshold = FeatureExtractor.ANOMALY_THRESHOLD

        # Dashboard state — written to _STATE_FILE after every tick
        self._dash: dict[str, Any] = {
            "metric_history":   {s: [] for s in services},
            "log_buffer":       [],
            "anomaly_events":   [],
            "agent_results":    [],
            "remediation_log":  [],
            "service_status":   {s: "healthy" for s in services},
            "pipeline_started_at": datetime.now(timezone.utc).isoformat(),
        }

    def run(self) -> None:
        """Start the pipeline: warm up, then enter the detection loop."""
        self._running = True
        logger.info("LogSentry starting — warm-up phase")
        self._warm_up()
        # Clear metric history accumulated during warm-up — those snapshots share
        # the same timestamp (generated in a tight loop) and would compress the
        # chart x-axis into a useless vertical sliver.
        for s in self._services:
            self._dash["metric_history"][s] = []
        self._dash["anomaly_events"] = []
        logger.info("Warm-up complete. Entering detection loop (Ctrl+C to stop)")

        while self._running:
            try:
                self._tick()
                time.sleep(self._tick_interval)
            except KeyboardInterrupt:
                break

        self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        logger.info("LogSentry stopped.")

    def inject_fault(
        self, service: str, fault_type: str, duration: float = 60.0,
    ) -> None:
        """Manually inject a fault for testing/demo purposes."""
        scenario = self._fault_injector.inject(service, fault_type, duration)
        logger.info(
            "Fault injected: %s on %s (duration=%ss, id=%s)",
            fault_type, service, duration, scenario.fault_id,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _warm_up(self) -> None:
        """Collect normal metric snapshots to fill stat windows and train ML models."""
        target = max(MIN_TRAINING_SAMPLES, 20)
        training_data: dict[str, list[np.ndarray]] = {s: [] for s in self._services}

        for _ in range(target):
            # One round = one snapshot per service
            for _ in range(len(self._services)):
                snapshot = next(self._metrics_iter)
                self._stat_detector.update(snapshot)
                self._feature_extractor.ingest_metric(snapshot)

            # Extract feature vectors after each round
            for service in self._services:
                fv = self._feature_extractor.extract(service)
                if fv is not None:
                    training_data[service].append(fv.features.copy())

        # Train ML detector per service
        for service in self._services:
            samples = training_data[service]
            if len(samples) >= MIN_TRAINING_SAMPLES:
                self._ml_detector.train(np.stack(samples), service)
                logger.info("ML model trained for %s (%d samples)", service, len(samples))
            else:
                logger.warning(
                    "Not enough samples to train ML for %s (%d/%d)",
                    service, len(samples), MIN_TRAINING_SAMPLES,
                )

    def _tick(self) -> None:
        """One simulation tick: generate metrics → detect → agent if anomaly."""
        # Advance fault injector (expires timed-out faults)
        self._fault_injector.tick()

        # Generate logs for this tick and feed into dashboard buffer
        log_entries = self._log_gen.generate_tick()
        for entry in log_entries:
            self._dash["log_buffer"].append({
                "timestamp": entry.timestamp.isoformat(),
                "service":   entry.service,
                "level":     entry.level,
                "message":   entry.message,
            })
        # Cap the log buffer
        if len(self._dash["log_buffer"]) > _MAX_LOG_BUFFER:
            self._dash["log_buffer"] = self._dash["log_buffer"][-_MAX_LOG_BUFFER:]

        # Generate one snapshot per service
        for _ in range(len(self._services)):
            snapshot = next(self._metrics_iter)
            service = snapshot.service

            # Statistical detection
            stat_result = self._stat_detector.update(snapshot)

            # Feature extraction
            self._feature_extractor.ingest_metric(snapshot)
            fv = self._feature_extractor.extract(service)

            # ML detection (if model trained)
            ml_score = 0.0
            if fv is not None and self._ml_detector.is_trained(service):
                ml_result = self._ml_detector.detect(fv)
                ml_score = ml_result.anomaly_score

            # Ensemble score
            ensemble_score, is_anomaly = self._feature_extractor.compute_ensemble_score(
                stat_score=stat_result.anomaly_score,
                ml_score=ml_score,
            )

            # Update dashboard metric history (cap at 500 points per service)
            history = self._dash["metric_history"][service]
            history.append(snapshot.to_dict())
            if len(history) > 500:
                self._dash["metric_history"][service] = history[-500:]

            if is_anomaly and fv is not None:
                logger.warning(
                    "Anomaly detected on %s — score=%.3f, triggered=%s",
                    service, ensemble_score, stat_result.triggered_metrics,
                )
                self._dash["anomaly_events"].append({
                    "service":          service,
                    "anomaly_score":    ensemble_score,
                    "triggered_metrics": stat_result.triggered_metrics,
                    "detected_at":      datetime.now(timezone.utc).isoformat(),
                })
                self._handle_anomaly(
                    service, ensemble_score, stat_result, fv, snapshot,
                )

        # Update service status based on active faults
        active_faults = {s.service: s.fault_type for s in self._fault_injector.active_faults()}
        for svc in self._services:
            if svc not in active_faults:
                self._dash["service_status"][svc] = "healthy"
            elif active_faults[svc] == "crash":
                self._dash["service_status"][svc] = "down"
            else:
                self._dash["service_status"][svc] = "degraded"

        # Sync remediation log from executor and persist state to disk
        self._dash["remediation_log"] = [
            r.to_dict() for r in self._executor.execution_log()
        ]
        self._save_dashboard_state()

    def _handle_anomaly(
        self, service, ensemble_score, stat_result, fv, snapshot,
    ) -> None:
        """Build AgentContext and invoke the ReAct agent."""
        context = AgentContext(
            service=service,
            anomaly_score=ensemble_score,
            triggered_metrics=stat_result.triggered_metrics,
            feature_vector=fv,
            recent_logs=[
                e for e in self._dash["log_buffer"][-20:]
                if e.get("service") == service
            ],
            metric_snapshot=snapshot.to_dict(),
        )

        executor = self._executor if not self._dry_run else None

        try:
            result = self._agent.run(context, executor=executor)
        except RuntimeError as exc:
            # LLM call failed (no API key, network error, etc.)
            logger.error("Agent failed for %s: %s", service, exc)
            return

        if result.escalated:
            logger.warning(
                "Incident escalated for %s — agent exhausted %d steps",
                service, len(result.reasoning_trace),
            )
        if result.rca_report:
            logger.info(
                "RCA: %s — %s (confidence=%.2f)",
                result.rca_report.get("root_cause_service", "?"),
                result.rca_report.get("summary", ""),
                result.rca_report.get("confidence", 0.0),
            )

        # Append serialized agent result for the dashboard
        self._dash["agent_results"].append({
            "context": {
                "service":          context.service,
                "anomaly_score":    context.anomaly_score,
                "triggered_metrics": context.triggered_metrics,
                "detected_at":      context.detected_at.isoformat(),
            },
            "reasoning_trace": [
                {
                    "step_number": s.step_number,
                    "thought":     s.thought,
                    "action":      s.action.model_dump() if s.action else None,
                    "observation": s.observation,
                }
                for s in result.reasoning_trace
            ],
            "rca_report": result.rca_report,
            "resolved":   result.resolved,
            "escalated":  result.escalated,
            "completed_at": result.completed_at.isoformat(),
        })


    def _save_dashboard_state(self) -> None:
        """Write dashboard data to _STATE_FILE so the Streamlit process can read it."""
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".tmp")
        try:
            self._dash["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(tmp, "w") as f:
                json.dump(self._dash, f, default=str)
            tmp.replace(_STATE_FILE)
        except Exception as exc:
            logger.debug("Dashboard state write failed: %s", exc)


def build_pipeline(config: dict, dry_run: bool = False) -> Pipeline:
    """Instantiate and wire all pipeline components from config.

    Args:
        config:  Loaded configuration dictionary.
        dry_run: If True, executor logs actions without applying them.

    Returns:
        A fully wired Pipeline ready to run.
    """
    return Pipeline(config, dry_run=dry_run)


def main(config_path: Optional[Path] = None) -> None:
    """Main entry point. Parses args, loads config, builds and runs the pipeline."""
    load_dotenv()  # Load .env file for OPENAI_API_KEY and other env vars

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    path = config_path or args.config
    config = load_config(path)

    pipeline = build_pipeline(config, dry_run=args.dry_run)
    pipeline.run()


if __name__ == "__main__":
    main()
