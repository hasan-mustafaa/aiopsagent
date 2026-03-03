# Assumptions & Future Improvements — LogSentry Agent

---

## Assumptions

### Simulation

- **No real infrastructure.** All microservices are simulated in-process. There are no actual containers, Kubernetes pods, or network calls. The simulator produces statistically realistic metric and log data, but actions like `restart_service` only reset in-memory state.

- **Single-node, single-process.** The pipeline runs as one Python process. In production, each component (simulator, detector, agent, dashboard) would run as an independent service with a message bus between them.

- **Gaussian metric distributions.** Service metrics are modelled as independent Gaussian random walks around hardcoded baselines. Real services exhibit autocorrelation, seasonal patterns, and cross-metric dependencies that a simple Gaussian model doesn't capture.

- **Synchronous agent loop.** The ReAct agent blocks the main pipeline thread while reasoning. A production system would run the agent asynchronously and continue collecting metrics in parallel.

- **Single LLM provider per run.** The `agent.llm_provider` config key selects either OpenAI or Anthropic for the entire session. There is no fallback or load balancing between providers.

- **Drain algorithm warm-up.** The log parser needs to see enough log volume to build a stable template library. Early in a run, templates may be incomplete and error rate features less reliable.

- **Static service topology.** The FCT service dependency graph is hardcoded in config and in the agent's system prompt. Adding or removing services requires updating both.

### Detection

- **Z-score assumes stationarity.** The statistical detector computes rolling means and standard deviations. If a service's normal operating point drifts slowly over time (e.g., gradual memory growth), the rolling window will adapt — but very slow drifts may not trigger the threshold.

- **Isolation Forest trained on warm-up only.** The ML model is trained once during the 20-round warm-up phase and never retrained. Concept drift (changing normal behaviour over time) is not handled.

- **Fixed ensemble weights.** The 0.4/0.6 statistical/ML weighting was chosen empirically for this simulation. Different environments may require different calibration.

### Remediation

- **Actions are idempotent in simulation.** Restarting a service just clears its faults and resets its metric baseline. In a real system, restarts have side effects (in-flight requests dropped, cache invalidated, etc.).

- **No rollback versioning.** `rollback_service` is implemented identically to `restart_service` in the simulator (both clear faults and restore baseline). A real rollback would revert to a specific previous artifact version.

- **Alert severity is LLM-determined.** The agent decides whether to send a P1, P2, or P3 alert based on its reasoning. There is no formal severity calculation or escalation policy beyond the guardrail thresholds.

---

## Known Limitations

- **Log stream is per-tick only.** The log buffer holds the last 200 entries across all services. Older log context is discarded and not available to the agent.

- **No persistent storage.** All pipeline state (metric history, agent results, remediation log) is held in memory and in `data/dashboard_state.json`. The state resets when the pipeline restarts.

- **Dashboard polling latency.** The dashboard reads `dashboard_state.json` every 2 seconds. There is a maximum 2-second lag between a pipeline event and its appearance on the dashboard.

- **LLM prompt token limits.** For long-running sessions with many anomaly events, the observation context injected into the LLM prompt may approach token limits. No truncation or summarization is applied.

- **No multi-anomaly handling.** If two services detect anomalies in the same tick, they are handled sequentially. The second agent invocation does not know about the first.

---

## Future Improvements

### Detection

- **Online model retraining.** Periodically retrain the Isolation Forest on a sliding window of recent data to handle concept drift and changing operating conditions.

- **Seasonal decomposition.** Add time-of-day and day-of-week seasonality removal before anomaly scoring to reduce false positives during expected traffic spikes.

- **Cross-service correlation.** Detect cascading anomalies by correlating anomaly events across the dependency graph, rather than treating each service independently.

- **Trace-based analysis.** Add distributed trace IDs to log entries to correlate a single request's journey across all four services, enabling precise latency attribution.

### Agent

- **Memory across incidents.** Give the agent a persistent incident history so it can recognize recurring patterns (e.g., "this service restarts every 6 hours — likely a memory leak, not a one-off crash").

- **Confidence calibration.** Add an explicit uncertainty estimate to the RCA output and only auto-remediate when confidence exceeds a threshold; otherwise escalate to human.

- **Multi-agent parallelism.** Run a separate agent instance per anomalous service in parallel, rather than sequentially, to reduce incident response latency.

- **Tool use / function calling.** Replace free-form JSON parsing with native LLM function calling (OpenAI tool use / Anthropic tool use) for more reliable action extraction.

### Remediation

- **Kubernetes integration.** Replace the simulated executor with a real Kubernetes client (`kubectl rollout restart`, `kubectl scale`) for actual service management.

- **PagerDuty / Slack integration.** Wire `alert_on_call` to a real paging system (PagerDuty Events API) and notification channel (Slack webhook).

- **Rollback with versioning.** Track deployed artifact versions and implement genuine rollback to the last known-good version rather than a simple fault clear.

- **Chaos engineering integration.** Connect the fault injector to a real chaos engineering tool (Chaos Monkey, Litmus) to validate agent behaviour against production-like failure modes.

### Dashboard

- **Multi-incident timeline.** Show all historical anomaly and remediation events on a shared timeline so operators can see incident patterns over time.

- **Alert deduplication.** Suppress duplicate anomaly cards when the same service fires multiple anomalies within a short window.

- **Live fault injection UI.** Add a sidebar panel to inject faults directly from the dashboard without requiring a code change or CLI call.

- **Exportable RCA reports.** Allow one-click PDF/JSON export of the structured RCA report for incident post-mortems.
