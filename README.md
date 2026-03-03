# LogSentry Agent

> AI-driven AIOps system that monitors simulated microservice logs and metrics, detects anomalies using ML, diagnoses root causes with a ReAct LLM agent, and automatically remediates issues.

---

## Overview

LogSentry Agent is a self-healing observability pipeline built for simulated FCT (Financial Crime & Technology) microservices. It combines traditional statistical methods with modern ML and LLM-powered reasoning to detect, diagnose, and remediate operational incidents — all without human intervention.

**Core capabilities:**
- Real-time structured log generation across 4 interdependent microservices
- Fault injection (crashes, latency spikes, memory leaks, connection failures)
- Hybrid anomaly detection (Z-score + Isolation Forest ensemble)
- LLM-based root cause analysis via the ReAct reasoning pattern
- Automated remediation with safety guardrails
- Live Streamlit dashboard with agent reasoning traces

---

## Architecture

Full system design: [ARCHITECTURE.md](ARCHITECTURE.md)

![LogSentry Architecture](docs/architecture.png)

The pipeline has five stages:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Simulator** | Generates metrics + logs for 4 FCT services; injects faults |
| 2 | **Detection** | Z-score + Isolation Forest ensemble; scores anomalies 0-1 |
| 3 | **Agent** | ReAct LLM loop (Observe → Think → Act → RCA report) |
| 4 | **Remediation** | Executes actions against simulator with safety guardrails |
| 5 | **Dashboard** | Live Streamlit UI reading from shared JSON state file |

---

## AI Techniques

Full explanation: [AI_TECHNIQUES.md](AI_TECHNIQUES.md)

| Technique | File | Purpose |
|-----------|------|---------|
| **Drain Algorithm** | `src/detection/log_parser.py` | Mines structured templates from raw log lines |
| **Z-Score Detection** | `src/detection/statistical_detector.py` | Per-metric rolling baseline anomaly detection |
| **Isolation Forest** | `src/detection/ml_detector.py` | Multivariate ML anomaly detection per service |
| **Ensemble Scoring** | `src/detection/feature_extractor.py` | `0.4 x stat + 0.6 x ML`, triggers at score >= 0.5 |
| **ReAct Agent** | `src/agent/react_agent.py` | Observe → Think → Act LLM reasoning loop |
| **LLM RCA** | `src/agent/prompts.py` | Structured root cause report with confidence score |
| **Safety Guardrails** | `src/remediation/guardrails.py` | Restart limits, cooldown, auto-escalation |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- An OpenAI API key (or Anthropic API key)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd aiopsagent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### Configuration

All tuneable parameters are in `config/config.yaml`:

```yaml
simulator:
  metrics_interval_seconds: 5   # tick rate

detection:
  z_score_threshold: 3.0        # statistical sensitivity
  ensemble_weights:
    statistical: 0.4
    ml: 0.6

agent:
  llm_provider: "openai"        # or "anthropic"
  model: "gpt-4o-mini"
  max_reasoning_steps: 5

remediation:
  max_restarts_per_service: 3
  restart_cooldown_seconds: 300
```

### Running the Pipeline

```bash
# Start the detection + agent + remediation pipeline
python -m src.main

# Dry-run mode (logs actions without executing them)
python -m src.main --dry-run
```

### Running the Dashboard

In a separate terminal:

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The dashboard auto-refreshes every 2 seconds.

---

## How It Works

1. The **Simulator** generates structured JSON logs and time-series metrics for 4 FCT microservices every 5 seconds.
2. The **FaultInjector** injects realistic faults (crashes, latency spikes, memory leaks, connection failures) that cascade upstream through the dependency graph.
3. The **LogParser** (Drain algorithm) extracts log templates and computes error rate features.
4. The **FeatureExtractor** builds time-windowed multivariate feature vectors from metric snapshots.
5. The **Ensemble Detector** (Z-score + Isolation Forest) scores each window and triggers on score >= 0.5.
6. On anomaly detection, the **ReAct Agent** is invoked with the full anomaly context including triggered metrics and recent logs.
7. The Agent iterates Observe → Think → Act (up to 5 steps), reasoning over the service dependency graph to identify root cause.
8. The **Executor** applies the chosen action (restart, scale, rollback, alert) with guardrail safety checks before execution.
9. The **Dashboard** reflects all state changes in real time via a shared JSON state file written after every tick.

---

## Project Structure

```
aiopsagent/
├── config/
│   └── config.yaml             # Single source of truth for all parameters
├── src/
│   ├── main.py                 # Pipeline orchestrator
│   ├── simulator/
│   │   ├── metrics_generator.py    # Gaussian random-walk metric streams
│   │   ├── log_generator.py        # Structured log entry generation
│   │   └── fault_injector.py       # Fault scenario injection
│   ├── detection/
│   │   ├── log_parser.py           # Drain log template mining
│   │   ├── feature_extractor.py    # Feature vectors + ensemble scoring
│   │   ├── statistical_detector.py # Z-score rolling window detection
│   │   └── ml_detector.py          # Isolation Forest per-service model
│   ├── agent/
│   │   ├── react_agent.py          # ReAct Observe → Think → Act loop
│   │   ├── prompts.py              # LLM prompt construction
│   │   └── action_planner.py       # JSON action parsing + Pydantic models
│   ├── remediation/
│   │   ├── executor.py             # Action dispatch + simulator integration
│   │   └── guardrails.py           # Safety limits on automated actions
│   └── dashboard/
│       └── app.py                  # Streamlit live dashboard
├── tests/                      # Unit tests per component
├── docs/
│   └── architecture.png        # System architecture diagram
├── ARCHITECTURE.md             # Full system design
├── AI_TECHNIQUES.md            # AI/ML technique explanations
├── ASSUMPTIONS.md              # Assumptions and future improvements
└── .env.example                # Environment variable template
```

---

## Demo Video

> Demo video link: _coming soon_

---

## Assumptions & Future Improvements

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for a full list of assumptions, known limitations, and planned improvements.

**Key assumptions:**
- All microservices are simulated in-process (no real containers or network)
- Isolation Forest is trained once during warm-up and not retrained
- Single LLM provider per run (no fallback)

**Planned improvements:**
- Online model retraining for concept drift
- Kubernetes and PagerDuty integration for real remediation
- Multi-agent parallelism for concurrent incident handling
- Persistent incident memory across agent runs

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| ML / Detection | scikit-learn, numpy, scipy, pandas |
| Log Parsing | drain3 |
| LLM | OpenAI GPT-4o-mini or Anthropic Claude |
| Dashboard | Streamlit + Plotly |
| Config | PyYAML + Pydantic |
| Testing | pytest |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
