# AI Techniques — LogSentry Agent

Detailed explanation of every AI and ML technique used in the LogSentry pipeline.

---

## 1. Drain Algorithm — Structured Log Parsing

**File:** `src/detection/log_parser.py`

**What it is:**
Drain is an online log parsing algorithm that groups raw log lines into structured templates in a single pass. It uses a fixed-depth parse tree where log tokens are matched against known templates or used to create new ones.

**Why it matters here:**
Raw log lines like `"Transaction TXN-84721 failed after 3 retries"` and `"Transaction TXN-11093 failed after 7 retries"` are semantically identical — only the variable values differ. Drain extracts the template `"Transaction <*> failed after <*> retries"` and discards the noise. This lets the detection layer reason about *error template frequency* rather than raw string content.

**How it's used:**
- Each tick, raw log entries are fed into `LogParser.parse_batch()`
- Drain groups them into templates and returns error rate and template frequency features
- These features are merged into the feature vector used by the ensemble detector

---

## 2. Z-Score Statistical Anomaly Detection

**File:** `src/detection/statistical_detector.py`

**What it is:**
Z-score measures how many standard deviations a new observation is from the rolling mean of recent values.

```
z = (x - μ) / σ
```

A high absolute Z-score means the new value is unusually far from recent history.

**Why it matters here:**
Z-score is fast, interpretable, and requires no training data. It catches sudden changes in individual metrics (e.g., CPU jumps from 35% to 95%) that would be obvious to a human operator.

**How it's used:**
- A rolling window of the last N metric snapshots is maintained per service per metric
- On each new snapshot, Z-scores are computed for: `cpu_percent`, `memory_mb`, `latency_ms`, `error_rate`, `request_rate`, `active_connections`
- Any metric exceeding `z_score_threshold` (default: 3.0) is flagged
- The fraction of flagged metrics becomes the `stat_score` (0.0–1.0)

**Tuning:**
- Threshold of 3.0 means ~0.3% false positive rate under Gaussian noise
- Lower threshold → more sensitive but more false positives

---

## 3. Isolation Forest — ML Anomaly Detection

**File:** `src/detection/ml_detector.py`

**What it is:**
Isolation Forest is an unsupervised anomaly detection algorithm that works by randomly partitioning the feature space. Anomalous points require fewer splits to isolate, resulting in shorter path lengths through the forest.

**Why it matters here:**
Unlike Z-score, Isolation Forest can detect *multivariate* anomalies — cases where no single metric is individually abnormal, but the *combination* is unusual (e.g., CPU is normal but latency is high AND error rate is elevated simultaneously).

**How it's used:**
- During the 20-round warm-up phase, feature vectors from normal operation are collected
- A separate Isolation Forest model is trained per service (each service has different baselines)
- Raw sklearn `decision_function` scores (negative = anomalous) are normalized to [0, 1] via sigmoid:
  ```
  normalized = 1 / (1 + exp(raw_score))
  ```
- Services that haven't finished warm-up return a neutral score of 0.0

**Key parameters (config.yaml):**
```yaml
isolation_forest:
  contamination: 0.1     # expected anomaly rate in training data
  n_estimators: 100      # number of trees
```

---

## 4. Ensemble Scoring

**File:** `src/detection/feature_extractor.py`

**What it is:**
A weighted linear combination of the statistical and ML anomaly scores.

```
ensemble_score = 0.4 × stat_score + 0.6 × ml_score
is_anomaly     = ensemble_score ≥ 0.5
```

**Why this combination:**
- Z-score is fast and catches sudden single-metric spikes immediately — even before the ML model is warm
- Isolation Forest captures subtle multivariate patterns that Z-score misses
- Weighting ML at 0.6 because multivariate detection is generally more reliable once the model is trained
- The 0.4/0.6 split is configurable via `config.yaml`

**Graceful degradation:**
During warm-up, Isolation Forest returns 0.0, so the ensemble effectively runs on Z-score alone until the models are trained. This prevents a cold-start false-positive burst.

---

## 5. ReAct Agent Pattern (Reason + Act)

**File:** `src/agent/react_agent.py`

**What it is:**
ReAct (Reasoning + Acting) is a prompting pattern for LLMs that interleaves natural language reasoning steps with explicit actions. The agent produces a *thought* explaining its reasoning, then an *action* to execute, then receives an *observation* as feedback before deciding the next step.

```
Observe (anomaly context)
  └─▶ Think: "title-search-service latency is 15x baseline..."
        └─▶ Act: {"action": "restart_service", "target": "title-search-service"}
              └─▶ Observe: "Service restarted. Latency back to 18ms."
                    └─▶ Think: "Root cause confirmed..."
                          └─▶ Act: {"action": "no_action", "reason": "resolved"}
                                └─▶ Final RCA report
```

**Why it matters here:**
A single LLM call asking "what's wrong?" often produces vague answers. ReAct forces the model to break down its reasoning, take concrete steps, and update its understanding based on real feedback — exactly how a human SRE would investigate an incident.

**How it's used:**
- On anomaly detection, an `AgentContext` is built with: service name, ensemble score, triggered metrics, recent logs, and the current metric snapshot
- The agent is given the service dependency graph in its system prompt so it can reason about cascading failures
- Up to `max_reasoning_steps` (default: 5) Think → Act → Observe cycles are run
- After the loop, a final LLM call produces a structured RCA report as JSON

**Structured output:**
Actions are constrained to a validated schema via Pydantic:
- `restart_service` — target service name
- `scale_service` — target service + replica count
- `rollback_service` — target service + version
- `alert_on_call` — severity (P1/P2/P3) + message
- `no_action` — reason (terminates the loop)

---

## 6. LLM Root Cause Analysis

**File:** `src/agent/prompts.py`

**What it is:**
After the ReAct loop concludes, a final LLM call produces a structured RCA (Root Cause Analysis) report.

**Output schema:**
```json
{
  "root_cause_service": "title-search-service",
  "root_cause_type": "latency_spike",
  "summary": "title-search-service latency exceeded 15× baseline...",
  "confidence": 0.92,
  "affected_services": ["fraud-check-service", "document-processor"],
  "recommended_action": "Monitor for recurrence; consider connection pool tuning",
  "reasoning_steps": 3
}
```

**Dependency graph awareness:**
The system prompt explicitly describes the FCT service topology:
```
transaction-validator
  ├── fraud-check-service → title-search-service
  └── document-processor  → title-search-service
```
This lets the LLM correctly identify that an anomaly on `fraud-check-service` is likely a cascade from `title-search-service`, not an independent failure.

---

## 7. Safety Guardrails

**File:** `src/remediation/guardrails.py`

**What it is:**
Not an AI technique per se, but a critical safety layer that prevents the AI agent from making incidents worse.

**Rules enforced:**
- **Max restarts:** No more than N restarts per service per session (default: 3)
- **Cooldown window:** Minimum gap between consecutive restarts (default: 300s)
- **Auto-escalation:** After M consecutive failures, force a human alert instead of retrying automation (default: 2)

**Why it matters:**
Without guardrails, an agent diagnosing a persistent fault could restart the same service in a tight loop, amplifying the incident. The guardrails ensure the agent degrades gracefully to human escalation.
