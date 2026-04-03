# 🔧 Pipeline Debug Env

**An OpenEnv-compatible reinforcement learning environment for debugging broken data pipelines.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-in--memory-yellow.svg)](https://duckdb.org)
[![Docker Ready](https://img.shields.io/badge/Docker-port%207860-blue.svg)](https://hub.docker.com)

---

## Overview

Pipeline Debug Env presents agents with **broken SQL data pipelines** and challenges them to diagnose and fix faults through structured actions. The environment uses DuckDB for in-memory pipeline execution, procedural fault injection across 5 fault classes, and a deterministic reward function that evaluates schema correctness, row-level accuracy, and step efficiency.

### Key Features

- **6 structured action types** — `patch_schema`, `rewrite_transform`, `add_null_guard`, `add_type_cast`, `fix_join_key`, `invert_filter`
- **5 fault classes** — Schema drift, type mismatch, null propagation, boolean inversion, filter removal
- **3 difficulty levels** — Easy (1 fault, 4 steps), Medium (2 faults, 8 steps), Hard (3 faults, 12 steps)
- **Deterministic grading** — Weighted reward: 35% schema + 35% row accuracy + 20% efficiency + clamped penalties
- **Per-episode variation** — Seed-based fault diversity ensures non-trivial evaluation variance
- **REST API** — Standard `POST /reset`, `POST /step`, `GET /state` endpoints

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server (:7860)                │
│                                                         │
│  POST /reset ──► PipelineEnvironment.reset()            │
│  POST /step  ──► PipelineEnvironment.step()             │
│  GET  /state ──► EpisodeManager.get_state()             │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           PipelineEnvironment (Orchestrator)     │   │
│  │                                                  │   │
│  │  reset(): Load Template → Execute Clean DAG →    │   │
│  │           Inject Faults → Execute Faulty DAG →   │   │
│  │           Build Observation                      │   │
│  │                                                  │   │
│  │  step():  Parse → Validate → Apply → Execute →   │   │
│  │           Grade → Observe → Update               │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────┐ ┌─────────┐ ┌──────────────────────┐   │
│  │FaultInject.│ │ Grader  │ │  ObservationBuilder  │   │
│  │ 5 classes  │ │ clamped │ │  cached + sanitized  │   │
│  └────────────┘ └─────────┘ └──────────────────────┘   │
│                                                         │
│  ┌────────────┐ ┌───────────────┐ ┌────────────────┐   │
│  │ActionParser│ │PipelineExec.  │ │EpisodeManager  │   │
│  │ semantic   │ │ DuckDB engine │ │ repeat detect  │   │
│  └────────────┘ └───────────────┘ └────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation Results

**Agent:** Heuristic (rule-based, no external API)  
**Episodes per task:** 10  
**Total runtime:** 39.48 seconds  
**Seed:** 42  

| Task   | Avg Score | Std Dev | Avg Steps | Failure Rate (< 0.70) |
|--------|-----------|---------|-----------|----------------------|
| Easy   | **0.71**  | 0.2951  | 4.0       | 20.0%                |
| Medium | **0.60**  | 0.3105  | 8.0       | 50.0%                |
| Hard   | **0.60**  | 0.3558  | 12.0      | 40.0%                |

### Observations

- **Clear difficulty progression**: Easy tasks (1 fault) are reliably fixed; Medium/Hard tasks with compound faults challenge the agent
- **Non-zero variance**: Different fault/template combinations produce score diversity (std_dev 0.29–0.36)
- **Efficient execution**: All 30 episodes complete in under 40 seconds on CPU
- **No crashes**: Fault-resilient reset with automatic seed retry eliminates 500 errors

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip / venv

### Installation

```bash
git clone <repo-url>
cd pipeline_debug_env

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn pipeline_debug_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run Evaluation

```bash
python -m pipeline_debug_env.baseline.run_eval
```

### Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

### `POST /reset`

Initialize a new episode.

**Request Body:**
```json
{"task_level": "easy"}
```

**Response:** Observation dict containing `pipeline_dag`, `error_log`, `schema_diff`, `sample_rows`, `current_score`, `step_count`, `max_steps`, `done`.

### `POST /step`

Submit an action to fix the pipeline.

**Request Body:**
```json
{
  "action_type": "rewrite_transform",
  "target_node": "clean",
  "params": {"new_sql": "SELECT * FROM ingest WHERE amount IS NOT NULL"},
  "reasoning": "Fix null propagation in clean node"
}
```

**Response:** `{observation, reward, done, info}`

### `GET /state`

Returns the current episode state including `step_count`, `current_score`, `best_score`, `score_history`, and `done`.

---

## Reward Function

The grader computes a weighted score:

```
raw_score = 0.35 × schema_score + 0.35 × row_score + 0.20 × efficiency
final_score = max(0, raw_score - clamped_penalties)
```

| Component        | Weight | Description                              |
|------------------|--------|------------------------------------------|
| Schema Score     | 0.35   | Jaccard similarity of column sets        |
| Row Score        | 0.35   | Cell-level match (order-independent)     |
| Efficiency       | 0.20   | `1 - step_count / max_steps`             |
| Penalties (max)  | 0.25   | Invalid action + repeat + regression     |

---

## Project Structure

```
pipeline_debug_env/
├── server/
│   ├── app.py                 # FastAPI endpoints
│   ├── environment.py         # Orchestrator (reset/step/state)
│   ├── pipeline_executor.py   # DuckDB execution engine
│   ├── fault_injector.py      # Dynamic fault generation
│   ├── grader.py              # Deterministic reward function
│   ├── observation_builder.py # Observation construction + caching
│   ├── action_parser.py       # Semantic action validation
│   └── episode_manager.py     # Step/score/repeat tracking
├── baseline/
│   ├── inference.py           # LLM-based agent (OpenAI-compatible)
│   ├── heuristic_agent.py     # Rule-based agent (no API needed)
│   └── run_eval.py            # Evaluation runner
├── templates/
│   ├── ecommerce_orders.yaml
│   ├── user_engagement.yaml
│   └── financial_revenue.yaml
├── models.py                  # Pydantic schemas
├── client.py                  # Async HTTP client SDK
├── __init__.py
├── Dockerfile
└── openenv.yaml
```

---

## Docker Deployment

```bash
docker build -t pipeline-debug-env .
docker run -p 7860:7860 pipeline-debug-env
```

The container exposes port 7860 and is compatible with Hugging Face Spaces deployment.

---

## Environment Variables

| Variable         | Default                              | Description                    |
|------------------|--------------------------------------|--------------------------------|
| `TASK_LEVEL`     | `easy`                               | Difficulty: easy/medium/hard   |
| `PIPELINE_SEED`  | `42`                                 | Base seed for reproducibility  |
| `PORT`           | `7860`                               | Server port                    |
| `HF_TOKEN`       | —                                    | Hugging Face API token (optional, for LLM agent) |
| `API_BASE_URL`   | `https://router.huggingface.co/v1`   | LLM API endpoint               |
| `MODEL_NAME`     | `Qwen/Qwen2.5-72B-Instruct`         | LLM model ID                   |

---

## Design Decisions

| Decision                  | Choice                        | Rationale                                           |
|---------------------------|-------------------------------|-----------------------------------------------------|
| Data Generation           | Static YAML templates         | Reproducible, no external dependencies              |
| Execution Engine          | DuckDB in-memory              | Fast SQL execution, zero config                     |
| API Protocol              | REST (FastAPI)                | OpenEnv compatible, simple integration              |
| Fault Injection           | Dynamic SQL-aware patterns    | Template-agnostic, real SQL mutations               |
| Penalty System            | Clamped at 0.25               | Prevents reward collapse during training            |
| Score Tracking            | Best-score per episode        | Efficiency penalty doesn't penalize final score     |
| NaN Handling              | Sanitized before serialization| Prevents JSON 500 errors from pandas NaN values     |

---

## License

MIT
