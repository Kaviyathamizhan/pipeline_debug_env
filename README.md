---
title: Pipeline Debug Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🚀 Pipeline Debug Env — RL Environment for Debugging Data Pipelines

## 🧠 Overview
This is an OpenEnv-compatible reinforcement learning environment that simulates broken data pipelines. It is specially designed to train and benchmark LLM and RL agents to diagnose and repair SQL workflows through causal reasoning and multi-step execution paths.

## ⚙️ Features
- **Fault Injection:** Generates complex interdependent failures (schema drift, type mismatch, null propagation).
- **Observation Filtering:** Provides robust semantic state components including error logs, schema diffs, and row samples.
- **Action Validation System:** Safe abstraction mapped seamlessly onto underlying mutable DAG configurations.
- **Reward-Based Grader:** Deterministic reward computation isolating correctness, efficiency, and regression.
- **Execution Engine:** Lightweight orchestration driven natively via FastAPI + DuckDB.
- **Fully Dockerized Deployment:** Easily portable container image compatible with all managed compute platforms.

## 🌐 Live Demo
https://thebosskt-pipeline-debug-env.hf.space

## 🛠 How It Works
1. `POST /reset` initializes the environment with a perturbed, broken data pipeline model.
2. `POST /step` applies targeted semantic fixes proposed by the evaluating agent.
3. The agent loops and sequentially resolves upstream and downstream nodes, directly maximizing health.
4. A deterministic grader verifies the accuracy of the executed tables against ground truth expectations.

## 📊 Evaluation
**System Accuracy Scaling Matrix:**
- **Easy**   → `0.76`
- **Medium** → `0.58`
- **Hard**   → `0.37`

*Difficulty Scaling:* The performance decay strictly originates from the logical complexity and topological depth of interacting SQL faults. The reward mechanisms accurately isolate the agent's problem-solving capability across varying tiers of degradation.

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
uvicorn pipeline_debug_env.server.app:app --port 7860
```

## 🤖 Run Baseline Agent

```bash
python inference.py
```

## 🐳 Docker

```bash
docker build -t pipeline-env .
docker run -p 7860:7860 pipeline-env
```

## 🧩 API Endpoints
- `POST /reset`
- `POST /step`
- `GET /state`
