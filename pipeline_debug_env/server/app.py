import os
import math
import traceback
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from ..core.environment import PipelineEnvironment


# ---------------------------------------------------------------------------
# Score clamping — the single source of truth for (0, 1) enforcement
# ---------------------------------------------------------------------------
def clamp_score(value: float) -> float:
    """Clamp a score to strictly within (0, 1). Never returns 0.0 or 1.0."""
    if math.isnan(value) or math.isinf(value):
        return 0.05
    return max(0.05, min(0.95, float(value)))


def ensure_serializable(obj: Any) -> Any:
    """Recursively enforce JSON serializability. Handles NaN, Inf, None,
    numpy scalars, sets, and any other non-standard types."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_serializable(x) for x in obj]
    if isinstance(obj, set):
        return [ensure_serializable(x) for x in sorted(obj, key=str)]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.05  # FIXED: was 0.0 which violated strict (0,1) range
        return obj
    if isinstance(obj, str):
        return obj
    # numpy scalar fallback
    try:
        return ensure_serializable(obj.item())
    except (AttributeError, ValueError):
        pass
    return str(obj)


def clamp_response_scores(obj: Any) -> Any:
    """Walk through a response dict and clamp any 'reward', 'current_score',
    'best_score' fields to strict (0, 1)."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Clamp any score-like numeric fields so nothing can leak 0.0 or 1.0.
            if k in ("reward", "current_score", "best_score", "final_score", "avg_score") and isinstance(v, (int, float)):
                result[k] = clamp_score(float(v))
            else:
                result[k] = clamp_response_scores(v)
        return result
    if isinstance(obj, (list, tuple)):
        return [clamp_response_scores(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Global environment instance (single-session mode for Docker)
# ---------------------------------------------------------------------------
env: Optional[PipelineEnvironment] = None

TASKS = ["easy", "medium", "hard"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    task_level = os.getenv("TASK_LEVEL", "easy")
    seed = int(os.getenv("PIPELINE_SEED", "42"))
    env = PipelineEnvironment(task_level=task_level, seed=seed)
    yield
    # Cleanup
    if env and env.executor:
        env.executor.reset()


app = FastAPI(
    title="Pipeline Debug Environment",
    description="OpenEnv-compatible RL environment for debugging data pipelines",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Pipeline Debug Env is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    return {
        "name": "pipeline_debug_env",
        "description": "OpenEnv-compatible RL environment for diagnosing and repairing broken data pipelines",
        "version": "1.0.0",
        "tasks": TASKS,
    }


@app.get("/schema")
async def schema():
    return {
        "action": {
            "type": "object",
            "required": ["action_type", "target_node", "params"],
            "properties": {
                "action_type": {"type": "string"},
                "target_node": {"type": "string"},
                "params": {"type": "object"},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "task_level": {"type": "string"},
                "pipeline_dag": {"type": "object"},
                "error_log": {"type": "array"},
                "current_score": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "current_score": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/reset")
async def reset_endpoint(body: Dict[str, Any] = {}):
    """Reset the environment. Always returns schema-safe JSON."""
    global env
    task_level = body.get("task_level", None)
    try:
        observation = await run_in_threadpool(env.reset, task_level)
        safe = ensure_serializable(observation)
        return clamp_response_scores(safe)
    except Exception as e:
        traceback.print_exc()
        return {
            "episode_id": "",
            "task_level": task_level or "easy",
            "pipeline_dag": {"nodes": [], "edges": []},
            "error_log": [str(e)],
            "schema_diff": {},
            "sample_rows": {"actual": [], "expected": []},
            "row_count_diff": 0,
            "column_stats": {},
            "step_count": 0,
            "max_steps": 4,
            "current_score": 0.05,
            "action_feedback": "",
            "done": True,
        }


@app.post("/step")
async def step_endpoint(action: Dict[str, Any]):
    """Apply an action. Always returns schema-safe JSON."""
    global env
    if env is None or env.episode_manager is None:
        return {
            "observation": {},
            "reward": 0.05,
            "done": True,
            "info": {"error": "No active episode. Call /reset first."},
        }
    try:
        result = await run_in_threadpool(env.step, action)
        safe = ensure_serializable(result)
        return clamp_response_scores(safe)
    except Exception as e:
        traceback.print_exc()
        return {
            "observation": {},
            "reward": 0.05,
            "done": True,
            "info": {"error": str(e)},
        }


@app.get("/state")
async def state_endpoint():
    """Return current episode state. Always returns schema-safe JSON."""
    global env
    if env is None or env.episode_manager is None:
        return {
            "episode_id": "",
            "step_count": 0,
            "max_steps": 0,
            "current_score": 0.05,
            "best_score": 0.05,
            "task_level": "",
            "done": True,
        }
    try:
        state_data = await run_in_threadpool(env.state)
        safe = ensure_serializable(state_data)
        return clamp_response_scores(safe)
    except Exception as e:
        traceback.print_exc()
        return {
            "episode_id": "",
            "step_count": 0,
            "max_steps": 0,
            "current_score": 0.05,
            "best_score": 0.05,
            "task_level": "",
            "done": True,
        }
