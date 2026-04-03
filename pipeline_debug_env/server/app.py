import os
import math
import traceback
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from ..core.environment import PipelineEnvironment


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
            return 0.0
        return obj
    if isinstance(obj, str):
        return obj
    # numpy scalar fallback
    try:
        return ensure_serializable(obj.item())
    except (AttributeError, ValueError):
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# Global environment instance (single-session mode for Docker)
# ---------------------------------------------------------------------------
env: Optional[PipelineEnvironment] = None


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
    return {"status": "ok"}


@app.post("/reset")
async def reset_endpoint(body: Dict[str, Any] = {}):
    """Reset the environment. Always returns schema-safe JSON."""
    global env
    task_level = body.get("task_level", None)
    try:
        observation = await run_in_threadpool(env.reset, task_level)
        return ensure_serializable(observation)
    except Exception as e:
        traceback.print_exc()
        # Schema-safe fallback: return a minimal valid observation
        return ensure_serializable({
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
            "current_score": 0.0,
            "action_feedback": "",
            "done": True,
        })


@app.post("/step")
async def step_endpoint(action: Dict[str, Any]):
    """Apply an action. Always returns schema-safe JSON."""
    global env
    if env is None or env.episode_manager is None:
        return ensure_serializable({
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": "No active episode. Call /reset first."},
        })
    try:
        result = await run_in_threadpool(env.step, action)
        return ensure_serializable(result)
    except Exception as e:
        traceback.print_exc()
        return ensure_serializable({
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)},
        })


@app.get("/state")
async def state_endpoint():
    """Return current episode state. Always returns schema-safe JSON."""
    global env
    if env is None or env.episode_manager is None:
        return ensure_serializable({
            "episode_id": "",
            "step_count": 0,
            "max_steps": 0,
            "current_score": 0.0,
            "best_score": 0.0,
            "task_level": "",
            "done": True,
        })
    try:
        state_data = await run_in_threadpool(env.state)
        return ensure_serializable(state_data)
    except Exception as e:
        traceback.print_exc()
        return ensure_serializable({
            "episode_id": "",
            "step_count": 0,
            "max_steps": 0,
            "current_score": 0.0,
            "best_score": 0.0,
            "task_level": "",
            "done": True,
        })
