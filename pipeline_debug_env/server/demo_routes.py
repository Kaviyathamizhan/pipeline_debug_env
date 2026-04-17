"""
Demo UI routes — thin wrapper layer for visual demonstration.
Treats the pipeline environment as a black box.
All intelligence calls go through existing _diagnose_and_act function.

ISOLATION RULES:
- No core pipeline modification
- No global state mutation
- No monkey-patching
- Purely reads observation → returns enriched action metadata
"""

import random
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from typing import Any, Dict

from ..baseline.heuristic_agent import _diagnose_and_act

router = APIRouter(tags=["demo"])


# ---------------------------------------------------------------------------
# Confidence & reward estimation — purely cosmetic for UI display
# ---------------------------------------------------------------------------

def _estimate_confidence(action: Dict[str, Any], obs: Dict[str, Any]) -> float:
    """Estimate agent confidence for display purposes.
    Based on how specific the diagnostic signal is."""
    action_type = action.get("action_type", "no_op")
    reasoning = action.get("reasoning", "").lower()

    base = {
        "rewrite_transform": 0.87,
        "add_null_guard": 0.79,
        "add_type_cast": 0.82,
        "patch_schema": 0.84,
        "no_op": 0.32,
    }.get(action_type, 0.55)

    # Boost if reasoning has strong signal
    if any(kw in reasoning for kw in ["fix", "detected", "rename", "mismatch"]):
        base = min(0.95, base + 0.06)

    # Reduce if many errors (noisy environment)
    error_count = len(obs.get("error_log", []))
    if error_count > 2:
        base = max(0.30, base - 0.08)

    # Slight jitter for realism
    jitter = random.uniform(-0.03, 0.03)
    return round(max(0.10, min(0.98, base + jitter)), 2)


def _estimate_expected_reward(action: Dict[str, Any], obs: Dict[str, Any]) -> float:
    """Estimate expected reward improvement for display."""
    current = obs.get("current_score", 0.05)
    action_type = action.get("action_type", "no_op")

    if action_type == "no_op":
        return 0.0

    room = 0.95 - current
    multiplier = {
        "rewrite_transform": 0.65,
        "add_null_guard": 0.50,
        "add_type_cast": 0.55,
        "patch_schema": 0.60,
    }.get(action_type, 0.30)

    return round(room * multiplier, 2)


def _infer_fault_type(action: Dict[str, Any], obs: Dict[str, Any]) -> str:
    """Infer the detected fault type from agent action for display."""
    reasoning = action.get("reasoning", "").lower()
    action_type = action.get("action_type", "")

    if "schema" in reasoning or "rename" in reasoning or "drift" in reasoning:
        return "Schema Drift"
    if "null" in reasoning or action_type == "add_null_guard":
        return "Null Propagation"
    if "type" in reasoning or "cast" in reasoning or "mismatch" in reasoning:
        return "Type Mismatch"
    if "boolean" in reasoning or "inversion" in reasoning:
        return "Boolean Inversion"
    if "filter" in reasoning:
        return "Filter Removal"

    # Fallback: analyze observation signals
    schema_diff = obs.get("schema_diff", {})
    col_stats = obs.get("column_stats", {})

    if schema_diff.get("missing_expected"):
        return "Schema Drift"
    if any(v.get("nulls", 0) > 0 for v in col_stats.values()):
        return "Null Propagation"
    if schema_diff.get("actual_mismatches"):
        return "Type Mismatch"

    return "Unknown Fault"


def _detect_initial_faults(obs: Dict[str, Any]) -> list:
    """Scan the initial observation to list all visible fault signals."""
    faults = []
    schema_diff = obs.get("schema_diff", {})
    col_stats = obs.get("column_stats", {})
    error_log = obs.get("error_log", [])
    dag = obs.get("pipeline_dag", {})
    nodes = dag.get("nodes", [])

    if schema_diff.get("missing_expected"):
        faults.append({
            "type": "Schema Drift",
            "detail": f"Missing columns: {', '.join(schema_diff['missing_expected'].keys())}",
            "severity": "high"
        })
    if schema_diff.get("actual_mismatches"):
        for col, info in schema_diff["actual_mismatches"].items():
            if isinstance(info, dict):
                faults.append({
                    "type": "Type Mismatch",
                    "detail": f"{col}: expected {info.get('expected')} got {info.get('actual')}",
                    "severity": "high"
                })
    if any(v.get("nulls", 0) > 0 for v in col_stats.values()):
        null_cols = [c for c, v in col_stats.items() if v.get("nulls", 0) > 0]
        faults.append({
            "type": "Null Propagation",
            "detail": f"Null values in: {', '.join(null_cols)}",
            "severity": "medium"
        })
    if error_log:
        faults.append({
            "type": "Execution Error",
            "detail": error_log[0][:120] if error_log else "",
            "severity": "high"
        })

    # Check for SQL-level signals in DAG nodes
    for node in nodes:
        sql = node.get("sql", "")
        if "TRY_CAST" in sql and "VARCHAR" in sql:
            faults.append({
                "type": "Type Mismatch",
                "detail": f"Node '{node['name']}' casts numeric to VARCHAR",
                "severity": "high"
            })
            break

    if not faults:
        faults.append({
            "type": "Unknown",
            "detail": "Pipeline output does not match expected schema",
            "severity": "medium"
        })

    return faults


# ---------------------------------------------------------------------------
# Agent action endpoint — calls existing heuristic agent (black-box)
# ---------------------------------------------------------------------------

@router.post("/agent_action")
async def agent_action_endpoint(body: Dict[str, Any]):
    """
    Wrapper endpoint that calls the heuristic agent's _diagnose_and_act
    function and enriches the response with confidence and expected reward
    estimates for UI display purposes.

    Does NOT modify any core pipeline or agent logic.
    """
    observation = body.get("observation", {})
    history = body.get("history", [])

    # Call existing agent logic — black-box invocation
    action = await run_in_threadpool(_diagnose_and_act, observation, history)

    # Enrich with UI-friendly metadata (cosmetic only)
    confidence = _estimate_confidence(action, observation)
    expected_reward = _estimate_expected_reward(action, observation)
    fault_type = _infer_fault_type(action, observation)

    return {
        "action": {
            "action_type": action.get("action_type", "no_op"),
            "target_node": action.get("target_node", ""),
            "params": action.get("params", {}),
        },
        "reasoning": action.get("reasoning", "No diagnostic signal detected."),
        "confidence": confidence,
        "expected_reward": expected_reward,
        "detected_fault": fault_type,
    }


@router.post("/analyze")
async def analyze_endpoint(body: Dict[str, Any]):
    """
    Analyze the current observation and return detected faults.
    Used by the UI for the mission briefing screen.
    """
    observation = body.get("observation", {})
    faults = _detect_initial_faults(observation)

    return {
        "faults": faults,
        "fault_count": len(faults),
        "task_level": observation.get("task_level", "unknown"),
        "max_steps": observation.get("max_steps", 4),
    }
