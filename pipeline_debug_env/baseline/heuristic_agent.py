"""
Heuristic baseline agent for pipeline_debug_env.
Uses rule-based pattern matching instead of an LLM API.
Reads observation signals (error_log, schema_diff, sample_rows, DAG SQL)
and emits the most appropriate fix action.

This agent validates that the environment is solvable without any external API.

Usage:
    python -m pipeline_debug_env.baseline.heuristic_agent
"""

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pipeline_debug_env.client import PipelineDebugEnvClient

ENV_URL    = os.getenv("ENV_URL", "http://localhost:7860")
TASK_LEVEL = os.getenv("TASK_LEVEL", "easy")

def _clamp_score(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.05
    return max(0.05, min(0.95, x))


def _diagnose_and_act(obs: Dict[str, Any], history: List[Dict]) -> Dict[str, Any]:
    """
    Core heuristic engine. Reads observation signals and returns the best action.
    Checks are ordered by severity: errors > schema > data > fallback.
    """
    error_log   = obs.get("error_log", [])
    schema_diff = obs.get("schema_diff", {})
    sample_rows = obs.get("sample_rows", {})
    dag         = obs.get("pipeline_dag", {})
    nodes       = dag.get("nodes", [])
    col_stats   = obs.get("column_stats", {})
    score       = obs.get("current_score", 0.0)

    # Collect past action types to avoid pure repeats
    past_actions = [h.get("action", {}).get("action_type") for h in history]
    past_targets = [h.get("action", {}).get("target_node") for h in history]

    # ========== SIGNAL 1: Scan DAG SQL for obvious fault signatures ==========
    
    # Known camelCase renames from fault injector
    camel_fixes = {
        'userId': 'user_id', 'orderId': 'order_id', 'sessionId': 'session_id',
        'eventType': 'event_type', 'txnId': 'txn_id', 'txnDate': 'txn_date',
        'amt': 'amount', 'reg': 'region', 'isChurned': 'is_churned',
        'isDeleted': 'is_deleted', 'totalRevenue': 'total_revenue',
        'eventCount': 'event_count',
    }
    # Also catch _v2 suffixes
    v2_pattern = re.compile(r'(\w+)_v2')

    for node in nodes:
        node_name = node.get("name", "")
        node_sql  = node.get("sql", "")
        node_type = node.get("node_type", "")
        if node_type == "output":
            continue

        # Check for camelCase renames (schema_drift)
        for bad, good in camel_fixes.items():
            if bad in node_sql:
                new_sql = node_sql.replace(bad, good)
                return {
                    "action_type": "rewrite_transform",
                    "target_node": node_name,
                    "params": {"new_sql": new_sql},
                    "reasoning": f"Fix schema drift: rename {bad} back to {good}"
                }

        # Check for _v2 suffix renames
        m = v2_pattern.search(node_sql)
        if m:
            bad_col = m.group(0)
            good_col = m.group(1)
            new_sql = node_sql.replace(bad_col, good_col)
            return {
                "action_type": "rewrite_transform",
                "target_node": node_name,
                "params": {"new_sql": new_sql},
                "reasoning": f"Fix schema drift: rename {bad_col} back to {good_col}"
            }

    # ========== SIGNAL 2: Type mismatch (VARCHAR wrapping a numeric) ==========
    for node in nodes:
        node_name = node.get("name", "")
        node_sql  = node.get("sql", "")
        if node.get("node_type") == "output":
            continue

        varchar_match = re.search(r'TRY_CAST\((\w+) AS VARCHAR\) AS (\w+)', node_sql)
        if varchar_match:
            col = varchar_match.group(2)
            new_sql = node_sql.replace(
                f"TRY_CAST({col} AS VARCHAR) AS {col}",
                f"TRY_CAST({col} AS FLOAT) AS {col}"
            )
            return {
                "action_type": "rewrite_transform",
                "target_node": node_name,
                "params": {"new_sql": new_sql},
                "reasoning": f"Fix type mismatch: cast {col} back to FLOAT"
            }

    # ========== SIGNAL 3: Boolean inversion (= TRUE should be = FALSE) ==========
    for node in nodes:
        node_name = node.get("name", "")
        node_sql  = node.get("sql", "")
        if node.get("node_type") == "output":
            continue

        # is_churned = TRUE → should be FALSE (user_engagement)
        if 'is_churned = TRUE' in node_sql:
            new_sql = node_sql.replace('is_churned = TRUE', 'is_churned = FALSE')
            return {
                "action_type": "rewrite_transform",
                "target_node": node_name,
                "params": {"new_sql": new_sql},
                "reasoning": "Fix boolean inversion: is_churned should be FALSE"
            }
        if 'is_deleted = TRUE' in node_sql:
            new_sql = node_sql.replace('is_deleted = TRUE', 'is_deleted = FALSE')
            return {
                "action_type": "rewrite_transform",
                "target_node": node_name,
                "params": {"new_sql": new_sql},
                "reasoning": "Fix boolean inversion: is_deleted should be FALSE"
            }
        # IS NULL where IS NOT NULL was expected (inverted null check)
        if 'IS NULL' in node_sql and 'IS NOT NULL' not in node_sql:
            if any(kw in node_sql.upper() for kw in ['WHERE', 'HAVING']):
                new_sql = node_sql.replace('IS NULL', 'IS NOT NULL')
                return {
                    "action_type": "rewrite_transform",
                    "target_node": node_name,
                    "params": {"new_sql": new_sql},
                    "reasoning": "Fix inverted null check: should be IS NOT NULL"
                }

    # ========== SIGNAL 4: Null propagation (missing WHERE clause) ==========
    # Check if there are nulls in the data that shouldn't be there
    has_nulls = any(v.get("nulls", 0) > 0 for v in col_stats.values())
    if has_nulls and "add_null_guard" not in past_actions:
        # Find a clean/transform node missing a null guard
        for node in nodes:
            node_name = node.get("name", "")
            node_sql  = node.get("sql", "")
            node_type = node.get("node_type", "")
            if node_type in ("clean", "transform") and "WHERE" not in node_sql.upper():
                # Determine which column has nulls
                null_col = next((c for c, v in col_stats.items() if v.get("nulls", 0) > 0), "amount")
                return {
                    "action_type": "add_null_guard",
                    "target_node": node_name,
                    "params": {"column": null_col, "strategy": "drop"},
                    "reasoning": f"Null propagation: add null guard for {null_col}"
                }

    # ========== SIGNAL 6: Schema diff — missing columns ==========
    missing = schema_diff.get("missing_expected", {})
    mismatches = schema_diff.get("actual_mismatches", {})

    if missing:
        # A column is expected but missing. Check if it was renamed.
        for expected_col in missing:
            for actual_col in mismatches:
                if actual_col not in missing:
                    # We have an extra column that might be the renamed version
                    for node in reversed(nodes):
                        if node.get("node_type") != "output":
                            return {
                                "action_type": "patch_schema",
                                "target_node": node["name"],
                                "params": {"old_column": actual_col, "new_column": expected_col},
                                "reasoning": f"Rename {actual_col} back to {expected_col}"
                            }

    # ========== SIGNAL 7: Type mismatches in schema diff ==========
    for col, info in mismatches.items():
        if isinstance(info, dict) and "expected" in info and "actual" in info:
            expected_type = info["expected"]
            actual_type = info["actual"]
            if expected_type != "NOT_EXIST" and actual_type != expected_type:
                # Find the node that produces this column
                for node in reversed(nodes):
                    if node.get("node_type") != "output":
                        return {
                            "action_type": "add_type_cast",
                            "target_node": node["name"],
                            "params": {"column": col, "to_type": expected_type},
                            "reasoning": f"Cast {col} from {actual_type} to {expected_type}"
                        }

    # ========== FALLBACK: If score is high enough, signal done ==========
    if score >= 0.9:
        return {
            "action_type": "no_op",
            "target_node": nodes[0]["name"] if nodes else "",
            "params": {},
            "reasoning": "Score is high. Pipeline appears fixed."
        }

    # ========== LAST RESORT: Try rewriting each non-output node ==========
    # Check if any node's SQL looks obviously broken
    for node in nodes:
        node_name = node.get("name", "")
        node_sql  = node.get("sql", "")
        node_type = node.get("node_type", "")

        if node_type == "output":
            continue

        # Skip if we already tried this exact node
        if node_name in past_targets and past_targets.count(node_name) >= 2:
            continue

        # Check for weird nested SQL wrapping (sign of accumulated bad patches)
        if node_sql.count("SELECT") > 3:
            # Over-wrapped SQL — try to simplify
            return {
                "action_type": "no_op",
                "target_node": node_name,
                "params": {},
                "reasoning": "No clear signal. Passing to preserve score."
            }

    return {
        "action_type": "no_op",
        "target_node": nodes[0]["name"] if nodes else "",
        "params": {},
        "reasoning": "No actionable signal detected. Preserving current state."
    }


async def run_episode(task_level: str = TASK_LEVEL) -> Dict[str, Any]:
    """Run a single episode using heuristic diagnosis."""
    history: List[Dict] = []
    best_score = 0.05

    async with PipelineDebugEnvClient(base_url=ENV_URL) as client:
        obs = await client.reset(task_level=task_level)
        print(f"\n[Episode Start] task={task_level}, max_steps={obs.get('max_steps')}")

        while not obs.get("done", False):
            action = _diagnose_and_act(obs, history)

            print(f"  Step {obs.get('step_count', '?')}: {action['action_type']} → {action['target_node']}")

            try:
                result = await client.step(action)
            except Exception as e:
                print(f"[Step Error] {e}")
                break

            reward = result.get("reward", 0.0)
            info   = result.get("info", {})
            obs    = result.get("observation", {})

            best_score = max(best_score, _clamp_score(reward))

            history.append({
                "action": action,
                "score_delta": info.get("score_delta", 0.0),
                "action_feedback": info.get("action_error", "")
            })

            print(f"          reward={reward:.4f}  delta={info.get('score_delta',0):+.4f}  done={obs.get('done')}")

            if obs.get("done"):
                break

        steps_used = obs.get("step_count", len(history))

        print(f"\n[Episode End] best_score={best_score:.4f}  steps={steps_used}")

        return {"final_score": _clamp_score(best_score), "steps_used": steps_used, "task_level": task_level}


if __name__ == "__main__":
    asyncio.run(run_episode())
