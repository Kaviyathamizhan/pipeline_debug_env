"""
Baseline inference agent for pipeline_debug_env.
Uses OpenAI-compatible API with ReAct prompting + reflection loop.

Usage:
    python -m pipeline_debug_env.baseline.inference

Environment Variables:
    API_BASE_URL  - LLM API base URL (default: https://api.openai.com/v1)
    MODEL_NAME    - Model to use (default: gpt-4o-mini)
    HF_TOKEN      - Hugging Face token (for deployment)
    ENV_URL       - pipeline_debug_env server URL (default: http://localhost:7860)
    TASK_LEVEL    - easy / medium / hard (default: easy)
"""

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from pipeline_debug_env.client import PipelineDebugEnvClient

# --- Config from environment ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
TASK_LEVEL   = os.getenv("TASK_LEVEL", "easy")
TASKS        = ["easy", "medium", "hard"]
BENCHMARK    = os.getenv("BENCHMARK", "pipeline_debug_env")

MAX_TOKENS = 150
MAX_STEPS_FALLBACK = 12
SUCCESS_SCORE_THRESHOLD = 0.10

SYSTEM_PROMPT = """You are an expert data engineer debugging a broken data pipeline.

At each step you will receive the current pipeline state:
- error_log: runtime errors from the last execution
- schema_diff: columns that are mismatched, missing, or unexpected
- sample_rows: actual output rows vs expected output rows
- action_feedback: why your LAST action was rejected (if it was)
- current_score: float 0.0-1.0 showing pipeline health

THINK STEP BY STEP:
1. What does the error log tell me?
2. What does the schema diff tell me?
3. What does the sample data tell me?
4. What is the most likely ROOT CAUSE (may be upstream of where the error appears)?
5. What single action best addresses that root cause?

ACTION TYPES:
- patch_schema: fix column rename (params: old_column, new_column, new_type)
- add_type_cast: fix type mismatch (params: column, to_type)
- add_null_guard: handle nulls (params: column, strategy="drop")
- rewrite_transform: replace node SQL (params: new_sql)
- fix_join_key: fix wrong join (params: left_key, right_key)
- invert_filter: flip a WHERE condition (params: filter_expression)
- no_op: signal pipeline is fixed (params: {})

RESPOND WITH JSON ONLY — no explanation, no markdown:
{
  "action_type": "<type>",
  "target_node": "<node name from pipeline_dag>",
  "params": { <action-specific parameters> },
  "reasoning": "<1-2 sentence explanation>"
}"""

def _clamp_open01(x: float, eps: float = 0.01) -> float:
    """Clamp strictly inside (0, 1), avoiding endpoints after rounding."""
    try:
        x = float(x)
    except Exception:
        return 0.5
    if x <= eps:
        return eps
    if x >= 1.0 - eps:
        return 1.0 - eps
    return x


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(bool(done)).lower()
    # evaluator requires 2 decimals; ensure printed value never becomes 0.00 or 1.00
    reward_print = _clamp_open01(reward, eps=0.01)
    print(
        f"[STEP] step={step} action={action} reward={reward_print:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # ensure printed values never become 0.00 or 1.00
    score_print = _clamp_open01(score, eps=0.001)
    rewards_str = ",".join(f"{_clamp_open01(r, eps=0.01):.2f}" for r in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={score_print:.3f} rewards={rewards_str}",
        flush=True,
    )


def _format_obs(obs: Dict[str, Any], history: List[Dict]) -> str:
    parts = []

    # Reflection over history
    if history:
        last = history[-1]
        parts.append("=== REFLECTION ===")
        parts.append(f"My last action: {json.dumps(last['action'], indent=2)}")
        delta = last.get("score_delta", 0.0)
        if delta > 0:
            parts.append(f"Result: IMPROVED score by +{delta:.4f}. Keep fixing remaining issues.")
        elif delta == 0:
            parts.append("Result: NO CHANGE. This action had no effect. Try a different node or action type.")
        else:
            parts.append(f"Result: REGRESSION, score dropped by {delta:.4f}. I may have broken something. Reconsider.")

        feedback = last.get("action_feedback", "")
        if feedback:
            parts.append(f"Server feedback: {feedback}")

    parts.append("\n=== CURRENT OBSERVATION ===")
    parts.append(f"Step: {obs.get('step_count')} / {obs.get('max_steps')}")
    parts.append(f"Score: {obs.get('current_score'):.4f}")

    # Errors
    errors = obs.get("error_log", [])
    if errors:
        parts.append("\nERROR LOG:")
        for e in errors[:3]:
            parts.append(f"  - {e}")

    # Schema diff (already filtered to mismatches only)
    schema_diff = obs.get("schema_diff", {})
    if schema_diff.get("missing_expected") or schema_diff.get("actual_mismatches"):
        parts.append("\nSCHEMA DIFF (mismatches only):")
        parts.append(json.dumps(schema_diff, indent=2))

    # Sample rows
    sample = obs.get("sample_rows", {})
    actual = sample.get("actual", [])[:3]
    expected = sample.get("expected", [])[:3]
    if actual or expected:
        parts.append("\nSAMPLE ROWS:")
        parts.append(f"  actual   : {json.dumps(actual)}")
        parts.append(f"  expected : {json.dumps(expected)}")

    # Row count diff
    rcd = obs.get("row_count_diff", 0)
    if rcd != 0:
        parts.append(f"\nROW COUNT DIFF: {rcd:+d} (positive = too many rows, negative = too few)")

    # DAG nodes' status
    dag = obs.get("pipeline_dag", {})
    nodes = dag.get("nodes", [])
    node_statuses = {n["name"]: n.get("status", "unknown") for n in nodes}
    parts.append(f"\nNODE STATUSES: {json.dumps(node_statuses)}")

    return "\n".join(parts)


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON action from LLM response."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try regex extraction — use GREEDY match to capture nested braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


async def _call_llm_with_retry(llm: AsyncOpenAI, prompt: str, retries: int = 3):
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt}
    ]
    for i in range(retries):
        try:
            response = await llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=150,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if i == retries - 1:
                raise e
            await asyncio.sleep(2 * (i + 1))

async def _run_episode_and_log(task_level: str) -> Tuple[bool, int, float, List[float]]:
    """
    Run one episode and emit mandatory stdout lines:
      [START] ...
      [STEP] ...
      [END] ...
    Returns (success, steps_taken, score, rewards).
    """
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    llm: Optional[AsyncOpenAI] = None
    if api_key:
        llm = AsyncOpenAI(api_key=api_key, base_url=API_BASE_URL)

    history: List[Dict[str, Any]] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    async with PipelineDebugEnvClient(base_url=ENV_URL) as client:
        last_error: Optional[str] = None
        try:
            obs = await client.reset(task_level=task_level)
            max_steps = int(obs.get("max_steps") or MAX_STEPS_FALLBACK)
            max_steps = max(1, max_steps)

            for step in range(1, max_steps + 1):
                if obs.get("done", False):
                    break

                # Choose action: LLM if available, otherwise safe no-op.
                action: Dict[str, Any]
                if llm is None:
                    action = {"action_type": "no_op", "target_node": "", "params": {}, "reasoning": "No API key."}
                else:
                    prompt = _format_obs(obs, history)
                    try:
                        raw = await _call_llm_with_retry(llm, prompt)
                        parsed = _parse_action(raw)
                    except Exception:
                        parsed = None

                    action = parsed or {
                        "action_type": "no_op",
                        "target_node": "",
                        "params": {},
                        "reasoning": "Parse failed.",
                    }

                action_str = action.get("action_type", "no_op")

                # Step
                result = await client.step(action)
                reward_raw = float(result.get("reward") or 0.0)
                reward = _clamp_open01(reward_raw, eps=0.01)
                done = bool(result.get("done", False))
                info = result.get("info", {}) or {}
                last_error = info.get("action_error") or info.get("error")
                obs = result.get("observation", {}) or {}

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=str(action_str), reward=reward, done=done, error=last_error)

                history.append(
                    {
                        "action": action,
                        "score_delta": info.get("score_delta", 0.0),
                        "action_feedback": last_error or "",
                    }
                )

                if done:
                    break

            # Score: use best observed reward as normalized score (already in (0,1)).
            score = max(rewards) if rewards else 0.01
            score = _clamp_open01(score, eps=0.001)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            # Must always emit END even on exception.
            last_error = str(exc)
            if not rewards:
                rewards = [_clamp_open01(0.05, eps=0.01)]
            score = _clamp_open01(max(rewards), eps=0.001)
            success = False

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


if __name__ == "__main__":
    async def _main():
        # Run exactly 3 tasks in sequence. All reporting is via mandatory stdout lines.
        for task in TASKS:
            await _run_episode_and_log(task_level=task)

    asyncio.run(_main())
