import asyncio
import statistics
import time
import os
from .heuristic_agent import run_episode

# Fixed seed per instructions
os.environ["PIPELINE_SEED"] = "42"
os.environ["ENV_URL"] = "http://localhost:7860"

NUM_EPISODES = 10  # More episodes since no API latency


async def run_task_eval(task_name: str, num_episodes: int) -> dict:
    """Run episodes SEQUENTIALLY — server is single-session."""
    scores = []
    steps_list = []

    for ep_idx in range(num_episodes):
        print(f"  Episode {ep_idx + 1}/{num_episodes}...", end=" ", flush=True)
        try:
            res = await run_episode(task_level=task_name)
            score = res.get("final_score", 0.0)
            steps = res.get("steps_used", 0)
            scores.append(score)
            steps_list.append(steps)
            print(f"score={score:.4f}  steps={steps}")
        except Exception as e:
            print(f"FAILED: {repr(e)}")
            scores.append(0.0)
            steps_list.append(0)

        # Small cooldown between episodes
        await asyncio.sleep(0.5)

    avg_score = statistics.mean(scores) if scores else 0.0
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    avg_steps = statistics.mean(steps_list) if steps_list else 0.0

    failure_rate = len([s for s in scores if s < 0.70]) / len(scores) if scores else 1.0

    return {
        "task": task_name,
        "avg_score": round(avg_score, 4),
        "std_dev": round(std_dev, 4),
        "avg_steps": round(avg_steps, 2),
        "failure_rate": round(failure_rate, 4),
        "raw_scores": scores
    }


async def main():
    print("=" * 60)
    print("  Pipeline Debug Env — Baseline Evaluation")
    print("  Agent: Heuristic (rule-based, no API)")
    print("=" * 60)
    print(f"Episodes per task: {NUM_EPISODES}")
    print(f"Mode: SEQUENTIAL (single-session server)")
    print()

    start_time = time.time()

    all_results = []
    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Evaluating Task: {task.upper()} ---")
        task_res = await run_task_eval(task, NUM_EPISODES)
        all_results.append(task_res)

    end_time = time.time()

    # Build report
    report_path = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation_report.md")
    report = f"# Pipeline Debug Env — Evaluation Report\n\n"
    report += f"**Agent:** Heuristic (rule-based, no external API)\n"
    report += f"**Episodes per task:** {NUM_EPISODES}\n"
    report += f"**Total runtime:** {round(end_time - start_time, 2)} seconds\n"
    report += f"**Seed:** 42\n\n"

    report += "| Task | Avg Score | Std Dev | Avg Steps | Failure Rate |\n"
    report += "|------|-----------|---------|-----------|-------------|\n"
    for r in all_results:
        report += f"| {r['task']} | {r['avg_score']} | {r['std_dev']} | {r['avg_steps']} | {r['failure_rate'] * 100:.1f}% |\n"

    print("\n" + report)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
