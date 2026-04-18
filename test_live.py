import requests
import json

base = "https://thebosskt-pipeline-debug-env.hf.space"
print("Testing live HF Space:", base)
print("=" * 60)

# Test health
try:
    r = requests.get(f"{base}/health", timeout=15)
    print(f"GET /health [{r.status_code}]:", r.text[:200])
except Exception as e:
    print(f"GET /health FAILED: {e}")

# Test metadata  
try:
    r = requests.get(f"{base}/metadata", timeout=15)
    print(f"GET /metadata [{r.status_code}]:", r.text[:200])
except Exception as e:
    print(f"GET /metadata FAILED: {e}")

# Test schema
try:
    r = requests.get(f"{base}/schema", timeout=15)
    print(f"GET /schema [{r.status_code}]:", r.text[:200])
except Exception as e:
    print(f"GET /schema FAILED: {e}")

# Test ALL 3 tasks
for task in ["easy", "medium", "hard"]:
    print(f"\n--- Task: {task} ---")
    try:
        r = requests.post(f"{base}/reset", json={"task_level": task}, timeout=15)
        data = r.json()
        cs = data.get("current_score")
        print(f"  RESET [{r.status_code}]: current_score={cs}")
        if cs is not None and (cs <= 0.0 or cs >= 1.0):
            print(f"  *** RESET SCORE VIOLATION: {cs} ***")
    except Exception as e:
        print(f"  RESET FAILED: {e}")
        continue

    # Do 3 steps per task
    for i in range(3):
        try:
            action = {"action_type": "no_op", "target_node": "ingest", "params": {}}
            r = requests.post(f"{base}/step", json=action, timeout=15)
            data = r.json()
            rw = data.get("reward")
            done = data.get("done")
            print(f"  STEP {i+1} [{r.status_code}]: reward={rw} done={done}")
            if rw is not None and (rw <= 0.0 or rw >= 1.0):
                print(f"  *** STEP SCORE VIOLATION: {rw} ***")
            if done:
                break
        except Exception as e:
            print(f"  STEP {i+1} FAILED: {e}")

print("\nDone.")
