import pandas as pd
from pipeline_debug_env.server.grader import Grader

def test_grader_zero_rows():
    grader = Grader()
    actual = pd.DataFrame()
    expected = pd.DataFrame({'a': [1, 2]})
    
    score = grader.compute_reward(actual, expected, 1, 10, 0.0)
    # schema=0, row=0, efficiency=0.9 => raw=0.18, step_penalty=0.02, quick_fix_cap=0.60
    # final = min(0.18-0.02, 0.60) = 0.16
    assert score < 0.20, f"Expected < 0.20 for zero rows, got {score}"

def test_grader_schema_mismatch():
    grader = Grader()
    actual = pd.DataFrame({'b': [1, 2]})
    expected = pd.DataFrame({'a': [1, 2]})
    
    score = grader.compute_reward(actual, expected, 1, 10, 0.0)
    # schema score = 0, row score = 0
    assert score < 0.3, f"Expected low score for schema mismatch, got {score}"

def test_grader_penalty_clamp():
    grader = Grader()
    actual = pd.DataFrame({'a': [1, 2]})
    expected = pd.DataFrame({'a': [1, 2]})
    
    score_no_penalty = grader.compute_reward(actual, expected, 1, 10, 0.0)
    score_heavy_penalty = grader.compute_reward(
        actual, expected, 1, 10, 10.0,
        is_invalid_action=True, repeat_count=5
    )
    
    diff = score_no_penalty - score_heavy_penalty
    # With step_penalty in both calls (both at step 1), the clamp still caps the diff
    assert diff <= 0.26, f"Penalty clamp failed! Expected drop <= 0.26, got {diff}"
    assert diff > 0.0, f"Heavy penalty should reduce score, got diff={diff}"

def test_grader_regression():
    grader = Grader()
    actual = pd.DataFrame({'a': [1]})
    expected = pd.DataFrame({'a': [1, 2]})
    
    # Sub-optimal state
    score = grader.compute_reward(actual, expected, 5, 10, 0.9)
    # Prev score was 0.9, raw score naturally is around 0.5. 
    # Regression penalty = 0.1 * (0.9 - 0.5) = 0.04
    assert score < 0.6
