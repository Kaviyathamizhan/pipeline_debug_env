import pandas as pd

class Grader:
    def __init__(self):
        pass

    def compute_reward(self, actual_output: pd.DataFrame, expected_output: pd.DataFrame, 
                       step_count: int, max_steps: int, prev_score: float,
                       is_invalid_action: bool = False, repeat_count: int = 0) -> float:
        # Component 1: Schema correctness
        expected_cols = set(expected_output.columns)
        actual_cols = set(actual_output.columns)
        
        if len(expected_cols | actual_cols) == 0:
            schema_score = 0.0
        else:
            schema_score = len(expected_cols & actual_cols) / len(expected_cols | actual_cols)

        # Component 2: Row-level accuracy
        if len(actual_output) == 0 or len(expected_output) == 0:
            row_score = 0.0
        else:
            matched_cells = self.compare_tables(actual_output, expected_output)
            total_expected_cells = len(expected_output) * len(expected_output.columns)
            row_score = matched_cells / max(total_expected_cells, 1)

        # Component 3: Step efficiency
        efficiency = max(0.0, 1.0 - (step_count / max_steps))

        # Weighted sum: ~0.9 max before penalties
        raw_score = 0.35 * schema_score + 0.35 * row_score + 0.20 * efficiency

        # Component 4: Regression penalty (if agent broke something)
        regression = max(0.0, prev_score - raw_score) * 0.1
        
        # Penalties logic
        penalties = regression
        
        if is_invalid_action:
            penalties += 0.05
            
        if repeat_count == 1:
            penalties += 0.05
        elif repeat_count == 2:
            penalties += 0.10
        elif repeat_count >= 3:
            penalties += 0.15

        MAX_PENALTY = 0.25
        total_penalty = min(MAX_PENALTY, penalties)

        final_score = max(0.0, min(1.0, raw_score - total_penalty))

        # FIX 1: Quick-fix cap
        QUICK_FIX_MAX_SCORE = 0.6
        full_match = (schema_score > 0.99 and row_score > 0.99)
        if not full_match:
            final_score = min(final_score, QUICK_FIX_MAX_SCORE)

        # FIX 2: Add efficiency penalty
        step_penalty = 0.02 * step_count
        final_score = max(0.0, final_score - step_penalty)

        import math
        if math.isnan(final_score) or math.isinf(final_score):
            final_score = 0.0
        return round(final_score, 4)

    def compare_tables(self, actual: pd.DataFrame, expected: pd.DataFrame) -> float:
        """
        Ordered cell-by-cell float match with precision check
        """
        common_cols = sorted(set(actual.columns) & set(expected.columns))
        if not common_cols:
            return 0.0
            
        try:
            # Sort to make order-independent
            actual_sorted = actual[common_cols].sort_values(common_cols).reset_index(drop=True)
            expected_sorted = expected[common_cols].sort_values(common_cols).reset_index(drop=True)
        except Exception:
            # Fallback if un-sortable (e.g. mix types)
            actual_sorted = actual[common_cols].reset_index(drop=True)
            expected_sorted = expected[common_cols].reset_index(drop=True)

        min_rows = min(len(actual_sorted), len(expected_sorted))
        actual_trunc = actual_sorted.iloc[:min_rows]
        expected_trunc = expected_sorted.iloc[:min_rows]

        total_cells = min_rows * len(common_cols)
        matched_cells = 0

        for col in common_cols:
            for a_val, e_val in zip(actual_trunc[col], expected_trunc[col]):
                if pd.isna(a_val) and pd.isna(e_val):
                    matched_cells += 1
                elif isinstance(e_val, float):
                    try:
                        matched_cells += int(abs(float(a_val) - e_val) < 1e-4)
                    except:
                        pass
                else:
                    matched_cells += int(a_val == e_val)

        return matched_cells
