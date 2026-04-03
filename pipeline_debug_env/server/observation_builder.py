from typing import Dict, Any, List
import pandas as pd
import math

class ObservationBuilder:
    def __init__(self, executor):
        self.executor = executor
        self._cached_rows = None
        self._last_state_hash = None

    @staticmethod
    def _sanitize_for_json(obj):
        """Replace NaN/Inf with None so JSON serialization doesn't crash."""
        if isinstance(obj, list):
            return [ObservationBuilder._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: ObservationBuilder._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    def build_schema_diff(self, actual_schema: Dict[str, Any], expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Only return mismatched or new columns to reduce token noise.
        """
        diff = {'actual_mismatches': {}, 'missing_expected': {}}
        
        for e_col, e_prop in expected_schema.items():
            if e_col not in actual_schema:
                diff['missing_expected'][e_col] = e_prop
            elif actual_schema[e_col]['dtype'] != e_prop['dtype']:
                diff['actual_mismatches'][e_col] = {
                    'expected': e_prop['dtype'],
                    'actual': actual_schema[e_col]['dtype']
                }
                
        for a_col, a_prop in actual_schema.items():
            if a_col not in expected_schema:
                diff['actual_mismatches'][a_col] = {
                    'expected': 'NOT_EXIST',
                    'actual': a_prop['dtype']
                }
                
        return diff

    def clean_error_log(self, raw_logs: List[str]) -> List[str]:
        """
        Truncates messy stack traces to just the key SQL / Engine exceptions
        """
        clean = []
        for log in raw_logs:
            # simple truncate
            lines = log.split('\n')
            if len(lines) > 3:
                clean.append(f"{lines[0]} ... {lines[-1]}")
            else:
                clean.append(log)
        return clean

    def build_observation(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs the final observation taking into account signal prioritization.
        """
        dag = env_state['pipeline_dag']
        actual_df = env_state.get('actual_output', pd.DataFrame())
        expected_df = env_state.get('expected_output', pd.DataFrame())
        logs = env_state.get('error_log', [])
        
        # 1. Error truncation
        clean_logs = self.clean_error_log(logs)

        # 2. Schema diff logic
        actual_schema = {}
        if not actual_df.empty:
            actual_schema = {c: {'dtype': str(actual_df[c].dtype)} for c in actual_df.columns}
        
        expected_schema = {}
        if not expected_df.empty:
            expected_schema = {c: {'dtype': str(expected_df[c].dtype)} for c in expected_df.columns}

        schema_diff = self.build_schema_diff(actual_schema, expected_schema)

        # 3. Smart row sampling (sanitize NaN for JSON)
        sample_rows = {
            'expected': self._sanitize_for_json(
                expected_df.head(5).to_dict(orient='records') if not expected_df.empty else []
            )
        }
        
        # Lightweight caching
        current_state_hash = hash(str(env_state.get('pipeline_dag', '')))
        
        if current_state_hash == self._last_state_hash and self._cached_rows is not None:
            sample_rows['actual'] = self._cached_rows
        else:
            if not actual_df.empty:
                # Local smart sample
                bad_mask = actual_df.isnull().any(axis=1)
                bad_rows = actual_df[bad_mask].head(5)
                rem = 8 - len(bad_rows)
                good_rows = actual_df[~bad_mask].head(rem)
                final_sample = self._sanitize_for_json(
                    pd.concat([bad_rows, good_rows]).to_dict(orient='records')
                )
                sample_rows['actual'] = final_sample
            else:
                sample_rows['actual'] = []
                
            self._cached_rows = sample_rows['actual']
            self._last_state_hash = current_state_hash

        # 4. Basic stats
        count_diff = len(actual_df) - len(expected_df)
        
        col_stats = {}
        if not actual_df.empty:
            for c in actual_df.columns:
                col_stats[c] = {
                    'nulls': int(actual_df[c].isnull().sum())
                }

        return {
            'episode_id': env_state['episode_id'],
            'task_level': env_state['task_level'],
            'pipeline_dag': dag,
            'error_log': clean_logs,
            'schema_diff': schema_diff,
            'sample_rows': sample_rows,
            'row_count_diff': count_diff,
            'column_stats': col_stats,
            'step_count': env_state['step_count'],
            'max_steps': env_state['max_steps'],
            'current_score': env_state['current_score'],
            'action_feedback': env_state['action_feedback'],
            'done': env_state['done']
        }
