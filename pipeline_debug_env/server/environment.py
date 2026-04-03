import uuid
import random
import copy
import yaml
import os
import pandas as pd
from typing import Any, Dict, Optional

from .pipeline_executor import PipelineExecutor
from .fault_injector import FaultInjector
from .grader import Grader
from .observation_builder import ObservationBuilder
from .action_parser import ActionParser
from .episode_manager import EpisodeManager

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')

TASK_CONFIG = {
    'easy':   {'max_steps': 4,  'templates': ['ecommerce_orders']},
    'medium': {'max_steps': 8,  'templates': ['user_engagement', 'financial_revenue']},
    'hard':   {'max_steps': 12, 'templates': ['user_engagement', 'financial_revenue', 'ecommerce_orders']},
}


class PipelineEnvironment:
    def __init__(self, task_level: str = 'easy', seed: int = 42):
        self.task_level = task_level
        self.seed = seed
        self._episode_counter = 0  # varies seed per episode
        self.grader = Grader()
        self.fault_injector = FaultInjector()
        self.executor: Optional[PipelineExecutor] = None
        self.obs_builder: Optional[ObservationBuilder] = None
        self.action_parser: Optional[ActionParser] = None
        self.episode_manager: Optional[EpisodeManager] = None
        self.current_dag: Optional[Dict] = None
        self.expected_output: Optional[pd.DataFrame] = None
        self.mock_data: Dict[str, pd.DataFrame] = {}  # persists across step resets

    def _load_template(self, name: str) -> Dict:
        path = os.path.join(TEMPLATES_DIR, f'{name}.yaml')
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _build_dag_dict(self, nodes: list, edges: list = None) -> Dict:
        return {
            'nodes': nodes,
            'edges': edges or []
        }

    def reset(self, task_level: Optional[str] = None) -> Dict[str, Any]:
        if task_level:
            self.task_level = task_level

        config = TASK_CONFIG.get(self.task_level, TASK_CONFIG['easy'])
        # Use episode-varied seed for template selection
        random.seed(self.seed + self._episode_counter)
        template_name = random.choice(config['templates'])
        template = self._load_template(template_name)

        # Fresh executor per episode - prevents state leakage
        self.executor = PipelineExecutor()
        self.obs_builder = ObservationBuilder(self.executor)
        self.action_parser = ActionParser(self.executor)

        episode_id = f"ep_{uuid.uuid4().hex[:6]}"
        max_steps = config['max_steps']
        self.episode_manager = EpisodeManager(episode_id, self.task_level, max_steps)

        mock_data = template.get('mock_data', {})
        self.mock_data = {}  # store for re-use in step()
        for table_name, rows in mock_data.items():
            df = pd.DataFrame(rows)
            self.mock_data[table_name] = df
            self.executor.load_data(table_name, df)

        # Run clean pipeline to get GROUND TRUTH
        clean_dag = self._build_dag_dict(template['nodes'])
        clean_output, _, _ = self.executor.execute_dag(clean_dag, {})

        # CRITICAL: deepcopy to ensure ground truth is never mutated
        self.expected_output = copy.deepcopy(clean_output)

        # Inject faults with per-episode seed variation; retry if fault breaks SQL
        for attempt in range(5):
            episode_seed = self.seed + self._episode_counter + attempt
            faulty_dag = self.fault_injector.apply_faults(
                copy.deepcopy({'nodes': template['nodes'], 'edges': []}),
                self.task_level,
                seed=episode_seed
            )
            self.current_dag = faulty_dag

            # Reload mock data for this attempt
            self.executor.reset()
            for table_name, df in self.mock_data.items():
                self.executor.load_data(table_name, df)

            broken_output, error_log, node_states = self.executor.execute_dag(faulty_dag, {})
            
            # If at least some nodes executed, accept this fault config
            if node_states or not error_log:
                break

        self._episode_counter += 1

        # Build initial observation
        env_state = {
            'episode_id': episode_id,
            'task_level': self.task_level,
            'pipeline_dag': faulty_dag,
            'actual_output': broken_output,
            'expected_output': self.expected_output,
            'error_log': error_log,
            'step_count': 0,
            'max_steps': max_steps,
            'current_score': 0.0,
            'action_feedback': '',
            'done': False
        }

        return self.obs_builder.build_observation(env_state)

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.episode_manager is None or self.episode_manager.done:
            raise RuntimeError("Episode not initialized. Call reset() first.")

        prev_score = self.episode_manager.current_score

        # 1. Check for repeat penalty before parsing
        repeat_count = self.episode_manager.check_repeated_action(action_dict)

        # 2. Parse + Validate + Apply action to DAG
        success, feedback, updated_dag = self.action_parser.validate_and_apply(
            action_dict, copy.deepcopy(self.current_dag)
        )

        is_invalid = not success

        if success:
            self.current_dag = updated_dag

        # 3. Execute updated DAG in DuckDB
        self.executor.reset()
        for table_name, df in self.mock_data.items():
            self.executor.load_data(table_name, df)
        actual_output, error_log, node_states = self.executor.execute_dag(self.current_dag, {})

        # 4. Compute reward
        reward = self.grader.compute_reward(
            actual_output=actual_output,
            expected_output=self.expected_output,
            step_count=self.episode_manager.step_count + 1,
            max_steps=self.episode_manager.max_steps,
            prev_score=prev_score,
            is_invalid_action=is_invalid,
            repeat_count=repeat_count
        )

        score_delta = round(reward - prev_score, 4)

        # 5. Build observation
        env_state = {
            'episode_id': self.episode_manager.episode_id,
            'task_level': self.task_level,
            'pipeline_dag': self.current_dag,
            'actual_output': actual_output,
            'expected_output': self.expected_output,
            'error_log': error_log,
            'step_count': self.episode_manager.step_count + 1,
            'max_steps': self.episode_manager.max_steps,
            'current_score': reward,
            'action_feedback': feedback,
            'done': False  # we set this after recording
        }

        observation = self.obs_builder.build_observation(env_state)

        # 6. Update episode state
        info = {
            'action_applied': success,
            'action_error': feedback if not success else None,
            'score_delta': score_delta,
            'faults_remaining': max(0, 5 - int(reward * 5))
        }
        self.episode_manager.record_step(action_dict, reward, info)

        # Inject done into observation
        observation['done'] = self.episode_manager.done

        return {
            'observation': observation,
            'reward': reward,
            'done': self.episode_manager.done,
            'info': info
        }

    def state(self) -> Dict[str, Any]:
        if self.episode_manager is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self.episode_manager.get_state().model_dump()
