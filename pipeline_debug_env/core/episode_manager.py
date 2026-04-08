import json

class EpisodeManager:
    def __init__(self, episode_id: str, task_level: str, max_steps: int):
        self.episode_id = episode_id
        self.task_level = task_level
        self.max_steps = max_steps
        self.step_count = 0
        self.current_score = 0.01
        self.best_score = 0.01
        self.score_history = []
        self.action_history = []
        self.done = False

    def check_repeated_action(self, action_dict: dict) -> int:
        """
        Check if the exact same action has been repeated.
        Returns the number of times it has been repeated previously.
        """
        if not self.action_history:
            return 0
            
        action_json = json.dumps({
            k: v for k, v in action_dict.items() if k != "reasoning"
        }, sort_keys=True)
        
        repeats = 0
        for past_action in reversed(self.action_history):
            past_json = json.dumps({
                k: v for k, v in past_action.items() if k != "reasoning"
            }, sort_keys=True)
            if past_json == action_json:
                repeats += 1
            else:
                break # only count consecutive repeats
                
        return repeats

    def record_step(self, action_dict: dict, reward: float, info: dict):
        self.step_count += 1
        self.current_score = reward
        if reward > self.best_score:
            self.best_score = reward
            
        self.score_history.append(reward)
        self.action_history.append(action_dict)
        
        # Determine if done
        if reward >= 0.95 or self.step_count >= self.max_steps:
            self.done = True

    def get_state(self):
        from .models import EpisodeState
        return EpisodeState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            current_score=self.current_score,
            best_score=self.best_score,
            task_level=self.task_level,
            done=self.done,
            score_history=self.score_history
        )
