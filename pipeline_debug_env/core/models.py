from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

class PipelineAction(BaseModel):
    action_type: str = Field(..., description="Action type: patch_schema, rewrite_transform, add_null_guard, reorder_nodes, fix_join_key, invert_filter, add_type_cast, no_op")
    target_node: str = Field(..., description="Pipeline node name to apply the action to")
    params: Dict[str, Any] = Field(..., description="Action-specific parameters")
    reasoning: Optional[str] = Field(None, description="Agent's explanation for this action")

class SchemaField(BaseModel):
    name: str
    dtype: str
    nullable: bool

class PipelineNode(BaseModel):
    name: str
    node_type: str
    status: str
    sql: Optional[str] = None
    
class PipelineDAG(BaseModel):
    nodes: List[PipelineNode]
    edges: List[List[str]]

class PipelineObservation(BaseModel):
    episode_id: str
    task_level: str
    pipeline_dag: PipelineDAG
    error_log: List[str]
    schema_diff: Dict[str, Any]
    sample_rows: Dict[str, List[Dict[str, Any]]]
    row_count_diff: int
    column_stats: Dict[str, Any]
    step_count: int
    max_steps: int
    current_score: float
    action_feedback: str
    done: bool

class EpisodeState(BaseModel):
    episode_id: str
    step_count: int
    max_steps: int
    current_score: float
    best_score: float
    task_level: str
    done: bool
    score_history: List[float]
