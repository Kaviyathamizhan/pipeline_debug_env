from typing import Dict, Any, Tuple
from ..models import PipelineAction

class ActionParser:
    def __init__(self, executor):
        self.executor = executor

    def validate_and_apply(self, action_dict: Dict[str, Any], dag: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates the action semantically. If valid, applies the change to the DAG.
        Returns: (success_bool, feedback_string, updated_dag)
        """
        try:
            action = PipelineAction(**action_dict)
        except Exception as e:
            return False, f"Action Schema Error: {str(e)}", dag

        target_node = action.target_node
        node_idx = next((i for i, n in enumerate(dag['nodes']) if n['name'] == target_node), -1)
        
        if node_idx == -1 and action.action_type != 'reorder_nodes':
            return False, f"Semantic Error: Target node '{target_node}' does not exist in DAG.", dag

        node_data = dag['nodes'][node_idx] if node_idx != -1 else {}
        params = action.params

        # 1. patch_schema: Requires old_column to exist in upstream or node's output
        if action.action_type == 'patch_schema':
            old_col = params.get('old_column')
            new_col = params.get('new_column')
            if not old_col or not new_col:
                return False, "Semantic Error: 'old_column' and 'new_column' required.", dag
                
            # Usually applied as a projection wrap.
            new_sql = f"SELECT * RENAME {old_col} TO {new_col} FROM ({node_data.get('sql', '')})"
            dag['nodes'][node_idx]['sql'] = new_sql
            return True, "", dag

        # 2. rewrite_transform
        elif action.action_type == 'rewrite_transform':
            if 'new_sql' not in params:
                return False, "Semantic Error: 'new_sql' parameter required.", dag
            dag['nodes'][node_idx]['sql'] = params['new_sql']
            return True, "", dag

        # 3. add_null_guard
        elif action.action_type == 'add_null_guard':
            col = params.get('column')
            strat = params.get('strategy', 'drop')
            if not col:
                return False, "Semantic Error: 'column' parameter required.", dag
            
            base_sql = node_data.get('sql', '')
            if strat == 'drop':
                new_sql = f"SELECT * FROM ({base_sql}) WHERE {col} IS NOT NULL"
            else:
                return False, f"Semantic Error: strategy '{strat}' not fully supported yet.", dag
            dag['nodes'][node_idx]['sql'] = new_sql
            return True, "", dag

        # 4. add_type_cast
        elif action.action_type == 'add_type_cast':
            col = params.get('column')
            to_type = params.get('to_type')
            if not col or not to_type:
                return False, "Semantic Error: 'column' and 'to_type' required.", dag

            base_sql = node_data.get('sql', '')
            new_sql = f"SELECT * EXCLUDE ({col}), TRY_CAST({col} AS {to_type}) AS {col} FROM ({base_sql})"
            dag['nodes'][node_idx]['sql'] = new_sql
            return True, "", dag

        # 5. fix_join_key
        elif action.action_type == 'fix_join_key':
            left_key = params.get('left_key')
            right_key = params.get('right_key')
            if not left_key or not right_key:
                return False, "Semantic Error: 'left_key' and 'right_key' required.", dag
            
            # Simple string replace for demonstration in DAG SQL
            old_sql = node_data.get('sql', '')
            # Very naive replacement. Real env might use AST mapping.
            new_sql = old_sql.replace(f"ON \n", f"ON {left_key} = {right_key}\n-- ")
            dag['nodes'][node_idx]['sql'] = new_sql
            return True, "", dag

        # 6. invert_filter
        elif action.action_type == 'invert_filter':
            expr = params.get('filter_expression', '')
            if not expr:
                return False, "Semantic Error: 'filter_expression' parameter required.", dag
            base_sql = node_data.get('sql', '')
            # Invert boolean conditions
            new_sql = base_sql.replace('= TRUE', '= FALSE').replace('= FALSE', '= TRUE')
            if new_sql == base_sql:
                # Try NOT wrapping
                new_sql = base_sql.replace('WHERE ', f'WHERE NOT (') + ')'
            dag['nodes'][node_idx]['sql'] = new_sql
            return True, "", dag

        elif action.action_type == 'no_op':
            return True, "", dag

        return False, f"Semantic Error: Unknown action_type '{action.action_type}'", dag
