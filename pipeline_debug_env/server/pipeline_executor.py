import duckdb
import pandas as pd
from typing import Dict, Any, List

class PipelineExecutor:
    def __init__(self):
        # Use an in-memory duckdb connection
        self.con = duckdb.connect(database=':memory:')

    def load_data(self, table_name: str, df: pd.DataFrame):
        # Register a dataframe into duckdb namespace
        self.con.register(table_name, df)

    def execute_node(self, node_name: str, sql: str) -> pd.DataFrame:
        """
        Executes a SQL transform and registers the result as a new table 
        matching the node_name.
        """
        try:
            # We first run the query to a df
            result_df = self.con.execute(sql).df()
            # Then we register the result under the node's name so downstream can use it
            self.con.register(node_name, result_df)
            return result_df
        except Exception as e:
            raise RuntimeError(f"Error in node '{node_name}': {str(e)}")

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Retrieves the exact schema from duckdb for a given registered table.
        """
        try:
            schema_df = self.con.execute(f"DESCRIBE {table_name}").df()
            schema = {}
            for _, row in schema_df.iterrows():
                schema[row['column_name']] = {
                    'dtype': row['column_type'],
                    'nullable': row['null'] == 'YES'
                }
            return schema
        except Exception:
            return {}

    def fetch_bad_rows(self, table_name: str) -> pd.DataFrame:
        """
        Executes a smart sample, prioritizing rows with NULLs for observation signal.
        """
        try:
            cols = self.get_table_schema(table_name).keys()
            if not cols:
                return pd.DataFrame()
                
            null_conditions = " OR ".join([f'"{c}" IS NULL' for c in cols])
            
            # Fetch up to 5 problematic rows
            bad_rows = self.con.execute(f"SELECT * FROM {table_name} WHERE {null_conditions} LIMIT 5").df()
            
            # Fetch normal rows to fill up to 8 max
            rem = 8 - len(bad_rows)
            if rem > 0:
                good_rows = self.con.execute(f"SELECT * FROM {table_name} LIMIT {rem}").df()
                return pd.concat([bad_rows, good_rows]).drop_duplicates().head(8)
            return bad_rows
        except Exception:
            return pd.DataFrame()

    def reset(self):
        self.con.close()
        self.con = duckdb.connect(database=':memory:')

    def execute_dag(self, dag: Dict[str, Any], initial_data: Dict[str, pd.DataFrame]):
        """
        Executes the entire DAG and captures outputs/errors.
        """
        # Load initial tables
        for k, v in initial_data.items():
            self.load_data(k, v)
            
        logs = []
        node_states = {}
        final_output = pd.DataFrame()
        
        # Sort nodes topologically based on edges, assuming linear or correctly ordered for now based on dag logic
        for node in dag.get('nodes', []):
            name = node['name']
            sql = node.get('sql')
            if not sql:
                continue
                
            try:
                df = self.execute_node(name, sql)
                node_states[name] = {'status': 'ok', 'df': df}
                if node['node_type'] == 'output':
                    final_output = df
            except Exception as e:
                logs.append(str(e))
                node_states[name] = {'status': 'error'}
                # Cascade failures downstream if we abort on first
                break
                
        return final_output, logs, node_states
