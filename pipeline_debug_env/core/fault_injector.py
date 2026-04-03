import random
import re
import copy
from typing import Dict, Any


class FaultInjector:
    def __init__(self):
        self.fault_types = [
            'schema_drift', 'type_mismatch', 'null_propagation',
            'boolean_inversion', 'filter_removal'
        ]

    def _find_columns_in_sql(self, sql: str) -> list:
        """Extract column-like identifiers from SQL, excluding table names."""
        # Find words after FROM/JOIN — these are table references, not columns
        table_refs = set(re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE))
        
        cols = re.findall(r'\b([a-z][a-z_]+(?:_[a-z]+)*)\b', sql.lower())
        keywords = {'select', 'from', 'where', 'and', 'or', 'not', 'null',
                    'group', 'by', 'order', 'as', 'is', 'in', 'on', 'join',
                    'left', 'right', 'inner', 'outer', 'count', 'sum', 'avg',
                    'try_cast', 'cast', 'float', 'varchar', 'int', 'integer',
                    'true', 'false', 'limit', 'asc', 'desc', 'having', 'replace',
                    'exclude', 'rename', 'like', 'between', 'case', 'when', 'then',
                    'else', 'end', 'distinct', 'all', 'into', 'values', 'insert',
                    'update', 'delete', 'create', 'table', 'drop', 'alter', 'index'}
        return [c for c in cols if c not in keywords and c not in table_refs and len(c) > 2]

    def _inject_schema_drift(self, dag, node):
        """Rename a real column found in the SQL to a camelCase variant."""
        sql = node['sql']
        # Only rename known column identifiers, never table names
        renames = {
            'user_id': 'userId',
            'order_id': 'orderId',
            'session_id': 'sessionId',
            'event_type': 'eventType',
            'txn_id': 'txnId',
            'txn_date': 'txnDate',
            'is_churned': 'isChurned',
            'is_deleted': 'isDeleted',
            'total_revenue': 'totalRevenue',
            'event_count': 'eventCount',
        }
        # Find table names to protect them
        table_refs = set(re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE))
        
        for col, camel in renames.items():
            # Only rename if it exists AND is not a table reference
            if col in sql and col.lower() not in {t.lower() for t in table_refs}:
                node['sql'] = sql.replace(col, camel)
                return

    def _inject_type_mismatch(self, dag, node):
        """Wrap a numeric column with VARCHAR cast, breaking downstream aggregations."""
        sql = node['sql']
        # Find columns that look numeric (amount, count, revenue, etc.)
        numeric_cols = [c for c in self._find_columns_in_sql(sql)
                        if any(k in c for k in ['amount', 'revenue', 'count', 'total', 'price', 'qty'])]

        if numeric_cols:
            col = numeric_cols[0]
            # Wrap entire node output with a type corruption
            node['sql'] = f"SELECT * REPLACE (TRY_CAST({col} AS VARCHAR) AS {col}) FROM ({sql})"
        else:
            # Fallback: just wrap with VARCHAR on first column
            cols = self._find_columns_in_sql(sql)
            if cols:
                col = cols[0]
                node['sql'] = f"SELECT * REPLACE (TRY_CAST({col} AS VARCHAR) AS {col}) FROM ({sql})"

    def _inject_null_propagation(self, dag, node):
        """Remove WHERE clauses that filter nulls, allowing bad data through."""
        sql = node['sql']
        # Remove null guards
        modified = re.sub(r'\s+WHERE\s+\w+\s+IS\s+NOT\s+NULL', '', sql, flags=re.IGNORECASE)
        if modified != sql:
            node['sql'] = modified
            return

        # Remove any WHERE clause entirely (more aggressive)
        modified = re.sub(r'\s+WHERE\s+.+?(?=\s+GROUP|\s+ORDER|\s+LIMIT|$)', '', sql, flags=re.IGNORECASE)
        if modified != sql:
            node['sql'] = modified

    def _inject_boolean_inversion(self, dag, node):
        """Flip boolean conditions in WHERE clauses."""
        sql = node['sql']
        if '= FALSE' in sql:
            node['sql'] = sql.replace('= FALSE', '= TRUE')
        elif '= TRUE' in sql:
            node['sql'] = sql.replace('= TRUE', '= FALSE')
        elif 'IS NOT NULL' in sql:
            node['sql'] = sql.replace('IS NOT NULL', 'IS NULL')

    def _inject_filter_removal(self, dag, node):
        """Completely remove a WHERE clause, letting all data through unfiltered."""
        sql = node['sql']
        # Remove WHERE ... but keep GROUP BY / ORDER BY
        modified = re.sub(r'\s+WHERE\s+.+?(?=\s+GROUP|\s+ORDER|\s+LIMIT|$)', ' ', sql, flags=re.IGNORECASE)
        if modified.strip() != sql.strip():
            node['sql'] = modified

    def apply_faults(self, template_dag: Dict[str, Any], task_level: str, seed: int = None) -> Dict[str, Any]:
        """
        Takes a healthy DAG and applies faults dynamically based on actual SQL content.
        Hard tasks deliberately stack faults to create misleading error propagation.
        """
        if seed is not None:
            random.seed(seed)

        faulty_dag = copy.deepcopy(template_dag)

        num_faults = {'easy': 1, 'medium': 3, 'hard': 5}.get(task_level, 1)
        # Only target non-output nodes
        eligible_nodes = [n for n in faulty_dag['nodes'] if n.get('node_type') not in ['output']]

        if not eligible_nodes:
            return faulty_dag

        applied_count = 0
        attempts = 0
        max_attempts = num_faults * 5

        while applied_count < num_faults and attempts < max_attempts:
            attempts += 1
            
            if task_level == 'hard':
                # HARD (FIX 3): Enforce specific interacting upstream faults
                # to create cascading failures and misleading downstream logs.
                if applied_count == 0:
                    fault = 'type_mismatch'
                elif applied_count == 1:
                    fault = 'null_propagation'
                elif applied_count == 2:
                    fault = 'schema_drift'
                else:
                    # Stacking extra random faults on top of core cascade
                    fault = random.choice(self.fault_types)
                    
                upstream = [n for n in eligible_nodes if n.get('node_type') in ('ingest', 'clean')]
                node = random.choice(upstream) if upstream else random.choice(eligible_nodes)
            else:
                # EASY/MEDIUM (FIX 2): Random faults, explicitly spread out across nodes.
                # No intentional hidden dependency chains.
                fault = random.choice(self.fault_types)
                if len(eligible_nodes) > 1:
                    available = [n for n in eligible_nodes
                                 if n['name'] not in {n2['name'] for i, n2 in enumerate(eligible_nodes)
                                                       if i < applied_count}]
                    if not available:
                        available = eligible_nodes
                    node = random.choice(available)
                else:
                    node = eligible_nodes[0]

            old_sql = node['sql']

            if fault == 'schema_drift':
                self._inject_schema_drift(faulty_dag, node)
            elif fault == 'type_mismatch':
                self._inject_type_mismatch(faulty_dag, node)
            elif fault == 'null_propagation':
                self._inject_null_propagation(faulty_dag, node)
            elif fault == 'boolean_inversion':
                self._inject_boolean_inversion(faulty_dag, node)
            elif fault == 'filter_removal':
                self._inject_filter_removal(faulty_dag, node)

            # Count if SQL actually changed
            if node['sql'] != old_sql:
                applied_count += 1

        return faulty_dag
