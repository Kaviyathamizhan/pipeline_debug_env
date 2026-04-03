import copy
from pipeline_debug_env.server.fault_injector import FaultInjector

def test_fault_actually_changes_sql():
    """Verify faults produce REAL changes across all templates."""
    import yaml
    
    templates = ['ecommerce_orders', 'user_engagement', 'financial_revenue']
    for tname in templates:
        with open(f'pipeline_debug_env/templates/{tname}.yaml') as f:
            t = yaml.safe_load(f)
        
        dag = {'nodes': t['nodes'], 'edges': []}
        fi = FaultInjector()
        
        for level in ['easy', 'medium', 'hard']:
            faulty = fi.apply_faults(copy.deepcopy(dag), level, seed=42)
            
            # At least one node must have changed SQL
            changed = any(
                c['sql'] != f['sql']
                for c, f in zip(dag['nodes'], faulty['nodes'])
            )
            assert changed, f"Template {tname}/{level}: No faults applied!"


def test_apply_faults_deterministic():
    injector = FaultInjector()
    base_dag = {
        'nodes': [
            {'name': 'ingest', 'node_type': 'ingest', 'sql': 'SELECT user_id, amount FROM t'},
        ],
        'edges': []
    }
    res1 = injector.apply_faults(copy.deepcopy(base_dag), 'easy', seed=42)
    res2 = injector.apply_faults(copy.deepcopy(base_dag), 'easy', seed=42)
    
    assert res1 == res2


def test_fault_variety_with_different_seeds():
    """Different seeds should produce different faults."""
    import yaml
    fi = FaultInjector()
    
    with open('pipeline_debug_env/templates/ecommerce_orders.yaml') as f:
        t = yaml.safe_load(f)
    dag = {'nodes': t['nodes'], 'edges': []}
    
    results = set()
    for seed in range(42, 52):
        faulty = fi.apply_faults(copy.deepcopy(dag), 'easy', seed=seed)
        # Create a signature of which nodes changed
        sig = tuple(
            (n['name'], n['sql'] != orig['sql'])
            for n, orig in zip(faulty['nodes'], dag['nodes'])
        )
        results.add(sig)
    
    # At least 2 different fault configurations across 10 seeds
    assert len(results) >= 2, f"Only got {len(results)} unique fault patterns across 10 seeds"
