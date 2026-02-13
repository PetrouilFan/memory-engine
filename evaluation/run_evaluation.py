import json
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from core.memory_engine import hybrid_search, EMB_DIM

EVAL_DIR = os.path.dirname(__file__)

# Load gold-standard queries
with open(os.path.join(EVAL_DIR, 'gold_queries.json'), 'r') as f:
    queries = json.load(f)

results = []
for q in queries:
    phrase = q['query']
    expected_ids = set(q['expected_ids'])
    vec = np.random.rand(EMB_DIM).tolist()
    hits = hybrid_search(phrase, vec, top_k=10)
    retrieved_ids = [r['id'] for r in hits]
    precision = len(set(retrieved_ids) & expected_ids) / len(retrieved_ids) if retrieved_ids else 0
    results.append({
        'query': phrase,
        'precision@10': precision,
        'timestamp': time.time()
    })

output_path = os.path.join(EVAL_DIR, 'eval_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Evaluation run complete, results saved to {output_path}')
