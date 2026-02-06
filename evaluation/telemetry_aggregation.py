import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from core.memory_engine import get_stats

EVAL_DIR = os.path.dirname(__file__)

stats = get_stats()
log_entry = {
    'timestamp': time.time(),
    'stats': stats
}

log_path = os.path.join(EVAL_DIR, 'telemetry_log.jsonl')
with open(log_path, 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
print(f'Logged telemetry to {log_path}')
