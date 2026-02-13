#!/bin/bash
cd /root/.openclaw/workspace/projects/memory-engine
echo "Running hallucination filter tests..."
python3 tests/test_hallucination_filter.py
