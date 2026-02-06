# Offline Evaluation Harness

This directory contains tools for building a robust offline evaluation suite for the memory engine.

## Goals
- Define a JSON schema for gold-standard query sets.
- Implement scripts to run queries against the memory engine and compute ranking metrics (precision@k, recall, MAP, nDCG).
- Store results over time for regression tracking.

## Files
- `gold_queries.json` – Example queries with expected memory IDs
- `run_evaluation.py` – Runs queries against memory_engine and computes metrics
- `telemetry_aggregation.py` – Logs memory engine stats over time
