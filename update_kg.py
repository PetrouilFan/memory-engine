#!/usr/bin/env python3
"""
Update knowledge graph from workspace sources.

Run this periodically (e.g., daily) to keep the KG in sync with workspace.
"""
import sys
import subprocess
from pathlib import Path

WORKSPACE = Path("/root/.openclaw/workspace")
MEMORY_ENGINE = WORKSPACE / "projects/memory-engine"

def main():
    print("Updating knowledge graph...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(MEMORY_ENGINE / "pipelines" / "knowledge_graph_exporter.py")],
            capture_output=True,
            text=True,
            cwd=str(MEMORY_ENGINE)
        )
        if result.returncode != 0:
            print(f"Exporter failed: {result.stderr}")
            return 1
        
        print(f"Exporter: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"Failed to run exporter: {e}")
        return 1
    
    try:
        result = subprocess.run(
            [sys.executable, str(MEMORY_ENGINE / "import_kg.py")],
            capture_output=True,
            text=True,
            cwd=str(MEMORY_ENGINE)
        )
        if result.returncode != 0:
            print(f"Importer failed: {result.stderr}")
            return 1
        
        print(f"Importer: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"Failed to run importer: {e}")
        return 1
    
    print("Knowledge graph updated successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
