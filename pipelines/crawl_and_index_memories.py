#!/usr/bin/env python3
"""
DEPRECATED: This crawler has been merged into crawl_memory_files.py.

Use crawl_memory_files.py instead — it includes:
  - Semantic chunking
  - Hallucination filtering
  - Leaked tag rejection
  - Multi-directory support
"""
import sys

def main():
    print("⚠ This script is deprecated. Use crawl_memory_files.py instead.")
    print("  python3 pipelines/crawl_memory_files.py")
    sys.exit(1)

if __name__ == "__main__":
    main()
