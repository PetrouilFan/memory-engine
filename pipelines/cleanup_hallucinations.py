#!/usr/bin/env python3
"""
Pipeline to identify and remove hallucinated memories from the workspace.
Supports scanning and cleaning memory files.
"""
import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add core directory to path for imports
core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core')
sys.path.insert(0, core_dir)

from hallucination_filter import HallucinationCleaner, HallucinationDetector


class MemoryCleanupPipeline:
    """Scans and cleans hallucinated memories from markdown files."""
    
    def __init__(self, memory_dir: str, dry_run: bool = True):
        self.memory_dir = memory_dir
        self.dry_run = dry_run
        self.detector = HallucinationDetector()
        self.cleaner = HallucinationCleaner(self.detector)
    
    def scan_memory_files(self) -> List[str]:
        """Find all memory markdown files."""
        pattern = os.path.join(self.memory_dir, "*.md")
        files = sorted(glob.glob(pattern))
        logger.info(f"Found {len(files)} memory files in {self.memory_dir}")
        return files
    
    def read_memory_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Read and parse a memory file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'content': content,
                'size_bytes': len(content.encode('utf-8'))
            }
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def scan_all_memories(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Scan all memory files and identify hallucinations."""
        files = self.scan_memory_files()
        
        memories_data = []
        hallucinated_files = []
        
        for filepath in files:
            mem = self.read_memory_file(filepath)
            if not mem:
                continue
            
            result = self.detector.detect(mem['content'])
            
            if result['confidence'] >= threshold:
                hallucinated_files.append({
                    **mem,
                    'hallucination_score': result['confidence'],
                    'hallucination_severity': result['severity'],
                    'matched_patterns': result['matched_patterns'],
                    'categories': result['categories']
                })
            
            memories_data.append(mem)
        
        return {
            'total_files': len(files),
            'hallucinated_count': len(hallucinated_files),
            'hallucinated_percentage': round(100 * len(hallucinated_files) / len(files), 2) if files else 0,
            'hallucinated_files': hallucinated_files,
            'severity_breakdown': self._breakdown_severity(hallucinated_files),
            'top_patterns': self._top_patterns(hallucinated_files)
        }
    
    def remove_hallucinated_files(self, threshold: float = 0.5, confirm: bool = False) -> Dict[str, Any]:
        """
        Remove files identified as hallucinated.
        
        Args:
            threshold: Confidence threshold for removal (0-1)
            confirm: If True, actually delete. If False (dry_run), just report.
        
        Returns:
            Summary of removed files
        """
        scan_result = self.scan_all_memories(threshold)
        files_to_remove = scan_result['hallucinated_files']
        
        removed = []
        failed = []
        
        for mem in files_to_remove:
            filepath = mem['filepath']
            try:
                if not confirm and self.dry_run:
                    logger.info(f"[DRY RUN] Would remove: {filepath} (score: {mem['hallucination_score']:.2f})")
                    removed.append({
                        'filepath': filepath,
                        'filename': mem['filename'],
                        'score': mem['hallucination_score'],
                        'severity': mem['hallucination_severity'],
                        'status': 'dry_run'
                    })
                else:
                    os.remove(filepath)
                    logger.info(f"✓ Removed: {filepath} (score: {mem['hallucination_score']:.2f})")
                    removed.append({
                        'filepath': filepath,
                        'filename': mem['filename'],
                        'score': mem['hallucination_score'],
                        'severity': mem['hallucination_severity'],
                        'status': 'removed'
                    })
            except Exception as e:
                logger.error(f"Failed to remove {filepath}: {e}")
                failed.append({
                    'filepath': filepath,
                    'error': str(e)
                })
        
        return {
            'scan_result': scan_result,
            'removed_count': len(removed),
            'failed_count': len(failed),
            'removed_files': removed,
            'failed_files': failed,
            'dry_run': not confirm,
            'size_freed_bytes': sum(mem.get('size_bytes', 0) for mem in removed)
        }
    
    def generate_report(self, threshold: float = 0.5, save_to: Optional[str] = None) -> str:
        """Generate detailed report of hallucinations."""
        scan_result = self.scan_all_memories(threshold)
        
        lines = [
            "=" * 80,
            "MEMORY HALLUCINATION ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Memory Directory: {self.memory_dir}",
            f"Total Files: {scan_result['total_files']}",
            f"Hallucinated: {scan_result['hallucinated_count']} ({scan_result['hallucinated_percentage']}%)",
            "",
            "SEVERITY BREAKDOWN:",
            json.dumps(scan_result['severity_breakdown'], indent=2),
            "",
            "TOP MATCHED PATTERNS:",
        ]
        
        for pattern, count in scan_result['top_patterns'][:15]:
            lines.append(f"  • {pattern}: {count} occurrences")
        
        if scan_result['hallucinated_files']:
            lines.extend([
                "",
                "HALLUCINATED FILES (sorted by confidence):",
                "-" * 80,
            ])
            
            sorted_files = sorted(
                scan_result['hallucinated_files'],
                key=lambda x: x['hallucination_score'],
                reverse=True
            )
            
            for mem in sorted_files[:50]:  # Show top 50
                lines.append(
                    f"  [{mem['hallucination_score']:.2f}] {mem['filename']} "
                    f"({mem['hallucination_severity']}) - {len(mem['matched_patterns'])} patterns"
                )
        
        report = "\n".join(lines)
        
        if save_to:
            with open(save_to, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_to}")
        
        return report
    
    @staticmethod
    def _breakdown_severity(memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count memories by severity."""
        breakdown = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for mem in memories:
            severity = mem.get('hallucination_severity')
            if severity in breakdown:
                breakdown[severity] += 1
        return breakdown
    
    @staticmethod
    def _top_patterns(memories: List[Dict[str, Any]], top_n: int = 15) -> List[tuple]:
        """Get top N patterns."""
        pattern_counts = {}
        for mem in memories:
            for pattern in mem.get('matched_patterns', []):
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean hallucinated memories from workspace"
    )
    parser.add_argument(
        "--memory-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'memory'),
        help="Path to memory directory (default: /root/.openclaw/workspace/memory)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Hallucination confidence threshold (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and report hallucinations (default)"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove hallucinated files (requires --confirm)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive operations (remove)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be removed without actually removing (default)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Save detailed report to file"
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=['json', 'text'],
        default='text',
        help="Output format (default: text)"
    )
    
    args = parser.parse_args()
    
    # Ensure memory dir exists
    if not os.path.isdir(args.memory_dir):
        logger.error(f"Memory directory not found: {args.memory_dir}")
        sys.exit(1)
    
    pipeline = MemoryCleanupPipeline(args.memory_dir, dry_run=not args.confirm)
    
    if args.remove:
        result = pipeline.remove_hallucinated_files(
            threshold=args.threshold,
            confirm=args.confirm
        )
        
        if args.output == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"\nRemoval Summary:")
            print(f"  Scanned: {result['scan_result']['total_files']} files")
            print(f"  Removed: {result['removed_count']}")
            print(f"  Failed: {result['failed_count']}")
            print(f"  Dry Run: {result['dry_run']}")
            if result['size_freed_bytes']:
                print(f"  Size Freed: {result['size_freed_bytes'] / 1024:.2f} KB")
    else:
        report = pipeline.generate_report(
            threshold=args.threshold,
            save_to=args.report
        )
        
        if args.output == 'json':
            scan_result = pipeline.scan_all_memories(args.threshold)
            print(json.dumps(scan_result, indent=2))
        else:
            print(report)


if __name__ == "__main__":
    main()
