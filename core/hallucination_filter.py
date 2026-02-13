#!/usr/bin/env python3
"""
Hallucination filter for removing unreliable/incorrect memories.

Two-stage approach:
  1. Extract only assistant-generated text from session logs
  2. Detect hallucination patterns in assistant text only

This avoids false positives from user messages, system output, and CLI errors.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stage 1: Extract assistant-only text from session logs
# --------------------------------------------------------------------------

# Patterns that mark the start of non-assistant content (skip these blocks)
_NON_ASSISTANT_MARKERS = [
    re.compile(r'^user:', re.MULTILINE),
    re.compile(r'^\[Telegram\b', re.MULTILINE),
    re.compile(r'^System:', re.MULTILINE),
    re.compile(r'^-e\s', re.MULTILINE),           # ANSI escape output
    re.compile(r'^\x1b\[', re.MULTILINE),        # ANSI CSI sequences (ESC[...)
    re.compile(r'^```', re.MULTILINE),              # code blocks (system output)
    re.compile(r'^#\s', re.MULTILINE),              # markdown headings (metadata)
    re.compile(r'^- \*\*Session', re.MULTILINE),    # session metadata
]

def _extract_assistant_text(full_text: str) -> str:
    """Extract only assistant-generated portions from a session log.
    
    Looks for lines starting with 'assistant:' or 'A:' and collects
    content until the next speaker marker.
    """
    lines = full_text.split('\n')
    assistant_chunks = []
    in_assistant = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check for assistant speaker markers
        if stripped.startswith('assistant:') or stripped.startswith('A:'):
            in_assistant = True
            # Grab content after the marker
            after = stripped.split(':', 1)[1].strip() if ':' in stripped else ''
            if after:
                assistant_chunks.append(after)
            continue
        
        # Check for non-assistant markers (end assistant block)
        if in_assistant:
            is_other_speaker = False
            for marker in _NON_ASSISTANT_MARKERS:
                if marker.match(stripped):
                    is_other_speaker = True
                    break
            
            if is_other_speaker:
                in_assistant = False
                continue
            
            # Still in assistant block — collect this line
            if stripped:
                assistant_chunks.append(stripped)
    
    return '\n'.join(assistant_chunks)


# --------------------------------------------------------------------------
# Stage 2: Hallucination patterns (applied to assistant text only)
# --------------------------------------------------------------------------

HALLUCINATION_PATTERNS = {
    # Terminal artifacts (ANSI escapes, MOTD banners, fastfetch/neofetch output)
    # These are not “hallucinations” per se, but they are almost never useful as memories.
    "terminal_artifact_patterns": [
        r"\x1b\[[0-9;]*[A-Za-z]",                        # any ANSI CSI sequence
        r"(?m)^\s*-e\s+.*\x1b\[",                        # dash/sh echo prints literal -e + escapes
        r"(?m)^\s*-e\s+.*\bOS:\b.*\bVersion:\b",         # e.g. "OS: Ubuntu - Version: 24.04"

        # The same sysinfo lines, even if the escape codes are stripped or rendered as visible glyphs.
        r"(?m)^\s*(?:-e\s+)?🖥️\s+.*\bOS:\b",             # OS line
        r"(?m)^\s*(?:-e\s+)?🏠\s+.*\bHostname:\b",       # Hostname line
        r"(?m)^\s*(?:-e\s+)?💡\s+.*\bIP Address:\b",     # IP Address line
        r"(?m)^\s*(?:OS|Hostname|IP Address):\s+",         # plain sysinfo labels (no emoji)

        r"\b(?:fastfetch|neofetch)\b",                    # common sysinfo tools
    ],

    # Confident fabrication — the model asserts things that don't exist
    "fabrication_patterns": [
        r"\b(?:installed|deployed|created|configured|set up)\b.*\b(?:successfully|complete|done)\b",
        r"✅.*\b(?:operational|running|active|ready|complete)\b",
        r"\b(?:launched|initialized|activated)\b.*\b(?:diagnostics|session|system)\b",
        r"\b(?:achieves?|reaches?)\s+\d+%",                     # fabricated metrics
        r"\b\d+(?:\.\d+)?x\s+(?:speedup|faster|improvement)\b", # fabricated benchmarks
    ],

    # Fabricated technical claims — nonexistent tools/packages/commands
    "fake_tech_patterns": [
        r"\bfrom\s+local_llm\s+import\b",           # nonexistent module
        r"\bllm[\-_]cpp\b",                          # nonexistent package
        r"\bCodeCritique\b",                         # nonexistent class
        r"\bfastembed\b",                            # likely fabricated recommendation
        r"\bMistral-7B.*\.gguf\b",                   # fabricated model paths
        r"\bclawhub\.ai\b",                          # nonexistent domain
    ],

    # Self-contradicting or retracted statements
    "retraction_patterns": [
        r"\bactually,?\s+(?:that's|this is|it's)\s+(?:not|wrong|incorrect)\b",
        r"\bcorrection:\b",
        r"\bignore\s+(?:the\s+)?(?:above|previous)\b",
        r"\bdisregard\b",
    ],

    # Speculative claims presented as fact by the assistant
    "speculation_patterns": [
        r"\bi\s+(?:think|believe|guess)\s+(?:the|this|that|it)\b",
        r"\bprobably\s+(?:should|need|would|requires?)\b",
        r"\bi'm\s+(?:guessing|assuming)\b",
        r"\bnot sure (?:if|whether|about)\b",
        r"\bunverified\b",
        r"\bunconfirmed\b",
    ],

    # Leaked structural / XML / prompt tags — never belong in memories
    "leaked_tag_patterns": [
        r"<[a-zA-Z_][\w.-]*(?:\s[^>]*)?>"  ,  # any <tag ...> — prompt/XML/tool-call leakage
        r"</[a-zA-Z_][\w.-]*>",                # closing tags
    ],

    # Placeholder/template content in assistant output
    "placeholder_patterns": [
        r"\byour-\w+-here\b",
        r"\b\[(?:PLACEHOLDER|TODO|FIXME|INSERT)\]\b",
        r"\bexample\.com\b",
        r"\byour-nextcloud\.example\.com\b",
    ],

    # Error/failure claims — the model memorized its own errors
    "error_patterns": [
        r"\berror\b.*\b(?:occurred|happened|found|detected)\b",
        r"\bfailed?\b.*\b(?:to|when|because)\b",
        r"\bfailure\b",
        r"\bdoes.*not.*work\b",
        r"\bdoesn't work\b",
        r"\bnot working\b",
        r"\bbroken\b",
        r"\bcrash(?:es|ed|ing)?\b",
    ],

    # Bad command/syntax memories — commands that were tried and didn't work
    "bad_command_patterns": [
        r"\bcommand\b.*\b(?:not found|syntax error|invalid|unknown)\b",
        r"\bsyntax\b.*\b(?:error|incorrect|wrong|invalid)\b",
        r"\binvalid\b.*\b(?:command|option|flag|argument|syntax)\b",
        r"\bunknown\b.*\b(?:command|option|flag|argument)\b",
        r"\bpermission denied\b",
        r"\bno such file or directory\b",
        r"\bcommand not found\b",
    ],

    # Self-reinforcing error loops — model re-ingesting its own failures
    "error_loop_patterns": [
        r"\btried\b.*\bbut\b.*\b(?:failed|didn't work|error)\b",
        r"\battempted\b.*\b(?:without success|unsuccessfully)\b",
        r"\bkeeps? (?:failing|crashing|erroring)\b",
        r"\bstill (?:broken|failing|not working)\b",
        r"\bpreviously failed\b",
        r"\blast time.*\b(?:failed|broke|crashed|error)\b",
    ],

    # Deprecated/obsolete approach claims
    "deprecated_patterns": [
        r"\bdeprecated\b",
        r"\bobsolete\b",
        r"\bno longer (?:works|supported|valid|available)\b",
        r"\bold (?:method|approach|way|syntax|command)\b",
        r"\breplaced by\b",
        r"\bworkaround\b.*\b(?:for|is needed)\b",
    ],

    # Stale configuration/state claims — memorized state that drifts
    "stale_state_patterns": [
        r"\bdocker\b.*\btailscale\b.*\b(?:needed|required|setup|configure)\b",
        r"\btailscale\b.*\bdocker\b.*\b(?:needed|required|setup|configure)\b",
        r"\bCNC\b.*\b(?:requires|needs|setup|config)\b",
        r"\bthe (?:correct|right|proper) (?:path|command|config|setup) is\b",
        r"\bpath\b.*\b(?:should be|needs to be|is set to)\b",
        r"\bcurrently (?:set to|configured as|running|using)\b.*\b(?:version|port|path)\b",
        r"\bverified that\b.*\b(?:works|is correct|is set)\b",
        r"\bconfirmed\b.*\b(?:working|correct|set up)\b",
    ],
}

# Severity levels
SEVERITY_LEVELS = {
    "critical": 0.9,    # Very likely false memory
    "high": 0.7,        # Probably false
    "medium": 0.5,      # Possibly false
    "low": 0.3,         # Minor uncertainty flag
}


class HallucinationDetector:
    """Detects and scores hallucinations in memory text.
    
    For session logs, extracts assistant-only text before scanning.
    For plain text, scans directly.
    """
    
    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        """Pre-compile regex patterns for efficiency."""
        compiled = {}
        flags = re.IGNORECASE if self.ignore_case else 0
        
        for category, patterns in HALLUCINATION_PATTERNS.items():
            compiled[category] = [
                (pattern, re.compile(pattern, flags))
                for pattern in patterns
            ]
        
        return compiled

    def _is_session_log(self, text: str) -> bool:
        """Detect if text is a session log (has session metadata headers)."""
        return bool(
            re.search(r'^#\s+Session:', text[:200], re.MULTILINE)
            or re.search(r'^\- \*\*Session Key\*\*:', text[:500], re.MULTILINE)
        )

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for hallucination indicators.
        
        For session logs, only assistant-generated text is analyzed.
        """
        if not text or not isinstance(text, str):
            return {
                'is_hallucination': False,
                'confidence': 0.0,
                'severity': None,
                'matched_patterns': [],
                'categories': [],
                'details': 'Invalid input'
            }
        
        # Terminal/MOTD artifacts: these should be removable even if they appear in
        # non-assistant blocks of a session log (e.g. user pasted terminal output).
        terminal_patterns = self.compiled_patterns.get('terminal_artifact_patterns', [])
        terminal_matches = []
        for pattern_str, compiled_pattern in terminal_patterns:
            if compiled_pattern.search(text):
                terminal_matches.append(pattern_str)

        if terminal_matches:
            # Avoid flagging a random literal "\x1b[" mention; require a strong sysinfo signal.
            has_sysinfo_labels = bool(
                re.search(
                    r'(?m)^\s*(?:-e\s+)?(?:🖥️|🏠|💡)?\s*(?:OS:|Hostname:|IP Address:)\b',
                    text,
                )
            )
            has_fastfetch = bool(re.search(r'\b(?:fastfetch|neofetch)\b', text, re.IGNORECASE))
            has_echo_e = bool(re.search(r'(?m)^\s*-e\s+.*\x1b\[', text))

            if has_sysinfo_labels or has_fastfetch or has_echo_e:
                return {
                    'is_hallucination': True,
                    'confidence': 0.85,
                    'severity': 'high',
                    'matched_patterns': terminal_matches,
                    'categories': ['terminal_artifact_patterns'],
                    'details': 'Terminal/MOTD ANSI sysinfo artifact detected — not suitable for memory'
                }

        # For session logs, only analyze assistant text
        if self._is_session_log(text):
            scan_text = _extract_assistant_text(text)
            if not scan_text.strip():
                return {
                    'is_hallucination': False,
                    'confidence': 0.0,
                    'severity': None,
                    'matched_patterns': [],
                    'categories': [],
                    'details': 'No assistant text found in session log'
                }
        else:
            scan_text = text
        
        matched_patterns = []
        matched_categories = set()

        # Fast-path: leaked tags are always critical — prompt/XML leakage
        tag_patterns = self.compiled_patterns.get('leaked_tag_patterns', [])
        for pattern_str, compiled_pattern in tag_patterns:
            if compiled_pattern.search(scan_text):
                matched_patterns.append(pattern_str)
                matched_categories.add('leaked_tag_patterns')
        if matched_patterns:
            return {
                'is_hallucination': True,
                'confidence': 0.95,
                'severity': 'critical',
                'matched_patterns': matched_patterns,
                'categories': ['leaked_tag_patterns'],
                'details': 'Leaked structural/XML/prompt tags detected — never valid in memories'
            }

        # Check remaining categories
        for category, patterns in self.compiled_patterns.items():
            if category == 'leaked_tag_patterns':
                continue  # already handled above
            for pattern_str, compiled_pattern in patterns:
                if compiled_pattern.search(scan_text):
                    matched_patterns.append(pattern_str)
                    matched_categories.add(category)
        
        # No matches
        if not matched_patterns:
            return {
                'is_hallucination': False,
                'confidence': 0.0,
                'severity': None,
                'matched_patterns': [],
                'categories': [],
                'details': 'No hallucination indicators detected'
            }

        # Score based on pattern count and category weight
        match_count = len(matched_patterns)
        category_count = len(matched_categories)
        base_confidence = min(0.9, (match_count * 0.25) + (category_count * 0.25))

        # Determine severity
        if category_count >= 3 or match_count >= 5:
            severity = "critical"
            confidence = min(0.95, base_confidence + 0.2)
        elif category_count >= 2 or match_count >= 3:
            severity = "high"
            confidence = min(0.85, base_confidence + 0.15)
        elif match_count >= 2:
            severity = "medium"
            confidence = min(0.7, base_confidence + 0.1)
        elif match_count == 1:
            severity = "low"
            confidence = min(0.55, base_confidence + 0.2)
        else:
            severity = "low"
            confidence = min(0.4, base_confidence)
        
        return {
            'is_hallucination': confidence >= 0.5,
            'confidence': round(confidence, 3),
            'severity': severity,
            'matched_patterns': matched_patterns,
            'categories': list(matched_categories),
            'details': f'Found {match_count} pattern(s) across {category_count} categor(ies)'
        }
    
    def is_hallucinated(self, text: str, threshold: float = 0.5) -> bool:
        """Quick boolean check: is this text likely hallucinated?"""
        result = self.detect(text)
        flagged = result['confidence'] >= threshold
        if flagged:
            logger.info(
                "Flagged as hallucination (confidence=%.3f, severity=%s): %s",
                result['confidence'], result['severity'], text[:120]
            )
        return flagged
    
    def filter_text(self, text: str, threshold: float = 0.5) -> Optional[str]:
        """Return None if hallucinated, else return original text."""
        result = self.detect(text)
        if result['confidence'] >= threshold:
            logger.warning(
                "Filtered hallucinated text (confidence=%.3f): %s",
                result['confidence'], text[:120]
            )
            return None
        return text


class HallucinationCleaner:
    """Batch cleanup of hallucinated memories."""
    
    def __init__(self, detector: Optional[HallucinationDetector] = None):
        self.detector = detector or HallucinationDetector()
    
    def score_batch(self, memories: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Score a batch of memories for hallucinations.
        
        Args:
            memories: List of dicts with 'text' or 'content' field
            threshold: Confidence threshold for flagging
        
        Returns:
            List of memories with 'hallucination_score' and 'should_remove' fields
        """
        scored = []
        
        for mem in memories:
            text = mem.get('text') or mem.get('content') or ''
            result = self.detector.detect(text)
            
            scored.append({
                **mem,
                'hallucination_score': result['confidence'],
                'hallucination_severity': result['severity'],
                'should_remove': result['confidence'] >= threshold,
                'matched_patterns': result['matched_patterns']
            })
        
        return scored
    
    def get_removal_summary(self, memories: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
        """Get summary stats on what would be removed."""
        scored = self.score_batch(memories, threshold)
        to_remove = [m for m in scored if m['should_remove']]
        
        return {
            'total_memories': len(memories),
            'flagged_count': len(to_remove),
            'flagged_percentage': round(100 * len(to_remove) / len(memories), 2) if memories else 0,
            'severity_breakdown': self._breakdown_severity(to_remove),
            'top_patterns': self._top_patterns(to_remove),
            'memories_to_remove': to_remove
        }
    
    @staticmethod
    def _breakdown_severity(memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count memories by severity level."""
        breakdown = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for mem in memories:
            severity = mem.get('hallucination_severity')
            if severity in breakdown:
                breakdown[severity] += 1
        return breakdown
    
    @staticmethod
    def _top_patterns(memories: List[Dict[str, Any]], top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most common matched patterns."""
        pattern_counts = {}
        for mem in memories:
            for pattern in mem.get('matched_patterns', []):
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


# Global detector instance
_detector = HallucinationDetector()
_cleaner = HallucinationCleaner(_detector)


def is_hallucinated(text: str, threshold: float = 0.5) -> bool:
    """Quick check: is text hallucinated?"""
    return _detector.is_hallucinated(text, threshold)


def detect_hallucination(text: str) -> Dict[str, Any]:
    """Full detection with details."""
    return _detector.detect(text)


def filter_hallucinations_batch(memories: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Filter and score batch of memories."""
    return _cleaner.score_batch(memories, threshold)


def get_removal_summary(memories: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    """Get summary of memories that would be removed."""
    return _cleaner.get_removal_summary(memories, threshold)


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python3 hallucination_filter.py <text>")
        print("  or: python3 hallucination_filter.py --batch <json_file>")
        sys.exit(1)
    
    if sys.argv[1] == "--batch" and len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            memories = json.load(f)
        summary = get_removal_summary(memories)
        print(json.dumps(summary, indent=2))
    else:
        text = " ".join(sys.argv[1:])
        result = detect_hallucination(text)
        print(json.dumps(result, indent=2))
