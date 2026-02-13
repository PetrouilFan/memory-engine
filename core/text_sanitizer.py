#!/usr/bin/env python3
"""
text_sanitizer.py – Centralized text sanitization for chat platform metadata
and tag leakage detection.

Combines sanitization logic from holographic_memory.py and memory_bridge.py
into a single reusable module.
"""
import re
from typing import Optional


class TextSanitizer:
    # Pre-compiled tag regex — any <tag>/</tag> is prompt leakage
    TAG_RE = re.compile(r"</?[a-zA-Z_][\w.-]*(?:\s[^>]*)?>")

    # Chat platform metadata lines to strip before storing
    CHAT_META_RE = re.compile(
        r"(?m)^\s*(?:user:\s*)?\[(?:Telegram|Discord|WhatsApp|Slack)\b[^\]]*\]\s*$"
    )
    # Bare role markers ("user:", "assistant:") with nothing after them
    BARE_ROLE_RE = re.compile(r"(?m)^\s*(?:user|assistant|system):\s*$")
    # Raw numeric platform IDs (id:1234567890)
    PLATFORM_ID_RE = re.compile(r"\bid:\d{5,}")
    # Timestamps like "+9h 2026-02-02 09:16 GMT+2"
    CHAT_TS_RE = re.compile(r"[+-]\d+h\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+GMT[+-]\d+")

    # Memory bridge specific: metadata prefix pattern (more comprehensive)
    META_PREFIX_RE = re.compile(
        r"^\s*\[(?:Telegram|Discord|WhatsApp|Signal|Slack|IRC|Matrix)\s[^\]]*\]\s*",
        re.IGNORECASE,
    )

    def __init__(self):
        pass

    def sanitize(self, text: str) -> str:
        """Strip chat platform metadata, IDs, and timestamps before storing.
        
        This applies all metadata stripping patterns followed by blank line collapsing.
        
        Args:
            text: The input text to sanitize.
            
        Returns:
            Sanitized text with metadata removed and excess blank lines collapsed.
        """
        text = self.META_PREFIX_RE.sub("", text)
        text = self.CHAT_META_RE.sub("", text)
        text = self.BARE_ROLE_RE.sub("", text)
        text = self.PLATFORM_ID_RE.sub("", text)
        text = self.CHAT_TS_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def strip_meta(self, text: str) -> str:
        """Strip chat platform metadata prefix from text.
        
        This is a lighter-weight strip used for query preparation in memory_bridge.
        
        Args:
            text: The input text to strip metadata from.
            
        Returns:
            Text with platform metadata prefix removed.
        """
        return self.META_PREFIX_RE.sub("", text).strip()

    def has_leaked_tags(self, text: str) -> bool:
        """Check if text contains leaked tags (prompt/XML/tool-call leakage).
        
        Args:
            text: The text to check for leaked tags.
            
        Returns:
            True if any leaked tags are found, False otherwise.
        """
        return bool(self.TAG_RE.search(text))

    def remove_leaked_tags(self, text: str) -> str:
        """Remove all HTML/XML-like tags from text.
        
        Args:
            text: The input text to clean.
            
        Returns:
            Text with all tags removed.
        """
        return self.TAG_RE.sub("", text)


# Singleton instance for convenience
_default_sanitizer: Optional[TextSanitizer] = None


def get_sanitizer() -> TextSanitizer:
    """Get the default TextSanitizer instance."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = TextSanitizer()
    return _default_sanitizer
