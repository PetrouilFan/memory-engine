#!/usr/bin/env python3
"""
Test suite for text_sanitizer module.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

from text_sanitizer import TextSanitizer, get_sanitizer


def test_sanitizer_singleton():
    """Test that get_sanitizer returns a singleton."""
    s1 = get_sanitizer()
    s2 = get_sanitizer()
    assert s1 is s2, "get_sanitizer should return singleton"
    print("[PASS] Singleton test")


def test_strip_meta():
    """Test metadata prefix stripping."""
    sanitizer = TextSanitizer()
    
    test_cases = [
        ("[Telegram Petros Fan (@petrouil) id:5381354945 +1m] Hello world", "Hello world"),
        ("[Discord user#1234] Some message", "Some message"),
        ("[Slack C01AB] regular text", "regular text"),
        ("Plain text without prefix", "Plain text without prefix"),
        ("", ""),
    ]
    
    for input_text, expected in test_cases:
        result = sanitizer.strip_meta(input_text)
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("[PASS] strip_meta test")


def test_has_leaked_tags():
    """Test leaked tag detection."""
    sanitizer = TextSanitizer()
    
    leaked_cases = [
        "<function_calls>",
        "</invoke>",
        "<parameter name='x'>",
        "<thinking>thought</thinking>",
        "<system>prompt</system>",
    ]
    
    clean_cases = [
        "This is plain text",
        "5 < 10 is true in math",
        "Peter's code uses stdio.h",
    ]
    
    for text in leaked_cases:
        assert sanitizer.has_leaked_tags(text), f"Should detect leaked tag in: {text}"
    
    for text in clean_cases:
        assert not sanitizer.has_leaked_tags(text), f"Should NOT detect leaked tag in: {text}"
    
    print("[PASS] has_leaked_tags test")


def test_remove_leaked_tags():
    """Test tag removal."""
    sanitizer = TextSanitizer()
    
    test_cases = [
        ("<tag>content</tag>", "content"),
        ("<function_calls><invoke>test</invoke></function_calls>", "test"),
        ("plain text", "plain text"),
        ("5 < 10", "5 < 10"),
    ]
    
    for input_text, expected in test_cases:
        result = sanitizer.remove_leaked_tags(input_text)
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("[PASS] remove_leaked_tags test")


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT SANITIZER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Singleton", test_sanitizer_singleton),
        ("strip_meta", test_strip_meta),
        ("has_leaked_tags", test_has_leaked_tags),
        ("remove_leaked_tags", test_remove_leaked_tags),
    ]
    
    all_pass = True
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {e}")
            all_pass = False
        except Exception as e:
            print(f"[ERROR] {e}")
            all_pass = False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
