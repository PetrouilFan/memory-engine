#!/usr/bin/env python3
"""
Test suite for hallucination_filter module.
"""
import sys
import os

# Add core directory to path
core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core')
sys.path.insert(0, core_dir)

from hallucination_filter import HallucinationDetector, HallucinationCleaner


def _run_cases(label, cases):
    detector = HallucinationDetector()
    passed = 0
    for text, expect_flag in cases:
        result = detector.detect(text)
        got = result['is_hallucination']
        ok = got == expect_flag
        passed += int(ok)
        mark = '✓' if ok else '✗'
        print(f"  [{mark}] '{text[:65]}' → hallucinated={got} "
              f"(confidence={result['confidence']:.2f}, severity={result['severity']})")
    print(f"  {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_leaked_tags():
    """Leaked XML/prompt/tool-call tags must be instant-critical."""
    return _run_cases("Leaked tags", [
        # These all contain <tags> and must be flagged
        ('<function_calls><invoke name="exec"><parameter name="cmd">ls</parameter></invoke></function_calls>', True),
        ('<parameter name="command">python3 recall.py</parameter>', True),
        ('<invoke name="read"><parameter name="path">memory</parameter></invoke>', True),
        ('<thinking>let me think about this</thinking>', True),
        ('<result>some output</result>', True),
        ('<system>you are a helpful assistant</system>', True),
        # Clean text should NOT be flagged by tag patterns alone
        ('The user prefers dark mode and uses vim keybindings', False),
        ('Peter lives in Amsterdam and works on OpenClaw', False),
        ('Meeting notes from the Monday standup', False),
    ])


def test_fabrication_patterns():
    """Confident fabrications (model claiming it did things)."""
    return _run_cases("Fabrication", [
        ('I installed and deployed the system successfully and it is complete', True),
        ('✅ All diagnostics operational and running and ready', True),
        ('Achieves 98% accuracy on the benchmark', True),
        ('3.5x speedup improvement over baseline', True),
        ('The team had a productive meeting', False),
    ])


def test_error_patterns():
    """Memorized errors should be flagged (the model re-ingesting failures)."""
    return _run_cases("Errors", [
        ('An error occurred when connecting to the database', True),
        ('The process failed to start because of missing deps', True),
        ("doesn't work with the latest version", True),
        ('The service crashed during deployment', True),
        ('We shipped the feature on Friday', False),
    ])


def test_speculation_patterns():
    """Speculative claims presented as fact."""
    return _run_cases("Speculation", [
        ('I think the server probably should use port 8080', True),
        ("I'm assuming this is the right approach", True),
        ('not sure if this will work', True),
        ('The server runs on port 8080', False),
    ])


def test_error_loop_patterns():
    """Self-reinforcing error loops."""
    return _run_cases("Error loops", [
        ('I tried deploying but it failed again', True),
        ('Keeps failing on the CI pipeline', True),
        ('Still not working after the update', True),
        ('Last time this broke the build and we had an error', True),
        ('The deployment pipeline runs nightly', False),
    ])


def test_deprecated_patterns():
    """Deprecated/obsolete approach memories."""
    return _run_cases("Deprecated", [
        ('This method is deprecated', True),
        ('No longer supported in the latest version', True),
        ('The old method was replaced by the new API', True),
        ('We use the v2 API for authentication', False),
    ])


def test_batch_cleaning():
    """Test batch memory cleaning."""
    cleaner = HallucinationCleaner()

    memories = [
        {"text": "Peter prefers dark mode", "id": 1},
        {"text": '<function_calls><invoke name="exec"></invoke></function_calls>', "id": 2},
        {"text": "The deployment crashed during rollout", "id": 3},
        {"text": "Meeting with the team went well", "id": 4},
        {"text": "I think probably we should try this approach", "id": 5},
        {"text": "This is deprecated and no longer works", "id": 6},
    ]

    summary = cleaner.get_removal_summary(memories, threshold=0.5)

    print(f"\n  Batch Cleaning Summary:")
    print(f"    Total: {summary['total_memories']}")
    print(f"    Flagged: {summary['flagged_count']} ({summary['flagged_percentage']:.1f}%)")
    print(f"    Severity: {summary['severity_breakdown']}")
    if summary['top_patterns']:
        print(f"    Top patterns:")
        for pattern, count in summary['top_patterns'][:5]:
            print(f"      • {pattern}: {count}x")

    # IDs 2, 3, 5, 6 should be flagged; 1, 4 should be kept
    flagged_ids = {m['id'] for m in summary['memories_to_remove']}
    assert 2 in flagged_ids, "Leaked tags (id=2) should be flagged"
    assert 1 not in flagged_ids, "Clean memory (id=1) should be kept"
    assert 4 not in flagged_ids, "Clean memory (id=4) should be kept"
    print("  Batch cleaning assertions passed ✓")


def test_clean_memories_not_flagged():
    """Ensure normal, useful memories are never flagged."""
    return _run_cases("Clean memories", [
        ('Peter uses Neovim with Lua config on his Mac', False),
        ('The gateway runs on port 18789 behind Tailscale', False),
        ('Discord bot token is stored in the config file', False),
        ('We deployed the fix for the auth flow on Monday', False),
        ('The cron job runs every 6 hours', False),
        ('OpenClaw supports Telegram, Discord, Slack, Signal, and iMessage', False),
        ('Memory engine uses FAISS with 384-dim embeddings', False),
        ('The hallucination filter was updated to catch leaked tags', False),
    ])


if __name__ == "__main__":
    print("=" * 80)
    print("HALLUCINATION FILTER TEST SUITE")
    print("=" * 80)

    tests = [
        ("Leaked Tags", test_leaked_tags),
        ("Fabrication", test_fabrication_patterns),
        ("Errors", test_error_patterns),
        ("Speculation", test_speculation_patterns),
        ("Error Loops", test_error_loop_patterns),
        ("Deprecated", test_deprecated_patterns),
        ("Batch Cleaning", test_batch_cleaning),
        ("Clean Memories", test_clean_memories_not_flagged),
    ]

    all_ok = True
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        print("-" * 80)
        try:
            result = fn()
            if result is False:
                all_ok = False
        except Exception as e:
            print(f"  FAILED: {e}")
            all_ok = False

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓" if all_ok else "SOME TESTS FAILED ✗")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)
