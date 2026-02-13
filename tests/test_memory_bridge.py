#!/usr/bin/env python3
"""
Test suite for memory_bridge improvements.
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_environment_variables():
    """Test that environment variables are properly configured."""
    from memory_bridge import (
        GROQ_API_URL,
        TARGET_MODEL,
        BRIDGE_PORT,
        MEMORY_RECALL_K,
        MEMORY_RECALL_THRESHOLD,
        BRIDGE_API_KEY,
    )
    
    assert GROQ_API_URL == os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    assert BRIDGE_PORT == int(os.getenv("BRIDGE_PORT", "19192"))
    assert MEMORY_RECALL_K == int(os.getenv("MEMORY_RECALL_K", "3"))
    assert MEMORY_RECALL_THRESHOLD == float(os.getenv("MEMORY_RECALL_THRESHOLD", "0.3"))
    
    print("[PASS] Environment variables configuration")


def test_extract_text():
    """Test _extract_text helper function."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("memory_bridge", "memory_bridge.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules['memory_bridge'] = module
        spec.loader.exec_module(module)
        
        _extract_text = module._extract_text
        
        result1 = _extract_text("plain text")
        assert result1 == "plain text", f"Expected 'plain text', got '{result1}'"
        
        result2 = _extract_text([{"type": "text", "text": "hello world"}])
        assert result2 == "hello world", f"Expected 'hello world', got '{result2}'"
        
        print("[PASS] _extract_text function")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise


def test_health_endpoint():
    """Test health check endpoint exists."""
    from fastapi.testclient import TestClient
    from memory_bridge import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "memory_bridge"
    
    print("[PASS] Health endpoint")


def test_bridge_api_key_validation():
    """Test API key validation when enabled."""
    import os
    os.environ["BRIDGE_API_KEY"] = "test-key-123"
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("memory_bridge", "memory_bridge.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules['memory_bridge'] = module
    spec.loader.exec_module(module)
    
    from fastapi.testclient import TestClient
    app = module.app
    
    client = TestClient(app)
    
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401
    
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
        headers={"X-API-Key": "test-key-123"}
    )
    assert response.status_code in [200, 400, 500, 502, 504]
    
    del os.environ["BRIDGE_API_KEY"]
    print("[PASS] Bridge API key validation")


def test_missing_messages_validation():
    """Test that missing messages are rejected."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("memory_bridge", "memory_bridge.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules['memory_bridge'] = module
        spec.loader.exec_module(module)
        
        from fastapi.testclient import TestClient
        app = module.app
        
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 400
        
        print("[PASS] Missing messages validation")
    except Exception as e:
        print(f"[FAIL] {e}")
        raise


def test_recall_config():
    """Test recall configuration."""
    from memory_bridge import MEMORY_RECALL_K, MEMORY_RECALL_THRESHOLD
    
    assert isinstance(MEMORY_RECALL_K, int)
    assert isinstance(MEMORY_RECALL_THRESHOLD, float)
    assert 0 < MEMORY_RECALL_THRESHOLD < 1
    
    print("[PASS] Recall configuration")


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY BRIDGE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Environment variables", test_environment_variables),
        ("_extract_text function", test_extract_text),
        ("Health endpoint", test_health_endpoint),
        ("Bridge API key validation", test_bridge_api_key_validation),
        ("Missing messages validation", test_missing_messages_validation),
        ("Recall configuration", test_recall_config),
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
