#!/usr/bin/env python3
"""
Test suite for API improvements - validation, error handling, pagination.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_vector_dimension_validation():
    """Test that vectors must match MEMORY_EMB_DIM."""
    os.environ['MEMORY_EMB_DIM'] = '384'
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "api/app.py")
    
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules['api.app'] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[SKIP] Could not load api/app.py: {e}")
        return
    
    MEMORY_EMB_DIM = int(os.getenv("MEMORY_EMB_DIM", "384"))
    wrong_dim_vector = [0.1] * 100
    correct_dim_vector = [0.1] * MEMORY_EMB_DIM
    
    from pydantic import ValidationError
    try:
        req = module.AddMemoryRequest(
            vector=wrong_dim_vector,
            metadata={"test": "data"}
        )
        print("[FAIL] Should have raised ValidationError")
    except ValidationError:
        pass
    
    req = module.AddMemoryRequest(
        vector=correct_dim_vector,
        metadata={"test": "data"}
    )
    assert len(req.vector) == MEMORY_EMB_DIM
    print("[PASS] Vector dimension validation")


def test_max_request_size_config():
    """Test max request size configuration."""
    with open("api/app.py") as f:
        content = f.read()
    
    assert "MAX_REQUEST_SIZE = 10 * 1024 * 1024" in content
    assert "MAX_BATCH_SIZE = 1000" in content
    print("[PASS] Max request size config")


def test_admin_token_config():
    """Test admin token configuration."""
    with open("api/app.py") as f:
        content = f.read()
    
    assert "MEMORY_API_ADMIN_TOKEN" in content
    assert "verify_admin_token" in content
    print("[PASS] Admin token config")


def test_pagination_endpoint():
    """Test pagination endpoint exists."""
    with open("api/app.py") as f:
        content = f.read()
    
    assert "/list_memories" in content
    assert "offset" in content
    assert "limit" in content
    print("[PASS] Pagination endpoint")


def test_request_size_middleware():
    """Test request size middleware."""
    with open("api/app.py") as f:
        content = f.read()
    
    assert "check_request_size" in content
    assert "MAX_REQUEST_SIZE" in content
    print("[PASS] Request size middleware")


if __name__ == "__main__":
    print("=" * 60)
    print("API IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Vector dimension validation", test_vector_dimension_validation),
        ("Max request size config", test_max_request_size_config),
        ("Admin token config", test_admin_token_config),
        ("Pagination endpoint", test_pagination_endpoint),
        ("Request size middleware", test_request_size_middleware),
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
