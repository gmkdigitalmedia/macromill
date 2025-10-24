"""
API Testing Script
Simple script to test the sentiment analysis API endpoints
"""

import requests
import json
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"


def test_health_check() -> bool:
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")

        if data.get("model_loaded"):
            print("[PASS] Health check passed - Model is loaded")
            return True
        else:
            print("[FAIL] Health check failed - Model not loaded")
            return False

    except Exception as e:
        print(f"[FAIL] Health check failed: {str(e)}")
        return False


def test_single_prediction() -> bool:
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Single Prediction Endpoint")
    print("="*60)

    test_cases = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Terrible waste of time. Poor acting, bad script, and boring storyline.",
        "It was okay, nothing special but not terrible either.",
    ]

    try:
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Input: {text[:80]}...")

            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": text}
            )
            response.raise_for_status()
            data = response.json()

            print(f"Sentiment: {data['sentiment']}")
            print(f"Confidence: {data['confidence']:.4f}")
            print(f"Probabilities: {data['probabilities']}")
            print("[PASS] Prediction successful")

        return True

    except Exception as e:
        print(f"[FAIL] Single prediction test failed: {str(e)}")
        return False


def test_batch_prediction() -> bool:
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction Endpoint")
    print("="*60)

    reviews = [
        "One of the best films I've ever seen! Highly recommend!",
        "Awful movie. I want my money back.",
        "The cinematography was stunning, but the story was lacking.",
        "A masterpiece of modern cinema!",
        "Boring and predictable. Skip this one."
    ]

    try:
        print(f"Sending {len(reviews)} reviews for batch prediction...")

        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={"reviews": reviews}
        )
        response.raise_for_status()
        data = response.json()

        print(f"\nTotal predictions: {data['total_count']}")
        print(f"Processing time: {data['processing_time_ms']:.2f} ms")
        print(f"\nResults:")

        for i, pred in enumerate(data['predictions'], 1):
            print(f"\n{i}. {pred['text'][:60]}...")
            print(f"   Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.4f})")

        print("\n[PASS] Batch prediction successful")
        return True

    except Exception as e:
        print(f"[FAIL] Batch prediction test failed: {str(e)}")
        return False


def test_model_info() -> bool:
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/model/info")
        response.raise_for_status()
        data = response.json()

        print(f"Model Type: {data['model_type']}")
        print(f"Device: {data['device']}")
        print(f"Parameters: {data['parameters']:,}")
        print(f"Max Length: {data['max_length']}")
        print("[PASS] Model info retrieved successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Model info test failed: {str(e)}")
        return False


def test_error_handling() -> bool:
    """Test error handling"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)

    # Test empty text
    print("\nTest: Empty text")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": "   "}
        )
        if response.status_code == 400:
            print("[PASS] Empty text correctly rejected (400)")
        else:
            print(f"[FAIL] Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Error test failed: {str(e)}")
        return False

    # Test oversized batch
    print("\nTest: Oversized batch")
    try:
        large_batch = ["Test review"] * 50  # Exceeds MAX_BATCH_SIZE of 32
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={"reviews": large_batch}
        )
        if response.status_code == 422:  # Validation error
            print("[PASS] Oversized batch correctly rejected (422)")
        else:
            print(f"[FAIL] Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Error test failed: {str(e)}")
        return False

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("SENTIMENT ANALYSIS API TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")

    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info),
        ("Error Handling", test_error_handling)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS] PASSED" if result else "[FAIL] FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n All tests passed!")
        return 0
    else:
        print(f"\nWARNING:  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
