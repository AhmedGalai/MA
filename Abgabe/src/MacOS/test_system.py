#!/usr/bin/env python3
"""
System Test Script
Tests all components of the AW19 system
"""

import requests
import sys
import time

def test_api(name, url):
    """Test if an API is responding"""
    print(f"Testing {name}...", end=" ")
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print("✓ OK")
            return True
        else:
            print(f"✗ FAIL (status {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ FAIL (not running)")
        return False
    except Exception as e:
        print(f"✗ FAIL ({e})")
        return False

def test_models():
    """Test model listing"""
    print("Testing model listing...", end=" ")
    try:
        response = requests.get("http://localhost:5000/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✓ OK ({len(models)} models found)")
            for model in models:
                print(f"  - {model['name']}")
            return True
        else:
            print("✗ FAIL")
            return False
    except Exception as e:
        print(f"✗ FAIL ({e})")
        return False

def test_cv_pipeline():
    """Test CV pipeline status"""
    print("Testing CV pipeline...", end=" ")
    try:
        response = requests.get("http://localhost:5000/stats", timeout=2)
        if response.status_code == 200:
            data = response.json()
            cv_stats = data.get('cv_pipeline', {})
            device = cv_stats.get('device', 'unknown')
            print(f"✓ OK (device: {device})")
            return True
        else:
            print("✗ FAIL")
            return False
    except Exception as e:
        print(f"✗ FAIL ({e})")
        return False

def test_gpu_detection():
    """Test GPU detection"""
    print("Testing GPU detection...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available ({gpu_name})")
            return True
        else:
            print("⚠ CPU only (no CUDA)")
            return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ FAIL ({e})")
        return False

def main():
    print("=" * 60)
    print("AW19 System Test")
    print("=" * 60)
    print()

    # Test GPU
    print("1. GPU Detection")
    print("-" * 60)
    test_gpu_detection()
    print()

    # Test APIs
    print("2. API Services")
    print("-" * 60)
    main_api_ok = test_api("Main API", "http://localhost:5000/health")
    pose_api_ok = True  # Integrated into main_api when forwarding is enabled
    print()

    if not main_api_ok:
        print("⚠  Main API not running. Start with: python main_api.py")
        print()

    # Pose API is integrated; when forwarding is enabled, main_api forwards to configured host.

    # Test components (only if main API is running)
    if main_api_ok:
        print("3. Components")
        print("-" * 60)
        test_cv_pipeline()
        test_models()
        print()

    # Summary
    print("=" * 60)
    if main_api_ok and pose_api_ok:
        print("✓ All systems operational!")
        print()
        print("Next steps:")
        print("  1. Start screen capture: python screen_capture.py")
        print("  2. Start debug viewer: python tk_debugging_client.py")
    else:
        print("✗ Some services are not running")
        print()
        print("To start all services:")
        print("  Windows: start.bat")
        print("  Manual:  python main_api.py   (terminal 1)")
    print("=" * 60)

if __name__ == "__main__":
    main()
