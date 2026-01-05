#!/usr/bin/env python3
"""
Test script for UCUP framework.
"""

import sys
import os

# Add the framework's Python modules to the path
framework_path = os.path.dirname(os.path.abspath(__file__))
python_modules_path = os.path.join(framework_path, "Versions", "A", "PythonModules")
sys.path.insert(0, python_modules_path)

def test_basic_import():
    """Test basic import of UCUP modules."""
    print("Testing basic imports...")

    try:
        import ucup
        print(f"✅ UCUP version: {ucup.__version__}")
        print(f"✅ UCUP author: {ucup.__author__}")

        # Test that core components can be imported
        from ucup.core.config import UCUPConfig
        from ucup.core.manager import UCUPManager
        from ucup.core.bridge import ObjectiveCBridge
        from ucup.probabilistic.engine import ProbabilisticEngine
        print("✅ Core modules imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test UCUP configuration system."""
    print("\nTesting configuration system...")

    try:
        from ucup.core.config import UCUPConfig

        # Test default config
        config = UCUPConfig.default()
        print("✅ Default config created")

        # Test config values
        core_ml_enabled = config.get("core_ml.enabled")
        gcd_enabled = config.get("gcd.enable_threading")
        print(f"✅ Core ML enabled: {core_ml_enabled}")
        print(f"✅ GCD enabled: {gcd_enabled}")

        # Test config setting
        config.set("test_key", "test_value")
        assert config.get("test_key") == "test_value"
        print("✅ Config get/set works")

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_probabilistic_engine():
    """Test probabilistic engine."""
    print("\nTesting probabilistic engine...")

    try:
        from ucup.core.config import UCUPConfig
        from ucup.core.manager import UCUPManager
        from ucup.probabilistic.engine import ProbabilisticEngine

        config = UCUPConfig.default()
        manager = UCUPManager(config)
        engine = manager.get_probabilistic_engine()

        # Test prediction
        input_data = {"input_value": 75}
        result = engine.predict(input_data)

        print("✅ Prediction result keys:", list(result.keys()))
        assert "probability" in result
        assert "confidence_interval" in result
        print(f"✅ Prediction probability: {result['probability']:.3f}")

        return True
    except Exception as e:
        print(f"❌ Probabilistic engine test failed: {e}")
        return False

def test_objective_c_bridge():
    """Test Objective-C bridge."""
    print("\nTesting Objective-C bridge...")

    try:
        from ucup.core.bridge import ObjectiveCBridge

        bridge = ObjectiveCBridge()
        initialized = bridge.initialize()
        print(f"✅ Bridge initialization: {initialized}")

        # Test system info
        system_info = bridge.get_system_info()
        if system_info:
            print("✅ System info retrieved:", list(system_info.keys()))
        else:
            print("⚠️  No system info available (expected in development)")

        return True
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        return False

def test_multimodal_processing():
    """Test multimodal data processing."""
    print("\nTesting multimodal processing...")

    try:
        from ucup.core.config import UCUPConfig
        from ucup.core.manager import UCUPManager

        config = UCUPConfig.default()
        manager = UCUPManager(config)

        # Test multimodal analysis (mock data)
        test_data = [
            {"type": "image", "id": "test_image_1"},
            {"type": "audio", "id": "test_audio_1"}
        ]

        result = manager.analyze_multimodal_data(test_data)
        print("✅ Multimodal analysis completed")
        print("✅ Result keys:", list(result.keys()))

        return True
    except Exception as e:
        print(f"❌ Multimodal test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 UCUP Framework Test Suite")
    print("=" * 40)

    tests = [
        test_basic_import,
        test_config,
        test_probabilistic_engine,
        test_objective_c_bridge,
        test_multimodal_processing,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! UCUP framework is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
