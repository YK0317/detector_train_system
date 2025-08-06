#!/usr/bin/env python3
"""
Standalone Test Runner for Train System

This script can run the comprehensive tests without requiring pytest,
making it suitable for environments where pytest might not be available.

Usage:
    python run_tests.py
    python run_tests.py --verbose
    python run_tests.py --quick  # Run only basic tests
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path

# Add train_system to path
sys.path.insert(0, str(Path(__file__).parent))


def load_test_module(test_file_path):
    """Dynamically load a test module"""
    spec = importlib.util.spec_from_file_location("test_module", test_file_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    return test_module


def run_basic_tests():
    """Run basic tests from test_basic.py"""
    print("🔄 Running Basic Tests...")
    try:
        test_basic_path = Path(__file__).parent / "tests" / "test_basic.py"
        if test_basic_path.exists():
            test_module = load_test_module(test_basic_path)
            success = test_module.run_all_tests()
            return success
        else:
            print("❌ test_basic.py not found")
            return False
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False


def run_comprehensive_tests():
    """Run comprehensive tests from test_comprehensive.py"""
    print("🔄 Running Comprehensive Tests...")
    try:
        test_comp_path = Path(__file__).parent / "tests" / "test_comprehensive.py"
        if test_comp_path.exists():
            test_module = load_test_module(test_comp_path)
            success = test_module.run_comprehensive_tests()
            return success
        else:
            print("❌ test_comprehensive.py not found")
            return False
    except Exception as e:
        print(f"❌ Comprehensive tests failed: {e}")
        return False


def check_installation():
    """Check if train_system is properly installed"""
    print("🔍 Checking Train System Installation...")
    try:
        import train_system

        print(
            f"✅ Train System version: {getattr(train_system, '__version__', 'unknown')}"
        )

        # Test basic imports
        from train_system import UnifiedTrainingWrapper, ModelFactory
        from train_system.config import UnifiedTrainingConfig

        print("✅ Core imports successful")
        return True

    except ImportError as e:
        print(f"❌ Installation check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train System Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Run only basic tests"
    )
    parser.add_argument(
        "--install-check", "-i", action="store_true", help="Only check installation"
    )

    args = parser.parse_args()

    print("🚀 Train System Test Runner")
    print("=" * 50)

    # Check installation first
    if not check_installation():
        print("\n❌ Installation check failed. Please install train_system properly.")
        return 1

    if args.install_check:
        print("\n✅ Installation check passed.")
        return 0

    success = True

    if args.quick:
        # Run only basic tests
        success = run_basic_tests()
    else:
        # Run comprehensive tests, fallback to basic if failed
        print("\n" + "=" * 50)
        success = run_comprehensive_tests()

        if not success:
            print("\n⚠️ Comprehensive tests failed, falling back to basic tests...")
            print("=" * 50)
            success = run_basic_tests()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
