#!/usr/bin/env python3
"""
Import Examples for VLM Interface

This script demonstrates different ways to import and use the VLM interfaces.
"""

def test_direct_imports():
    """Test importing directly from the modules."""
    print("🔗 Testing Direct Imports")
    print("-" * 30)

    # Direct import from enhanced module
    from enhanced_vlm_interface import (
        EnhancedVLMInterface,
        ModelCapability,
        ModelFactory
    )

    vlm = EnhancedVLMInterface()
    print(f"✅ Created EnhancedVLMInterface: {type(vlm).__name__}")

    # Test capabilities
    text_models = ModelFactory.get_models_by_capability(ModelCapability.TEXT_GENERATION)
    print(f"✅ Found {len(text_models)} text generation models")

    # Test legacy interface
    from vlm_interface import VLMInterface
    legacy_vlm = VLMInterface()
    print(f"✅ Created legacy VLMInterface: {type(legacy_vlm).__name__}")


def test_package_imports():
    """Test importing from the package __init__.py."""
    print("\n📦 Testing Package Imports")
    print("-" * 30)

    # Import from package (when running from parent directory)
    try:
        import sys
        import os

        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)

        from vlm_test import (
            EnhancedVLMInterface,
            VLMInterface,
            ModelCapability,
            ModelFactory
        )

        print("✅ Package imports successful")

        # Test both interfaces
        enhanced_vlm = EnhancedVLMInterface()
        legacy_vlm = VLMInterface()

        print(f"✅ Enhanced interface: {type(enhanced_vlm).__name__}")
        print(f"✅ Legacy interface: {type(legacy_vlm).__name__}")

        # Show model counts
        available_models = ModelFactory.get_available_models()
        print(f"✅ Available models: {len(available_models)}")

    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        print("ℹ️  Note: Run from project root directory for package imports")


def test_specific_imports():
    """Test importing specific components."""
    print("\n🎯 Testing Specific Component Imports")
    print("-" * 40)

    # Import specific capabilities
    from enhanced_vlm_interface import ModelCapability, ModelProvider

    print("Available Capabilities:")
    for capability in ModelCapability:
        print(f"  • {capability.value}")

    print("\nAvailable Providers:")
    for provider in ModelProvider:
        print(f"  • {provider.value}")

    # Import factory
    from enhanced_vlm_interface import ModelFactory

    # Show models by provider
    print("\nModels by Provider:")
    for provider in ModelProvider:
        models = ModelFactory.get_models_by_provider(provider)
        if models:
            print(f"  {provider.value}: {len(models)} models")


def main():
    """Run all import tests."""
    print("🧪 VLM Interface Import Tests")
    print("=" * 50)

    test_direct_imports()
    test_package_imports()
    test_specific_imports()

    print("\n" + "=" * 50)
    print("🎉 Import tests completed!")
    print("\nUsage Examples:")
    print("1. Direct import: from enhanced_vlm_interface import EnhancedVLMInterface")
    print("2. Package import: from src import EnhancedVLMInterface")
    print("3. Specific imports: from enhanced_vlm_interface import ModelCapability")


if __name__ == "__main__":
    main()