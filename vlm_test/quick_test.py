#!/usr/bin/env python3
"""
Quick test script for Enhanced VLM Interface

This script provides a simple demonstration of the enhanced VLM interface
capabilities, perfect for quick testing and validation.
"""

from enhanced_vlm_interface import (
    EnhancedVLMInterface,
    ModelCapability,
    ModelFactory
)
import tempfile
import os


def quick_demo():
    """Quick demonstration of enhanced VLM interface."""
    print("🚀 Enhanced VLM Interface - Quick Demo")
    print("=" * 50)

    # Initialize interface
    vlm = EnhancedVLMInterface()

    # Show available models
    print("📋 Available Models:")
    for model_id, config in ModelFactory.get_available_models().items():
        print(f"  • {model_id:<20} ({config.provider.value}) - {[c.value for c in config.capabilities]}")

    print("\n🔍 Model Discovery:")
    print(f"  Text generation: {len(ModelFactory.get_models_by_capability(ModelCapability.TEXT_GENERATION))} models")
    print(f"  Vision-language: {len(ModelFactory.get_models_by_capability(ModelCapability.VISION_LANGUAGE))} models")
    print(f"  Document understanding: {len(ModelFactory.get_models_by_capability(ModelCapability.DOCUMENT_UNDERSTANDING))} models")
    print(f"  Code generation: {len(ModelFactory.get_models_by_capability(ModelCapability.CODE_GENERATION))} models")

    # Create dummy model file for local testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy model data")
        temp_path = temp_file.name

    try:
        print("\n⚙️ Configuring Models:")

        # Configure API models (with dummy keys for demo)
        api_models = ['openai-gpt-4', 'gemini-pro', 'deepseek-chat']
        for model_id in api_models:
            try:
                vlm.configure_model(model_id, api_key='demo-key-123')
                print(f"  ✅ {model_id}")
            except Exception as e:
                print(f"  ❌ {model_id}: {e}")

        # Configure local models
        local_models = ['llama-7b', 'docowl', 'donut']
        for model_id in local_models:
            try:
                vlm.configure_model(model_id, model_path=temp_path)
                print(f"  ✅ {model_id}")
            except Exception as e:
                print(f"  ❌ {model_id}: {e}")

        print(f"\n📊 Configured Models: {vlm.get_configured_models()}")

        # Test different prompt types
        print("\n🧪 Testing Different Prompt Types:")

        # Text generation prompt
        text_prompt = vlm.create_prompt(
            system_input="You are a helpful assistant",
            user_prompt="Explain artificial intelligence in one sentence",
            task_type=ModelCapability.TEXT_GENERATION,
            temperature=0.7,
            max_tokens=100
        )

        # Document understanding prompt
        doc_prompt = vlm.create_prompt(
            system_input="Analyze document content",
            user_prompt="Extract key information",
            task_type=ModelCapability.DOCUMENT_UNDERSTANDING,
            document_path="/demo/document.pdf",
            output_format="json"
        )

        # Test model execution
        test_models = ['openai-gpt-4', 'llama-7b', 'docowl']

        for model_id in test_models:
            if model_id in vlm.get_configured_models():
                try:
                    if model_id == 'docowl':
                        response, exec_time = vlm.execute_query(model_id, doc_prompt)
                    else:
                        response, exec_time = vlm.execute_query(model_id, text_prompt)

                    print(f"\n  🎯 {model_id}:")
                    print(f"    Response: {str(response)[:80]}...")
                    print(f"    Time: {exec_time:.3f}s")

                    # Get model info
                    info = vlm.get_model_info(model_id)
                    print(f"    Provider: {info['provider']}")
                    print(f"    Capabilities: {info['capabilities']}")

                except Exception as e:
                    print(f"  ❌ {model_id}: {e}")

        # Test capability validation
        print("\n🛡️ Testing Capability Validation:")

        # Try to use image with text-only model
        try:
            image_prompt = vlm.create_prompt(
                system_input="Analyze this image",
                user_prompt="What do you see?",
                image_path="/demo/image.jpg"
            )
            vlm.execute_query('llama-7b', image_prompt)
        except ValueError as e:
            print(f"  ✅ Correctly caught capability mismatch: {e}")

        # Test batch configuration
        print("\n📦 Testing Batch Configuration:")
        batch_configs = {
            'openai-gpt-3.5-turbo': {'api_key': 'demo-key'},
            'gemini-pro-vision': {'api_key': 'demo-key'},
            'smoldocling': {'model_path': temp_path},
            'invalid-model': {'api_key': 'demo-key'}  # Should fail
        }

        results = vlm.batch_configure_models(batch_configs)
        for model_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {model_id}")

        print(f"\n🎉 Demo completed! Final model count: {len(vlm.get_configured_models())}")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

    finally:
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass


if __name__ == "__main__":
    quick_demo()