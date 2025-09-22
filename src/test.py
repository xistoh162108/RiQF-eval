#!/usr/bin/env python3
"""
Simple test script for Enhanced VLM Interface

This demonstrates basic usage of the enhanced VLM interface.
"""

import os
import tempfile
from enhanced_vlm_interface import EnhancedVLMInterface, ModelCapability, ModelFactory

def main():
    """Basic test of the enhanced VLM interface."""
    print("🧪 Enhanced VLM Interface Test")
    print("=" * 40)

    # Initialize interface
    vlm = EnhancedVLMInterface()

    # Show available models
    available_models = ModelFactory.get_available_models()
    print(f"📋 Available models: {len(available_models)}")

    # Create a temporary model file for local testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy model data")
        temp_path = temp_file.name

    try:
        # Configure some models
        print("\n⚙️ Configuring models...")

        # API model (with dummy key for demo)
        vlm.configure_model('openai-gpt-4', api_key='demo-key-123')
        print("✅ Configured OpenAI GPT-4")

        # Local model
        vlm.configure_model('llama-7b', model_path=temp_path)
        print("✅ Configured Llama 7B")

        # Create and test a prompt
        print("\n🎯 Testing prompt execution...")
        prompt = vlm.create_prompt(
            system_input="You are a helpful assistant",
            user_prompt="What is artificial intelligence?",
            task_type=ModelCapability.TEXT_GENERATION,
            temperature=0.7
        )

        # Execute query
        response, exec_time = vlm.execute_query('openai-gpt-4', prompt)
        print(f"Response: {response}")
        print(f"Execution time: {exec_time:.3f} seconds")

        # Test model info
        info = vlm.get_model_info('openai-gpt-4')
        print(f"Model capabilities: {info['capabilities']}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    finally:
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    main()
