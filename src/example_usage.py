#!/usr/bin/env python3
"""
Example usage of the VLM Interface

This script demonstrates how to use the VLM interface for both API-based
and on-device models with various prompt configurations.
"""

import os
from vlm_interface import VLMInterface


def example_api_model():
    """Example using an API-based model."""
    print("=== API Model Example ===")

    # Initialize the VLM interface
    vlm = VLMInterface()

    # Configure an API-based model
    try:
        model = vlm.configure_model(
            model_name='CLIP_API',
            api_key='your-api-key-here'  # Replace with actual API key
        )
        print(f"✓ Configured API model: {model.model_name}")

        # Create a prompt for image description
        prompt = vlm.create_prompt(
            system_input="This system uses the CLIP model to process both images and text simultaneously.",
            training_example="Example: Show image and generate a description.",
            user_prompt="Please describe the content of the image.",
            output_format='text'
        )

        # Execute the query
        response, execution_time = vlm.execute_query('CLIP_API', prompt)
        print(f"Response: {response}")
        print(f"Execution Time: {execution_time:.3f} seconds")

    except Exception as e:
        print(f"✗ Error with API model: {e}")


def example_local_model():
    """Example using a local on-device model."""
    print("\n=== Local Model Example ===")

    # Initialize the VLM interface
    vlm = VLMInterface()

    # Create a dummy model file for demonstration
    dummy_model_path = "/tmp/dummy_clip_model.pkl"

    try:
        # Create dummy file
        with open(dummy_model_path, 'w') as f:
            f.write("dummy model data")

        # Configure a local model
        model = vlm.configure_model(
            model_name='CLIP_Local',
            model_path=dummy_model_path
        )
        print(f"✓ Configured local model: {model.model_name}")

        # Create a prompt for JSON output
        prompt = vlm.create_prompt(
            system_input="Analyze the image and provide structured output.",
            training_example=[
                "Example 1: Image of a cat -> {'object': 'cat', 'confidence': 0.95}",
                "Example 2: Image of a car -> {'object': 'car', 'confidence': 0.87}"
            ],
            user_prompt="Analyze this image and return object detection results.",
            output_format='json'
        )

        # Execute the query
        response, execution_time = vlm.execute_query('CLIP_Local', prompt)
        print(f"Response: {response}")
        print(f"Execution Time: {execution_time:.3f} seconds")

        # Clean up dummy file
        os.remove(dummy_model_path)

    except Exception as e:
        print(f"✗ Error with local model: {e}")
        # Clean up dummy file if it exists
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)


def example_multiple_models():
    """Example managing multiple models."""
    print("\n=== Multiple Models Example ===")

    vlm = VLMInterface()

    # Create dummy model files
    model_paths = {
        'CLIP': '/tmp/clip_model.pkl',
        'BLIP': '/tmp/blip_model.pkl'
    }

    try:
        # Create dummy files
        for path in model_paths.values():
            with open(path, 'w') as f:
                f.write("dummy model data")

        # Configure multiple models
        vlm.configure_model('CLIP', model_path=model_paths['CLIP'])
        vlm.configure_model('BLIP', model_path=model_paths['BLIP'])
        vlm.configure_model('GPT4V', api_key='dummy-api-key')

        print(f"✓ Configured models: {vlm.get_configured_models()}")

        # Create different prompts for different models
        clip_prompt = vlm.create_prompt(
            system_input="CLIP model for image-text matching.",
            user_prompt="Find similarity between this image and the text 'a dog in the park'."
        )

        blip_prompt = vlm.create_prompt(
            system_input="BLIP model for image captioning.",
            user_prompt="Generate a detailed caption for this image."
        )

        # Execute queries on different models
        clip_response, clip_time = vlm.execute_query('CLIP', clip_prompt)
        blip_response, blip_time = vlm.execute_query('BLIP', blip_prompt)

        print(f"CLIP response: {clip_response} (Time: {clip_time:.3f}s)")
        print(f"BLIP response: {blip_response} (Time: {blip_time:.3f}s)")

        # Clean up
        for path in model_paths.values():
            os.remove(path)

    except Exception as e:
        print(f"✗ Error with multiple models: {e}")
        # Clean up dummy files
        for path in model_paths.values():
            if os.path.exists(path):
                os.remove(path)


def example_error_handling():
    """Example demonstrating error handling."""
    print("\n=== Error Handling Example ===")

    vlm = VLMInterface()

    # Test 1: No API key or model path
    try:
        vlm.configure_model('InvalidModel')
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    # Test 2: Non-existent model path
    try:
        vlm.configure_model('NonExistent', model_path='/non/existent/path')
    except FileNotFoundError as e:
        print(f"✓ Caught expected error: {e}")

    # Test 3: Query unconfigured model
    try:
        prompt = vlm.create_prompt("test", "test")
        response, exec_time = vlm.execute_query('UnconfiguredModel', prompt)
    except KeyError as e:
        print(f"✓ Caught expected error: {e}")

    # Test 4: Invalid output format
    dummy_path = '/tmp/test_model.pkl'
    try:
        # Create a dummy model first
        with open(dummy_path, 'w') as f:
            f.write("test")

        vlm.configure_model('TestModel', model_path=dummy_path)
        prompt = vlm.create_prompt("test", "test", output_format='invalid_format')
        response, exec_time = vlm.execute_query('TestModel', prompt)

    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)


def main():
    """Run all examples."""
    print("VLM Interface Usage Examples")
    print("=" * 40)

    example_api_model()
    example_local_model()
    example_multiple_models()
    example_error_handling()

    print("\n" + "=" * 40)
    print("All examples completed!")


if __name__ == "__main__":
    main()