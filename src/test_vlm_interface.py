#!/usr/bin/env python3
"""
Basic tests for the VLM Interface

Simple test cases to verify the VLM interface functionality including
image processing and time tracking features.
"""

import os
import tempfile
from vlm_interface import VLMInterface, Prompt, process_images


def test_prompt_creation():
    """Test prompt creation and validation."""
    print("Testing prompt creation...")

    vlm = VLMInterface()

    # Test basic prompt
    prompt = vlm.create_prompt(
        system_input="Test system input",
        user_prompt="Test user prompt"
    )

    assert prompt.system_input == "Test system input"
    assert prompt.user_prompt == "Test user prompt"
    assert prompt.output_format == "text"
    assert prompt.training_example is None

    # Test prompt with training examples and image path
    prompt_with_examples = vlm.create_prompt(
        system_input="Test system",
        user_prompt="Test prompt",
        training_example=["Example 1", "Example 2"],
        output_format="json",
        image_path="/test/path/image.jpg"
    )

    assert prompt_with_examples.training_example is not None
    assert len(prompt_with_examples.training_example) == 2
    assert prompt_with_examples.output_format == "json"
    assert prompt_with_examples.image_path == "/test/path/image.jpg"

    print("✓ Prompt creation tests passed")


def test_api_model_configuration():
    """Test API model configuration."""
    print("Testing API model configuration...")

    vlm = VLMInterface()

    # Test valid API model configuration
    model = vlm.configure_model("TestAPI", api_key="test-key")
    assert model.model_name == "TestAPI"
    assert model.is_available()

    # Test invalid configuration (no api_key or model_path)
    try:
        vlm.configure_model("InvalidModel")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ API model configuration tests passed")


def test_local_model_configuration():
    """Test local model configuration."""
    print("Testing local model configuration...")

    vlm = VLMInterface()

    # Create temporary model file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("dummy model data")
        temp_path = temp_file.name

    try:
        # Test valid local model configuration
        model = vlm.configure_model("TestLocal", model_path=temp_path)
        assert model.model_name == "TestLocal"
        assert model.is_available()

        # Test invalid model path
        try:
            vlm.configure_model("InvalidLocal", model_path="/non/existent/path")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    finally:
        # Clean up
        os.unlink(temp_path)

    print("✓ Local model configuration tests passed")


def test_query_execution():
    """Test query execution."""
    print("Testing query execution...")

    vlm = VLMInterface()

    # Configure a test model
    model = vlm.configure_model("TestModel", api_key="test-key")

    # Create test prompt
    prompt = vlm.create_prompt(
        system_input="Test system",
        user_prompt="Test query",
        output_format="text"
    )

    # Execute query with time tracking
    response, execution_time = vlm.execute_query("TestModel", prompt)
    assert isinstance(response, str)
    assert "Mock API response" in response
    assert isinstance(execution_time, float)
    assert execution_time >= 0

    # Test JSON output
    json_prompt = vlm.create_prompt(
        system_input="Test system",
        user_prompt="Test query",
        output_format="json"
    )

    json_response, json_time = vlm.execute_query("TestModel", json_prompt)
    assert isinstance(json_response, dict)
    assert "response" in json_response
    assert isinstance(json_time, float)
    assert json_time >= 0

    print("✓ Query execution tests passed")


def test_model_management():
    """Test model management features."""
    print("Testing model management...")

    vlm = VLMInterface()

    # Initially no models
    assert len(vlm.get_configured_models()) == 0

    # Add models
    vlm.configure_model("Model1", api_key="key1")
    vlm.configure_model("Model2", api_key="key2")

    assert len(vlm.get_configured_models()) == 2
    assert "Model1" in vlm.get_configured_models()
    assert "Model2" in vlm.get_configured_models()

    # Remove model
    vlm.remove_model("Model1")
    assert len(vlm.get_configured_models()) == 1
    assert "Model1" not in vlm.get_configured_models()

    print("✓ Model management tests passed")


def test_image_processing():
    """Test image processing functionality."""
    print("Testing image processing...")

    # Test with no image path
    images = process_images(None)
    assert len(images) == 0

    # Test with non-existent path
    try:
        process_images("/non/existent/path")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Test error handling for unsupported format
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write("not an image")
    temp_file.close()

    try:
        process_images(temp_file.name)
        assert False, "Should have raised ValueError for unsupported format"
    except (ValueError, ImportError):
        # ValueError for unsupported format, ImportError if PIL not available
        pass
    finally:
        os.unlink(temp_file.name)

    print("✓ Image processing tests passed")


def test_time_tracking():
    """Test execution time tracking."""
    print("Testing time tracking...")

    vlm = VLMInterface()
    vlm.configure_model("TimeTestModel", api_key="test-key")

    prompt = vlm.create_prompt(
        system_input="Time test",
        user_prompt="Test timing"
    )

    # Execute and verify time tracking
    response, execution_time = vlm.execute_query("TimeTestModel", prompt)

    assert isinstance(execution_time, float)
    assert execution_time >= 0
    assert execution_time < 1.0  # Should be very fast for mock response

    print("✓ Time tracking tests passed")


def run_all_tests():
    """Run all test functions."""
    print("Running VLM Interface Tests")
    print("=" * 30)

    try:
        test_prompt_creation()
        test_api_model_configuration()
        test_local_model_configuration()
        test_query_execution()
        test_model_management()
        test_image_processing()
        test_time_tracking()

        print("=" * 30)
        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)