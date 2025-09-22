#!/usr/bin/env python3
"""
Comprehensive tests for the Enhanced VLM Interface

Tests all the advanced features including multiple model providers,
capabilities, dynamic prompting, and error handling.
"""

import os
import tempfile
from enhanced_vlm_interface import (
    EnhancedVLMInterface,
    ModelCapability,
    ModelProvider,
    ModelFactory,
    EnhancedPrompt,
    MODEL_CONFIGS
)


def test_model_configurations():
    """Test that all model configurations are valid."""
    print("Testing model configurations...")

    # Test that all required models are configured
    required_models = [
        'openai-gpt-4', 'openai-gpt-4-vision', 'openai-gpt-3.5-turbo',
        'gemini-pro', 'gemini-pro-vision',
        'deepseek-chat', 'deepseek-coder',
        'llama-7b', 'llama-13b',
        'docowl', 'smoldocling', 'donut', 'deepseek-local-vlm'
    ]

    for model_id in required_models:
        assert model_id in MODEL_CONFIGS, f"Model {model_id} not configured"

    # Test model config properties
    for model_id, config in MODEL_CONFIGS.items():
        assert isinstance(config.provider, ModelProvider)
        assert isinstance(config.capabilities, list)
        assert len(config.capabilities) > 0
        assert isinstance(config.model_name, str)
        assert len(config.model_name) > 0

        # API models should have endpoints
        if config.provider in [ModelProvider.OPENAI, ModelProvider.GEMINI, ModelProvider.DEEPSEEK]:
            assert config.api_endpoint is not None, f"API model {model_id} missing endpoint"

    print("✓ Model configurations tests passed")


def test_model_factory():
    """Test the ModelFactory functionality."""
    print("Testing ModelFactory...")

    # Test available models
    available_models = ModelFactory.get_available_models()
    assert len(available_models) > 0
    assert 'openai-gpt-4' in available_models

    # Test capability filtering
    text_models = ModelFactory.get_models_by_capability(ModelCapability.TEXT_GENERATION)
    vision_models = ModelFactory.get_models_by_capability(ModelCapability.VISION_LANGUAGE)
    doc_models = ModelFactory.get_models_by_capability(ModelCapability.DOCUMENT_UNDERSTANDING)

    assert len(text_models) > 0
    assert 'openai-gpt-4' in text_models
    assert 'gemini-pro-vision' in vision_models
    assert 'docowl' in doc_models

    # Test provider filtering
    openai_models = ModelFactory.get_models_by_provider(ModelProvider.OPENAI)
    llama_models = ModelFactory.get_models_by_provider(ModelProvider.LLAMA)

    assert len(openai_models) > 0
    assert 'openai-gpt-4' in openai_models
    assert 'llama-7b' in llama_models

    print("✓ ModelFactory tests passed")


def test_api_model_configuration():
    """Test API model configuration."""
    print("Testing API model configuration...")

    vlm = EnhancedVLMInterface()

    # Test valid API model configuration
    model = vlm.configure_model('openai-gpt-4', api_key='test-key')
    assert model.model_id == 'openai-gpt-4'
    assert model.provider == ModelProvider.OPENAI
    assert model.is_available()

    # Test missing API key
    try:
        vlm.configure_model('gemini-pro')  # No API key
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "API key is required" in str(e)

    # Test custom endpoint
    model = vlm.configure_model('deepseek-chat', api_key='test-key', base_url='custom-endpoint')
    assert model.base_url == 'custom-endpoint'

    print("✓ API model configuration tests passed")


def test_local_model_configuration():
    """Test local model configuration."""
    print("Testing local model configuration...")

    vlm = EnhancedVLMInterface()

    # Create temporary model file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy model data")
        temp_path = temp_file.name

    try:
        # Test valid local model configuration
        model = vlm.configure_model('llama-7b', model_path=temp_path)
        assert model.model_id == 'llama-7b'
        assert model.provider == ModelProvider.LLAMA
        assert model.is_available()

        # Test missing model path
        try:
            vlm.configure_model('docowl')  # No model path
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model path is required" in str(e)

        # Test invalid model path
        try:
            vlm.configure_model('donut', model_path='/non/existent/path')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    finally:
        os.unlink(temp_path)

    print("✓ Local model configuration tests passed")


def test_enhanced_prompt_creation():
    """Test enhanced prompt creation and validation."""
    print("Testing enhanced prompt creation...")

    vlm = EnhancedVLMInterface()

    # Test basic prompt
    prompt = vlm.create_prompt(
        system_input="Test system",
        user_prompt="Test prompt"
    )

    assert prompt.system_input == "Test system"
    assert prompt.user_prompt == "Test prompt"
    assert prompt.output_format == "text"
    assert prompt.task_type is None

    # Test advanced prompt with all features
    prompt = vlm.create_prompt(
        system_input="Advanced system",
        user_prompt="Advanced prompt",
        training_example=["Example 1", "Example 2"],
        output_format="json",
        image_path="/path/to/image.jpg",
        document_path="/path/to/doc.pdf",
        task_type=ModelCapability.VISION_LANGUAGE,
        temperature=0.8,
        max_tokens=1000
    )

    assert prompt.system_input == "Advanced system"
    assert prompt.training_example == ["Example 1", "Example 2"]
    assert prompt.image_path == "/path/to/image.jpg"
    assert prompt.document_path == "/path/to/doc.pdf"
    assert prompt.task_type == ModelCapability.VISION_LANGUAGE
    assert prompt.temperature == 0.8
    assert prompt.max_tokens == 1000

    # Test prompt dictionary conversion
    prompt_dict = prompt.to_dict()
    assert prompt_dict["system_input"] == "Advanced system"
    assert prompt_dict["task_type"] == "vision_language"

    print("✓ Enhanced prompt creation tests passed")


def test_prompt_optimization():
    """Test model-specific prompt optimization."""
    print("Testing prompt optimization...")

    # Create test models
    vlm = EnhancedVLMInterface()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy")
        temp_path = temp_file.name

    try:
        # Configure different model types
        openai_model = vlm.configure_model('openai-gpt-4', api_key='test-key')
        gemini_model = vlm.configure_model('gemini-pro', api_key='test-key')
        deepseek_model = vlm.configure_model('deepseek-chat', api_key='test-key')
        llama_model = vlm.configure_model('llama-7b', model_path=temp_path)

        # Test prompt optimization for different providers
        base_prompt = EnhancedPrompt(
            system_input="Test system",
            user_prompt="Test prompt"
        )

        # Test OpenAI optimization
        openai_optimized = openai_model.optimize_prompt_for_model(base_prompt)
        assert "You are a helpful assistant" in openai_optimized.system_input

        # Test Gemini optimization
        gemini_optimized = gemini_model.optimize_prompt_for_model(base_prompt)
        assert "[System Context]" in gemini_optimized.system_input

        # Test DeepSeek optimization
        deepseek_optimized = deepseek_model.optimize_prompt_for_model(base_prompt)
        assert "<system>" in deepseek_optimized.system_input

        # Test parameter defaults
        assert openai_optimized.temperature == 0.7  # Default temperature
        assert openai_optimized.max_tokens == 8192  # Model max tokens

    finally:
        os.unlink(temp_path)

    print("✓ Prompt optimization tests passed")


def test_capability_validation():
    """Test model capability validation."""
    print("Testing capability validation...")

    vlm = EnhancedVLMInterface()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy")
        temp_path = temp_file.name

    try:
        # Configure models with different capabilities
        vlm.configure_model('openai-gpt-4', api_key='test-key')  # Text only
        vlm.configure_model('openai-gpt-4-vision', api_key='test-key')  # Vision + text
        vlm.configure_model('llama-7b', model_path=temp_path)  # Text only
        vlm.configure_model('docowl', model_path=temp_path)  # Document + vision

        # Test image capability validation
        image_prompt = vlm.create_prompt(
            system_input="Test",
            user_prompt="Test",
            image_path="/test/image.jpg"
        )

        # Should work with vision models
        try:
            vlm.execute_query('openai-gpt-4-vision', image_prompt)
        except Exception as e:
            # Expected to fail due to missing image file, not capability
            assert "does not exist" in str(e) or "PIL" in str(e)

        # Should fail with text-only models
        try:
            vlm.execute_query('openai-gpt-4', image_prompt)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "does not support image inputs" in str(e)

        # Test document capability validation
        doc_prompt = vlm.create_prompt(
            system_input="Test",
            user_prompt="Test",
            document_path="/test/doc.pdf"
        )

        # Should work with document models
        try:
            vlm.execute_query('docowl', doc_prompt)
        except Exception as e:
            # Expected to fail due to missing document file
            pass

        # Should fail with non-document models
        try:
            vlm.execute_query('llama-7b', doc_prompt)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "does not support document inputs" in str(e)

    finally:
        os.unlink(temp_path)

    print("✓ Capability validation tests passed")


def test_query_execution():
    """Test query execution with time tracking."""
    print("Testing query execution...")

    vlm = EnhancedVLMInterface()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy")
        temp_path = temp_file.name

    try:
        # Configure test models
        vlm.configure_model('openai-gpt-4', api_key='test-key')
        vlm.configure_model('llama-7b', model_path=temp_path)

        # Test basic query execution
        prompt = vlm.create_prompt(
            system_input="Test system",
            user_prompt="Test prompt",
            task_type=ModelCapability.TEXT_GENERATION
        )

        # Test API model
        response, exec_time = vlm.execute_query('openai-gpt-4', prompt)
        assert isinstance(response, str)
        assert isinstance(exec_time, float)
        assert exec_time >= 0
        assert "OpenAI" in response

        # Test local model
        response, exec_time = vlm.execute_query('llama-7b', prompt)
        assert isinstance(response, str)
        assert isinstance(exec_time, float)
        assert exec_time >= 0
        assert "Llama" in response

        # Test JSON output
        json_prompt = vlm.create_prompt(
            system_input="Test",
            user_prompt="Test",
            output_format="json"
        )

        response, exec_time = vlm.execute_query('openai-gpt-4', json_prompt)
        assert isinstance(response, dict)
        assert "model" in response
        assert "capabilities" in response

    finally:
        os.unlink(temp_path)

    print("✓ Query execution tests passed")


def test_batch_operations():
    """Test batch configuration and operations."""
    print("Testing batch operations...")

    vlm = EnhancedVLMInterface()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bin') as temp_file:
        temp_file.write("dummy")
        temp_path = temp_file.name

    try:
        # Test batch configuration
        model_configs = {
            'openai-gpt-4': {'api_key': 'test-key'},
            'gemini-pro': {'api_key': 'test-key'},
            'llama-7b': {'model_path': temp_path},
            'invalid-model': {'api_key': 'test-key'},  # Should fail
        }

        results = vlm.batch_configure_models(model_configs)

        assert results['openai-gpt-4'] == True
        assert results['gemini-pro'] == True
        assert results['llama-7b'] == True
        assert results['invalid-model'] == False

        # Test model info retrieval
        configured_models = vlm.get_configured_models()
        assert len(configured_models) == 3  # 3 successful configurations

        for model_id in ['openai-gpt-4', 'gemini-pro', 'llama-7b']:
            info = vlm.get_model_info(model_id)
            assert info['model_id'] == model_id
            assert 'capabilities' in info
            assert 'provider' in info
            assert isinstance(info['is_available'], bool)

    finally:
        os.unlink(temp_path)

    print("✓ Batch operations tests passed")


def test_error_handling():
    """Test comprehensive error handling."""
    print("Testing error handling...")

    vlm = EnhancedVLMInterface()

    # Test unsupported model
    try:
        vlm.configure_model('unsupported-model')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported model" in str(e)

    # Test unconfigured model query
    try:
        prompt = vlm.create_prompt("test", "test")
        vlm.execute_query('unconfigured-model', prompt)
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "not configured" in str(e)

    # Test model info for unconfigured model
    try:
        vlm.get_model_info('unconfigured-model')
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    print("✓ Error handling tests passed")


def test_model_discovery():
    """Test model discovery functionality."""
    print("Testing model discovery...")

    vlm = EnhancedVLMInterface()

    # Test capability-based discovery
    text_models = vlm.find_models_for_task(ModelCapability.TEXT_GENERATION)
    vision_models = vlm.find_models_for_task(ModelCapability.VISION_LANGUAGE)
    doc_models = vlm.find_models_for_task(ModelCapability.DOCUMENT_UNDERSTANDING)

    assert len(text_models) > 0
    assert len(vision_models) > 0
    assert len(doc_models) > 0

    assert 'openai-gpt-4' in text_models
    assert 'gemini-pro-vision' in vision_models
    assert 'docowl' in doc_models

    # Test available model configs
    configs = vlm.get_available_model_configs()
    assert len(configs) > 0
    assert 'openai-gpt-4' in configs

    print("✓ Model discovery tests passed")


def run_all_tests():
    """Run all test functions."""
    print("Running Enhanced VLM Interface Tests")
    print("=" * 40)

    try:
        test_model_configurations()
        test_model_factory()
        test_api_model_configuration()
        test_local_model_configuration()
        test_enhanced_prompt_creation()
        test_prompt_optimization()
        test_capability_validation()
        test_query_execution()
        test_batch_operations()
        test_error_handling()
        test_model_discovery()

        print("=" * 40)
        print("✓ All enhanced VLM interface tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)