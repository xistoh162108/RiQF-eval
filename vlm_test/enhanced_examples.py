#!/usr/bin/env python3
"""
Enhanced VLM Interface Examples

This script demonstrates the comprehensive capabilities of the enhanced VLM interface,
including support for multiple model providers, capabilities, and advanced features.
"""

import os
import tempfile
from enhanced_vlm_interface import (
    EnhancedVLMInterface,
    ModelCapability,
    ModelProvider,
    ModelFactory
)


def example_api_models():
    """Example configuring and using various API models."""
    print("=== API Models Example ===")

    vlm = EnhancedVLMInterface()

    # Configure different API models
    try:
        # OpenAI models
        vlm.configure_model('openai-gpt-4', api_key='sk-test-key')
        vlm.configure_model('openai-gpt-4-vision', api_key='sk-test-key')

        # Gemini models
        vlm.configure_model('gemini-pro', api_key='AIza-test-key')
        vlm.configure_model('gemini-pro-vision', api_key='AIza-test-key')

        # DeepSeek models
        vlm.configure_model('deepseek-chat', api_key='ds-test-key')
        vlm.configure_model('deepseek-coder', api_key='ds-test-key')

        print(f"✓ Configured API models: {vlm.get_configured_models()}")

        # Test text generation with OpenAI
        prompt = vlm.create_prompt(
            system_input="You are a helpful assistant",
            user_prompt="Explain quantum computing in simple terms",
            task_type=ModelCapability.TEXT_GENERATION,
            temperature=0.7,
            max_tokens=200
        )

        response, time_taken = vlm.execute_query('openai-gpt-4', prompt)
        print(f"OpenAI GPT-4 response: {response}")
        print(f"Execution time: {time_taken:.3f} seconds")

        # Test vision capabilities with Gemini
        vision_prompt = vlm.create_prompt(
            system_input="Analyze the provided image",
            user_prompt="Describe what you see in this image",
            task_type=ModelCapability.VISION_LANGUAGE,
            image_path="/path/to/test/image.jpg"  # Would process if image exists
        )

        try:
            response, time_taken = vlm.execute_query('gemini-pro-vision', vision_prompt)
            print(f"Gemini Vision response: {response}")
        except Exception as e:
            print(f"Note: Image processing demo: {e}")

        # Test code generation with DeepSeek
        code_prompt = vlm.create_prompt(
            system_input="You are an expert programmer",
            user_prompt="Write a Python function to calculate factorial",
            task_type=ModelCapability.CODE_GENERATION,
            temperature=0.2
        )

        response, time_taken = vlm.execute_query('deepseek-coder', code_prompt)
        print(f"DeepSeek Coder response: {response}")

    except Exception as e:
        print(f"API model example: {e}")


def example_local_models():
    """Example configuring and using various local models."""
    print("\n=== Local Models Example ===")

    vlm = EnhancedVLMInterface()

    # Create dummy model files for demonstration
    temp_dir = tempfile.mkdtemp(prefix="vlm_models_")
    model_files = {
        'llama-7b': os.path.join(temp_dir, 'llama-7b.bin'),
        'llama-13b': os.path.join(temp_dir, 'llama-13b.bin'),
        'docowl': os.path.join(temp_dir, 'docowl.pt'),
        'smoldocling': os.path.join(temp_dir, 'smoldocling.safetensors'),
        'donut': os.path.join(temp_dir, 'donut.pt'),
        'deepseek-local-vlm': os.path.join(temp_dir, 'deepseek-vlm.bin')
    }

    try:
        # Create dummy model files
        for model_path in model_files.values():
            with open(model_path, 'w') as f:
                f.write("dummy model data")

        # Configure local models
        for model_id, model_path in model_files.items():
            vlm.configure_model(model_id, model_path=model_path)

        print(f"✓ Configured local models: {vlm.get_configured_models()}")

        # Test text generation with Llama
        text_prompt = vlm.create_prompt(
            system_input="Complete the following text",
            user_prompt="The future of artificial intelligence",
            task_type=ModelCapability.TEXT_GENERATION
        )

        response, time_taken = vlm.execute_query('llama-13b', text_prompt)
        print(f"Llama 13B response: {response}")

        # Test document understanding with DocOWL
        doc_prompt = vlm.create_prompt(
            system_input="Analyze this document for key information",
            user_prompt="Extract the main topics from this document",
            task_type=ModelCapability.DOCUMENT_UNDERSTANDING,
            document_path="/path/to/document.pdf"
        )

        response, time_taken = vlm.execute_query('docowl', doc_prompt)
        print(f"DocOWL response: {response}")

        # Test vision-language with DeepSeek Local VLM
        vlm_prompt = vlm.create_prompt(
            system_input="Process this image and text together",
            user_prompt="Describe the relationship between the image and this text: 'Machine Learning'",
            task_type=ModelCapability.VISION_LANGUAGE,
            image_path="/path/to/image.jpg"
        )

        try:
            response, time_taken = vlm.execute_query('deepseek-local-vlm', vlm_prompt)
            print(f"DeepSeek Local VLM response: {response}")
        except Exception as e:
            print(f"Note: VLM demo: {e}")

    except Exception as e:
        print(f"Local model example: {e}")

    finally:
        # Cleanup
        for model_path in model_files.values():
            try:
                os.remove(model_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass


def example_model_discovery():
    """Example discovering and filtering models by capabilities."""
    print("\n=== Model Discovery Example ===")

    # Get all available models
    available_models = ModelFactory.get_available_models()
    print(f"Total available models: {len(available_models)}")

    # Find models by capability
    text_models = ModelFactory.get_models_by_capability(ModelCapability.TEXT_GENERATION)
    vision_models = ModelFactory.get_models_by_capability(ModelCapability.VISION_LANGUAGE)
    doc_models = ModelFactory.get_models_by_capability(ModelCapability.DOCUMENT_UNDERSTANDING)
    code_models = ModelFactory.get_models_by_capability(ModelCapability.CODE_GENERATION)

    print(f"Text generation models: {text_models}")
    print(f"Vision-language models: {vision_models}")
    print(f"Document understanding models: {doc_models}")
    print(f"Code generation models: {code_models}")

    # Find models by provider
    openai_models = ModelFactory.get_models_by_provider(ModelProvider.OPENAI)
    local_models = [
        model_id for model_id, config in available_models.items()
        if not config.api_endpoint
    ]

    print(f"OpenAI models: {openai_models}")
    print(f"Local models: {local_models}")


def example_batch_configuration():
    """Example batch configuring multiple models."""
    print("\n=== Batch Configuration Example ===")

    vlm = EnhancedVLMInterface()

    # Create dummy model files
    temp_dir = tempfile.mkdtemp(prefix="vlm_batch_")
    local_model_path = os.path.join(temp_dir, "test_model.bin")
    with open(local_model_path, 'w') as f:
        f.write("dummy model")

    # Batch configure models
    model_configs = {
        'openai-gpt-4': {'api_key': 'sk-test-key'},
        'gemini-pro': {'api_key': 'AIza-test-key'},
        'llama-7b': {'model_path': local_model_path},
        'docowl': {'model_path': local_model_path},
    }

    try:
        results = vlm.batch_configure_models(model_configs)
        print(f"Batch configuration results: {results}")

        # Get model information
        for model_id in vlm.get_configured_models():
            info = vlm.get_model_info(model_id)
            print(f"\nModel: {model_id}")
            print(f"  Provider: {info['provider']}")
            print(f"  Capabilities: {info['capabilities']}")
            print(f"  Supports images: {info['supports_images']}")
            print(f"  Max tokens: {info['max_tokens']}")
            print(f"  Type: {info['model_type']}")

    except Exception as e:
        print(f"Batch configuration example: {e}")

    finally:
        # Cleanup
        try:
            os.remove(local_model_path)
            os.rmdir(temp_dir)
        except:
            pass


def example_advanced_prompting():
    """Example advanced prompting with different model types."""
    print("\n=== Advanced Prompting Example ===")

    vlm = EnhancedVLMInterface()

    # Create a temporary model file
    temp_dir = tempfile.mkdtemp(prefix="vlm_advanced_")
    model_path = os.path.join(temp_dir, "model.bin")
    with open(model_path, 'w') as f:
        f.write("dummy model")

    try:
        # Configure different types of models
        vlm.configure_model('openai-gpt-4', api_key='test-key')
        vlm.configure_model('docowl', model_path=model_path)

        # Text generation prompt
        text_prompt = vlm.create_prompt(
            system_input="You are a creative writing assistant",
            user_prompt="Write a short story about AI",
            training_example=[
                "Input: Write about space -> Output: The stars twinkled...",
                "Input: Write about ocean -> Output: The waves crashed..."
            ],
            task_type=ModelCapability.TEXT_GENERATION,
            temperature=0.8,
            max_tokens=500,
            output_format='text'
        )

        response, time_taken = vlm.execute_query('openai-gpt-4', text_prompt)
        print(f"Creative writing response: {response}")

        # Document understanding prompt
        doc_prompt = vlm.create_prompt(
            system_input="Extract and analyze document content",
            user_prompt="Identify key sections and summarize main points",
            task_type=ModelCapability.DOCUMENT_UNDERSTANDING,
            document_path="/path/to/document.pdf",
            output_format='json'
        )

        response, time_taken = vlm.execute_query('docowl', doc_prompt)
        print(f"Document analysis response: {response}")

    except Exception as e:
        print(f"Advanced prompting example: {e}")

    finally:
        # Cleanup
        try:
            os.remove(model_path)
            os.rmdir(temp_dir)
        except:
            pass


def example_error_handling():
    """Example comprehensive error handling."""
    print("\n=== Error Handling Example ===")

    vlm = EnhancedVLMInterface()

    # Test 1: Unsupported model
    try:
        vlm.configure_model('unsupported-model', api_key='test')
    except ValueError as e:
        print(f"✓ Caught unsupported model error: {e}")

    # Test 2: Missing API key
    try:
        vlm.configure_model('openai-gpt-4')  # No API key
    except ValueError as e:
        print(f"✓ Caught missing API key error: {e}")

    # Test 3: Missing model path
    try:
        vlm.configure_model('llama-7b')  # No model path
    except ValueError as e:
        print(f"✓ Caught missing model path error: {e}")

    # Test 4: Invalid model path
    try:
        vlm.configure_model('llama-7b', model_path='/non/existent/path')
    except FileNotFoundError as e:
        print(f"✓ Caught invalid path error: {e}")

    # Test 5: Capability mismatch
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "test.bin")
    with open(model_path, 'w') as f:
        f.write("test")

    try:
        vlm.configure_model('llama-7b', model_path=model_path)  # Text-only model

        # Try to use images with text-only model
        prompt = vlm.create_prompt(
            system_input="Test",
            user_prompt="Test",
            image_path="/some/image.jpg"
        )

        vlm.execute_query('llama-7b', prompt)
    except ValueError as e:
        print(f"✓ Caught capability mismatch error: {e}")

    finally:
        try:
            os.remove(model_path)
            os.rmdir(temp_dir)
        except:
            pass


def example_model_comparison():
    """Example comparing different models for the same task."""
    print("\n=== Model Comparison Example ===")

    vlm = EnhancedVLMInterface()

    # Find all models that support text generation
    text_generation_models = vlm.find_models_for_task(ModelCapability.TEXT_GENERATION)
    print(f"Models supporting text generation: {text_generation_models}")

    # Configure a few models for comparison
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.bin")
    with open(model_path, 'w') as f:
        f.write("dummy")

    try:
        vlm.configure_model('openai-gpt-4', api_key='test-key')
        vlm.configure_model('gemini-pro', api_key='test-key')
        vlm.configure_model('llama-13b', model_path=model_path)

        # Same prompt for all models
        prompt = vlm.create_prompt(
            system_input="Explain concepts clearly",
            user_prompt="What is machine learning?",
            task_type=ModelCapability.TEXT_GENERATION,
            max_tokens=100
        )

        # Compare responses and timing
        models_to_test = ['openai-gpt-4', 'gemini-pro', 'llama-13b']
        results = {}

        for model_id in models_to_test:
            try:
                response, exec_time = vlm.execute_query(model_id, prompt)
                results[model_id] = {
                    'response': response,
                    'time': exec_time,
                    'info': vlm.get_model_info(model_id)
                }
            except Exception as e:
                results[model_id] = {'error': str(e)}

        # Display comparison results
        print("\nModel Comparison Results:")
        for model_id, result in results.items():
            print(f"\n{model_id}:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Response: {result['response'][:100]}...")
                print(f"  Time: {result['time']:.3f}s")
                print(f"  Provider: {result['info']['provider']}")

    finally:
        try:
            os.remove(model_path)
            os.rmdir(temp_dir)
        except:
            pass


def main():
    """Run all enhanced VLM interface examples."""
    print("Enhanced VLM Interface Examples")
    print("=" * 50)

    example_api_models()
    example_local_models()
    example_model_discovery()
    example_batch_configuration()
    example_advanced_prompting()
    example_error_handling()
    example_model_comparison()

    print("\n" + "=" * 50)
    print("All enhanced examples completed!")
    print("\nKey features demonstrated:")
    print("• Multiple API providers (OpenAI, Gemini, DeepSeek)")
    print("• Local model support (Llama, DocOWL, SmolDocLing, Donut)")
    print("• Capability-based model discovery")
    print("• Advanced prompting with task types")
    print("• Batch model configuration")
    print("• Comprehensive error handling")
    print("• Performance comparison")
    print("• Dynamic prompt optimization")


if __name__ == "__main__":
    main()