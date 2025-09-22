# Enhanced VLM Interface

A comprehensive Python interface for querying multiple Vision-Language Models (VLMs), supporting diverse API providers and local models with advanced capabilities and intelligent routing.

## 🚀 Features

### Multi-Provider Support
- **API Models**: OpenAI (GPT-4, GPT-4V), Google Gemini (Pro, Pro Vision), DeepSeek (Chat, Coder)
- **Local Models**: Llama (7B, 13B), DocOWL, SmolDocLing, Donut, DeepSeek Local VLM

### Advanced Capabilities
- **🎯 Capability-Based Routing**: Automatic model selection based on task requirements
- **🔄 Dynamic Prompt Optimization**: Model-specific prompt formatting and optimization
- **⚡ Batch Operations**: Configure and manage multiple models simultaneously
- **📊 Performance Tracking**: Built-in execution time monitoring and logging
- **🖼️ Multi-Modal Processing**: Image and document support with capability validation
- **🛡️ Robust Error Handling**: Comprehensive validation and graceful error recovery

## 📋 Supported Models

### API Models (Cloud-based)

| Provider | Model ID | Capabilities | Max Tokens | Image Support |
|----------|----------|--------------|------------|---------------|
| **OpenAI** | `openai-gpt-4` | Text, Code Generation | 8,192 | ❌ |
| | `openai-gpt-4-vision` | Vision-Language, Text | 4,096 | ✅ |
| | `openai-gpt-3.5-turbo` | Text Generation | 4,096 | ❌ |
| **Gemini** | `gemini-pro` | Text, Code Generation | 8,192 | ❌ |
| | `gemini-pro-vision` | Vision-Language, Text | 4,096 | ✅ |
| **DeepSeek** | `deepseek-chat` | Text, Code Generation | 4,096 | ❌ |
| | `deepseek-coder` | Code, Text Generation | 8,192 | ❌ |

### Local Models (On-device)

| Provider | Model ID | Capabilities | Max Tokens | Special Features |
|----------|----------|--------------|------------|------------------|
| **Llama** | `llama-7b` | Text Generation | 2,048 | Fast inference |
| | `llama-13b` | Text, Code Generation | 4,096 | Better performance |
| **DocOWL** | `docowl` | Document Understanding, Vision | 2,048 | Document analysis |
| **SmolDocLing** | `smoldocling` | Document Understanding | 1,024 | Lightweight docs |
| **Donut** | `donut` | Document Understanding, Vision | 1,024 | OCR + understanding |
| **DeepSeek** | `deepseek-local-vlm` | Vision-Language, Text | 4,096 | Local VLM |

## 🏁 Quick Start

### Basic Setup

```python
from enhanced_vlm_interface import (
    EnhancedVLMInterface,
    ModelCapability,
    ModelProvider
)

# Initialize the interface
vlm = EnhancedVLMInterface()

# Configure API models
vlm.configure_model('openai-gpt-4', api_key='your-openai-key')
vlm.configure_model('gemini-pro-vision', api_key='your-gemini-key')

# Configure local models
vlm.configure_model('llama-13b', model_path='/path/to/llama-13b.bin')
vlm.configure_model('docowl', model_path='/path/to/docowl.pt')
```

### Advanced Usage

```python
# Create capability-specific prompts
text_prompt = vlm.create_prompt(
    system_input="You are a helpful assistant",
    user_prompt="Explain quantum computing",
    task_type=ModelCapability.TEXT_GENERATION,
    temperature=0.7,
    max_tokens=500
)

# Vision-language task
vision_prompt = vlm.create_prompt(
    system_input="Analyze visual content",
    user_prompt="Describe this image in detail",
    image_path="/path/to/image.jpg",
    task_type=ModelCapability.VISION_LANGUAGE,
    output_format="json"
)

# Document understanding task
doc_prompt = vlm.create_prompt(
    system_input="Extract key information from documents",
    user_prompt="Summarize the main topics",
    document_path="/path/to/document.pdf",
    task_type=ModelCapability.DOCUMENT_UNDERSTANDING
)

# Execute with automatic optimization
response, exec_time = vlm.execute_query('openai-gpt-4', text_prompt)
vision_response, vision_time = vlm.execute_query('gemini-pro-vision', vision_prompt)
doc_response, doc_time = vlm.execute_query('docowl', doc_prompt)
```

## 🔧 Model Discovery & Management

### Find Models by Capability

```python
# Discover models for specific tasks
text_models = vlm.find_models_for_task(ModelCapability.TEXT_GENERATION)
vision_models = vlm.find_models_for_task(ModelCapability.VISION_LANGUAGE)
doc_models = vlm.find_models_for_task(ModelCapability.DOCUMENT_UNDERSTANDING)

print(f"Text generation models: {text_models}")
print(f"Vision-language models: {vision_models}")
print(f"Document models: {doc_models}")
```

### Batch Configuration

```python
# Configure multiple models at once
model_configs = {
    'openai-gpt-4': {'api_key': 'your-openai-key'},
    'gemini-pro': {'api_key': 'your-gemini-key'},
    'llama-13b': {'model_path': '/path/to/llama.bin'},
    'docowl': {'model_path': '/path/to/docowl.pt'}
}

results = vlm.batch_configure_models(model_configs)
print(f"Configuration results: {results}")

# Get detailed model information
for model_id in vlm.get_configured_models():
    info = vlm.get_model_info(model_id)
    print(f"{model_id}: {info['capabilities']}")
```

## 🎯 Capability System

The interface uses a sophisticated capability system to ensure models are used appropriately:

```python
from enhanced_vlm_interface import ModelCapability

# Available capabilities
ModelCapability.TEXT_GENERATION      # General text generation
ModelCapability.VISION_LANGUAGE      # Image + text processing
ModelCapability.DOCUMENT_UNDERSTANDING  # Document analysis
ModelCapability.CODE_GENERATION      # Code generation and analysis
```

### Automatic Validation

```python
# The interface automatically validates compatibility
try:
    # This will fail - Llama doesn't support images
    image_prompt = vlm.create_prompt(
        system_input="Analyze this image",
        user_prompt="What do you see?",
        image_path="/path/to/image.jpg"
    )
    vlm.execute_query('llama-7b', image_prompt)
except ValueError as e:
    print(f"Validation error: {e}")
    # "Model llama-7b does not support image inputs"
```

## 🔄 Dynamic Prompt Optimization

The interface automatically optimizes prompts based on the target model:

```python
# Base prompt
prompt = vlm.create_prompt(
    system_input="Complete this task",
    user_prompt="Write a poem about AI"
)

# Automatically optimized for different providers:
# OpenAI: "You are a helpful assistant. Complete this task"
# Gemini: "[System Context] Complete this task"
# DeepSeek: "<system>Complete this task</system>"
# DocOWL: "Document Analysis Task: Complete this task"
```

## 📊 Performance Monitoring

```python
# Built-in performance tracking
response, exec_time = vlm.execute_query('openai-gpt-4', prompt)
print(f"Response generated in {exec_time:.3f} seconds")

# Compare model performance
models_to_test = ['openai-gpt-4', 'gemini-pro', 'llama-13b']
results = {}

for model_id in models_to_test:
    response, time_taken = vlm.execute_query(model_id, prompt)
    results[model_id] = {
        'response': response,
        'time': time_taken,
        'tokens_per_second': len(response.split()) / time_taken
    }
```

## 🛡️ Error Handling

The interface provides comprehensive error handling and logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

try:
    # Unsupported model
    vlm.configure_model('unknown-model')
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    # Missing API key
    vlm.configure_model('openai-gpt-4')  # No API key provided
except ValueError as e:
    print(f"Authentication error: {e}")

try:
    # Capability mismatch
    vlm.execute_query('text-only-model', vision_prompt)
except ValueError as e:
    print(f"Capability error: {e}")
```

## 🏗️ Architecture

```
EnhancedVLMInterface
├── ModelFactory (Model Creation & Discovery)
├── EnhancedModelBase (Abstract Base)
│   ├── EnhancedAPIModel (Cloud Models)
│   └── EnhancedLocalModel (Local Models)
├── EnhancedPrompt (Structured Prompts)
├── ModelCapability (Capability Enum)
└── ModelProvider (Provider Enum)
```

## 📝 Configuration Examples

### OpenAI Models

```python
# GPT-4 for text generation
vlm.configure_model('openai-gpt-4', api_key='sk-your-key-here')

# GPT-4 Vision for image analysis
vlm.configure_model('openai-gpt-4-vision', api_key='sk-your-key-here')
```

### Gemini Models

```python
# Gemini Pro for advanced text tasks
vlm.configure_model('gemini-pro', api_key='AIza-your-key-here')

# Gemini Pro Vision for multimodal tasks
vlm.configure_model('gemini-pro-vision', api_key='AIza-your-key-here')
```

### DeepSeek Models

```python
# DeepSeek Chat for general conversation
vlm.configure_model('deepseek-chat', api_key='ds-your-key-here')

# DeepSeek Coder for programming tasks
vlm.configure_model('deepseek-coder', api_key='ds-your-key-here')
```

### Local Models

```python
# Llama models for local text generation
vlm.configure_model('llama-7b', model_path='/models/llama-7b.bin')
vlm.configure_model('llama-13b', model_path='/models/llama-13b.bin')

# Specialized document models
vlm.configure_model('docowl', model_path='/models/docowl.pt')
vlm.configure_model('smoldocling', model_path='/models/smoldocling.safetensors')
vlm.configure_model('donut', model_path='/models/donut.pt')

# Local vision-language model
vlm.configure_model('deepseek-local-vlm', model_path='/models/deepseek-vlm.bin')
```

## 🧪 Testing

```bash
# Run comprehensive tests
python test_enhanced_vlm.py

# Run examples
python enhanced_examples.py
```

## 📚 Examples

The repository includes comprehensive examples:

- **`enhanced_examples.py`**: Full feature demonstrations
- **`test_enhanced_vlm.py`**: Comprehensive test suite
- **API model integration**: OpenAI, Gemini, DeepSeek
- **Local model setup**: Llama, DocOWL, Donut, etc.
- **Multi-modal processing**: Images and documents
- **Performance comparison**: Benchmarking different models

## 🔧 Requirements

### Core Requirements
- Python 3.7+
- Built-in logging and error handling
- No external dependencies for core functionality

### Optional Dependencies

```bash
# For image processing
pip install Pillow

# For enhanced logging
pip install colorlog

# For model-specific requirements (install as needed)
pip install torch transformers  # For local models
pip install openai  # For OpenAI API
pip install google-generativeai  # For Gemini API
```

## 🚀 Advanced Features

### Custom Model Integration

```python
# Extend for custom models
from enhanced_vlm_interface import EnhancedLocalModel, ModelConfig, ModelProvider

# Define custom model configuration
custom_config = ModelConfig(
    provider=ModelProvider.CUSTOM,
    model_name="my-custom-model",
    capabilities=[ModelCapability.TEXT_GENERATION],
    max_tokens=4096
)

# Implement custom model class
class CustomModel(EnhancedLocalModel):
    def generate_response(self, prompt, images=None, **kwargs):
        # Custom implementation
        return "Custom model response"
```

### Integration with Existing Workflows

```python
# Easy integration with existing code
def process_with_best_model(task_type, prompt_text, **kwargs):
    """Automatically select and use the best model for a task."""

    # Find suitable models
    suitable_models = vlm.find_models_for_task(task_type)

    # Use the first available configured model
    for model_id in suitable_models:
        if model_id in vlm.get_configured_models():
            prompt = vlm.create_prompt(
                system_input=f"Perform {task_type.value} task",
                user_prompt=prompt_text,
                task_type=task_type,
                **kwargs
            )
            return vlm.execute_query(model_id, prompt)

    raise ValueError(f"No configured models found for {task_type}")

# Usage
response, time_taken = process_with_best_model(
    ModelCapability.CODE_GENERATION,
    "Write a Python function to sort a list"
)
```

## 📄 License

This enhanced VLM interface is part of the HyperX Parser VLM Test suite.

## 🤝 Contributing

The interface is designed to be easily extensible. To add new models:

1. Add model configuration to `MODEL_CONFIGS`
2. Implement provider-specific optimizations
3. Add comprehensive tests
4. Update documentation

---

**Built for flexibility, performance, and ease of use. Supporting the full spectrum of modern VLM capabilities.**