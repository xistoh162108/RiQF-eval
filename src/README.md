# VLM Interface

A unified Python interface for querying Vision-Language Models (VLMs), supporting both API-based closed models and on-device local models with enhanced image processing and performance tracking capabilities.

## Features

- **Unified Interface**: Single API for both cloud-based and local VLM models
- **Image Processing**: Support for single image files and batch folder processing
- **Execution Time Tracking**: Automatic performance monitoring for all queries
- **Flexible Prompting**: Structured prompt system with support for system inputs, training examples, and output formats
- **Error Handling**: Comprehensive validation and error handling
- **Type Safety**: Full type annotations and runtime validation
- **Extensible**: Abstract base class architecture for easy model integration

## Quick Start

### Basic Usage

```python
from vlm_interface import VLMInterface

# Initialize the interface
vlm = VLMInterface()

# Configure an API-based model
vlm.configure_model('GPT4V', api_key='your-api-key')

# Configure a local model
vlm.configure_model('CLIP_Local', model_path='/path/to/model.pkl')

# Create a prompt with image support
prompt = vlm.create_prompt(
    system_input="Process this image and text input",
    user_prompt="Describe what you see in the image",
    image_path="/path/to/image.jpg",  # Single image or folder path
    output_format="text"
)

# Execute query with time tracking
response, execution_time = vlm.execute_query('GPT4V', prompt)
print(f"Response: {response}")
print(f"Execution time: {execution_time:.3f} seconds")
```

### Advanced Usage

```python
# Process multiple images from a folder
prompt = vlm.create_prompt(
    system_input="Batch image processing system",
    training_example=[
        "Image: cat.jpg -> A fluffy orange cat sitting on a windowsill",
        "Image: dog.jpg -> A golden retriever playing in the park"
    ],
    user_prompt="Generate captions for all images",
    image_path="/path/to/image/folder/",  # Process entire folder
    output_format="json"
)

# Execute with additional parameters and get timing
response, exec_time = vlm.execute_query('CLIP_Local', prompt, temperature=0.7)
print(f"Processed {response['images_count']} images in {exec_time:.3f} seconds")

# Direct image processing
from vlm_interface import process_images
images = process_images("/path/to/images/")
print(f"Loaded {len(images)} images")
```

## API Reference

### Classes

#### `VLMInterface`
Main interface class for managing VLM models.

**Methods:**
- `configure_model(model_name, api_key=None, model_path=None)` - Configure a model
- `create_prompt(system_input, user_prompt, training_example=None, output_format='text', image_path=None)` - Create structured prompt
- `execute_query(model_name, prompt, **kwargs)` - Execute model query, returns (response, execution_time)
- `get_configured_models()` - List configured models
- `remove_model(model_name)` - Remove a model

#### `Prompt`
Data class for structured prompts.

**Attributes:**
- `system_input: str` - System context
- `user_prompt: str` - User query
- `training_example: Optional[Union[str, List[str]]]` - Training examples
- `output_format: str` - Output format ('text' or 'json')
- `image_path: Optional[str]` - Path to image file or folder

#### `ModelBase`
Abstract base class for model implementations.

#### `APIModel`
Implementation for API-based models.

#### `LocalModel`
Implementation for local/on-device models.

### Image Processing

The interface supports processing both individual images and batch processing of image folders:

```python
# Single image processing
images = process_images("/path/to/image.jpg")

# Folder processing (processes all supported image formats)
images = process_images("/path/to/image/folder/")

# Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .gif
```

**Requirements**: PIL/Pillow is required for image processing. Install with `pip install Pillow`.

### Execution Time Tracking

All queries automatically track execution time and return both the response and timing information:

```python
response, execution_time = vlm.execute_query('model_name', prompt)
print(f"Query completed in {execution_time:.3f} seconds")
```

This is useful for:
- Performance monitoring
- Model comparison
- Optimization analysis
- Benchmarking different configurations

### Output Formats

- **`text`**: Plain text response (default)
- **`json`**: Structured JSON response with metadata including image count and execution details

## Configuration

### API Models
For cloud-based models requiring API authentication:

```python
vlm.configure_model(
    model_name='GPT4V',
    api_key='your-api-key',
    base_url='https://api.example.com'  # optional
)
```

### Local Models
For on-device models:

```python
vlm.configure_model(
    model_name='CLIP_Local',
    model_path='/path/to/your/model.pkl'
)
```

## Error Handling

The interface provides comprehensive error handling:

- `ValueError`: Invalid configuration or parameters
- `FileNotFoundError`: Local model file not found
- `KeyError`: Unconfigured model accessed
- `RuntimeError`: Model not available

```python
try:
    response = vlm.execute_query('MyModel', prompt)
except KeyError as e:
    print(f"Model not configured: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

## Examples

Run the included examples:

```bash
# Basic functionality tests (including new image and timing features)
python test_vlm_interface.py

# Comprehensive usage examples
python example_usage.py

# Image processing and time tracking examples
python example_image_usage.py
```

## Architecture

The VLM interface follows a modular architecture:

```
VLMInterface
├── ModelBase (ABC)
│   ├── APIModel (API-based models)
│   └── LocalModel (Local models)
└── Prompt (Data structure)
```

This design allows for easy extension with new model types while maintaining a consistent interface.

## Requirements

- Python 3.7+
- No external dependencies for core functionality
- **PIL/Pillow** - Required for image processing features (`pip install Pillow`)
- Specific model implementations may require additional libraries

### Optional Dependencies

- **PIL/Pillow**: For image processing capabilities
  ```bash
  pip install Pillow
  ```

If PIL is not installed, image processing features will raise an `ImportError` with installation instructions.

## License

This project is part of the HyperX Parser VLM Test suite.