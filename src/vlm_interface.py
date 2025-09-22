"""
VLM Interface for Vision-Language Models

This module provides a unified interface for querying Vision-Language Models (VLM),
supporting both API-based closed models and on-device models with image processing
and execution time tracking capabilities.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


def process_images(image_path: Optional[str]) -> List[Any]:
    """
    Process images given a file path or directory.

    Args:
        image_path (str, optional): Path to a single image file or a folder containing images.

    Returns:
        list: A list of processed image objects.

    Raises:
        ImportError: If PIL is not available.
        FileNotFoundError: If the image path doesn't exist.
    """
    if not image_path:
        return []

    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for image processing. Install with: pip install Pillow")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    images = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    if os.path.isdir(image_path):
        # Process all images in the folder
        for img_file in os.listdir(image_path):
            if img_file.lower().endswith(supported_extensions):
                try:
                    img_full_path = os.path.join(image_path, img_file)
                    img = Image.open(img_full_path)  # type: ignore
                    images.append({
                        'image': img,
                        'path': img_full_path,
                        'filename': img_file
                    })
                except Exception as e:
                    print(f"Warning: Could not load image {img_file}: {e}")
    else:
        # Single image file path
        if image_path.lower().endswith(supported_extensions):
            try:
                img = Image.open(image_path)  # type: ignore
                images.append({
                    'image': img,
                    'path': image_path,
                    'filename': os.path.basename(image_path)
                })
            except Exception as e:
                raise ValueError(f"Could not load image {image_path}: {e}")
        else:
            raise ValueError(f"Unsupported image format. Supported formats: {supported_extensions}")

    return images


@dataclass
class Prompt:
    """Data class for structured prompt configuration."""
    system_input: str
    user_prompt: str
    training_example: Optional[Union[str, List[str]]] = None
    output_format: str = 'text'
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt to dictionary format."""
        return {
            "system_input": self.system_input,
            "training_example": self.training_example,
            "user_prompt": self.user_prompt,
            "output_format": self.output_format,
            "image_path": self.image_path
        }


class ModelBase(ABC):
    """Abstract base class for all VLM models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_response(self, prompt: Prompt, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Generate response from the model based on the prompt and images."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and properly configured."""
        pass


class APIModel(ModelBase):
    """Model implementation for API-based closed models."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self._validate_api_key()

    def _validate_api_key(self):
        """Validate that the API key is provided."""
        if not self.api_key:
            raise ValueError(f"API key is required for model {self.model_name}")

    def generate_response(self, prompt: Prompt, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Generate response using API call."""
        # Process images if provided
        if images is None and prompt.image_path:
            images = process_images(prompt.image_path)

        # This would be implemented based on the specific API
        # For now, return a mock response
        response_data = {
            "model": self.model_name,
            "prompt": prompt.to_dict(),
            "images_count": len(images) if images else 0,
            "response": f"Mock API response for {self.model_name} with {len(images) if images else 0} images",
            "metadata": kwargs
        }

        if prompt.output_format == "json":
            return response_data
        elif prompt.output_format == "text":
            return response_data["response"]
        else:
            raise ValueError(f"Unsupported output format: {prompt.output_format}")

    def is_available(self) -> bool:
        """Check if API is available."""
        return bool(self.api_key)


class LocalModel(ModelBase):
    """Model implementation for on-device models."""

    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name)
        self.model_path = model_path
        self._validate_model_path()
        self.model = None
        self._load_model()

    def _validate_model_path(self):
        """Validate that the model path exists."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

    def _load_model(self):
        """Load the local model."""
        # This would be implemented based on the specific model type
        # For now, just store the path
        self.model = {"path": self.model_path, "loaded": True}

    def generate_response(self, prompt: Prompt, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Generate response using local model."""
        if not self.model:
            raise RuntimeError(f"Model {self.model_name} is not loaded")

        # Process images if provided
        if images is None and prompt.image_path:
            images = process_images(prompt.image_path)

        # This would be implemented based on the specific model
        # For now, return a mock response
        response_data = {
            "model": self.model_name,
            "model_path": self.model_path,
            "prompt": prompt.to_dict(),
            "images_count": len(images) if images else 0,
            "response": f"Mock local response for {self.model_name} with {len(images) if images else 0} images",
            "metadata": kwargs
        }

        if prompt.output_format == "json":
            return response_data
        elif prompt.output_format == "text":
            return response_data["response"]
        else:
            raise ValueError(f"Unsupported output format: {prompt.output_format}")

    def is_available(self) -> bool:
        """Check if local model is available."""
        return self.model is not None and os.path.exists(self.model_path)


class VLMInterface:
    """Main interface for Vision-Language Models."""

    def __init__(self):
        self.models = {}

    def configure_model(self, model_name: str, api_key: Optional[str] = None,
                       model_path: Optional[str] = None, base_url: Optional[str] = None) -> ModelBase:
        """
        Configure the model for either API-based or on-device usage.

        Args:
            model_name (str): The name of the model (e.g., 'CLIP', 'BLIP').
            api_key (str, optional): API key for closed models.
            model_path (str, optional): Local file path for on-device models.
            base_url (str, optional): Base URL for API calls.

        Returns:
            ModelBase: Loaded model instance.

        Raises:
            ValueError: If neither api_key nor model_path is provided.
        """
        if api_key:
            model = APIModel(model_name, api_key, base_url)
        elif model_path:
            model = LocalModel(model_name, model_path)
        else:
            raise ValueError("Either api_key or model_path must be provided.")

        self.models[model_name] = model
        return model

    def create_prompt(self, system_input: str, user_prompt: str,
                     training_example: Optional[Union[str, List[str]]] = None,
                     output_format: str = 'text',
                     image_path: Optional[str] = None) -> Prompt:
        """
        Constructs the prompt for the model to process.

        Args:
            system_input (str): Defines the context for the model.
            user_prompt (str): The user's query to be answered by the model.
            training_example (str or list of str, optional): One-shot or few-shot training examples.
            output_format (str): Desired output format, default is 'text'.
            image_path (str, optional): Path to image file or folder containing images.

        Returns:
            Prompt: A structured prompt object.
        """
        return Prompt(
            system_input=system_input,
            user_prompt=user_prompt,
            training_example=training_example,
            output_format=output_format,
            image_path=image_path
        )

    def execute_query(self, model_name: str, prompt: Prompt, **kwargs) -> Tuple[Any, float]:
        """
        Executes the model query based on the provided prompt and model configuration.

        Args:
            model_name (str): The name of the configured model to use.
            prompt (Prompt): The constructed prompt object.
            **kwargs: Additional parameters to pass to the model.

        Returns:
            Tuple[Any, float]: The model's response and execution time in seconds.

        Raises:
            KeyError: If the model is not configured.
            RuntimeError: If the model is not available.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not configured. Call configure_model() first.")

        model = self.models[model_name]

        if not model.is_available():
            raise RuntimeError(f"Model '{model_name}' is not available.")

        # Track execution time
        start_time = time.time()

        # Process images if provided in the prompt
        images = None
        if prompt.image_path:
            images = process_images(prompt.image_path)

        # Generate response
        response = model.generate_response(prompt, images, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        return response, execution_time

    def get_configured_models(self) -> List[str]:
        """Get list of configured model names."""
        return list(self.models.keys())

    def remove_model(self, model_name: str):
        """Remove a configured model."""
        if model_name in self.models:
            del self.models[model_name]


# Convenience functions for backward compatibility
def configure_model(model_name: str, api_key: Optional[str] = None,
                   model_path: Optional[str] = None, base_url: Optional[str] = None) -> ModelBase:
    """Legacy function for configuring a single model."""
    vlm = VLMInterface()
    return vlm.configure_model(model_name, api_key, model_path, base_url)


def create_prompt(system_input: str, user_prompt: str,
                 training_example: Optional[Union[str, List[str]]] = None,
                 output_format: str = 'text',
                 image_path: Optional[str] = None) -> Prompt:
    """Legacy function for creating a prompt."""
    vlm = VLMInterface()
    return vlm.create_prompt(system_input, user_prompt, training_example, output_format, image_path)


def execute_query(model: ModelBase, prompt: Prompt, **kwargs) -> Tuple[Any, float]:
    """Legacy function for executing a query on a model instance with time tracking."""
    start_time = time.time()

    # Process images if provided in the prompt
    images = None
    if prompt.image_path:
        images = process_images(prompt.image_path)

    response = model.generate_response(prompt, images, **kwargs)

    end_time = time.time()
    execution_time = end_time - start_time

    return response, execution_time