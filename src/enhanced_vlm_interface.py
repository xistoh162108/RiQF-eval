"""
Enhanced VLM Interface for Vision-Language Models

This module provides a comprehensive interface for querying multiple types of
Vision-Language Models (VLMs), supporting both API-based and local models with
specialized configurations for different model providers and capabilities.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Enum defining different model capabilities."""
    TEXT_GENERATION = "text_generation"
    VISION_LANGUAGE = "vision_language"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    CODE_GENERATION = "code_generation"


class ModelProvider(Enum):
    """Enum defining different model providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    LLAMA = "llama"
    DOCOWL = "docowl"
    SMOLDOCLING = "smoldocling"
    DONUT = "donut"
    DEEPSEEK_LOCAL = "deepseek-local-vlm"


@dataclass
class ModelConfig:
    """Configuration for different model types."""
    provider: ModelProvider
    model_name: str
    capabilities: List[ModelCapability]
    api_endpoint: Optional[str] = None
    requires_auth: bool = True
    supports_images: bool = False
    supports_documents: bool = False
    max_tokens: Optional[int] = None
    default_temperature: float = 0.7


# Model configurations database
MODEL_CONFIGS = {
    # OpenAI Models
    "openai-gpt-4": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
        api_endpoint="https://api.openai.com/v1/chat/completions",
        supports_images=False,
        max_tokens=8192
    ),
    "openai-gpt-4-vision": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-vision-preview",
        capabilities=[ModelCapability.VISION_LANGUAGE, ModelCapability.TEXT_GENERATION],
        api_endpoint="https://api.openai.com/v1/chat/completions",
        supports_images=True,
        max_tokens=4096
    ),
    "openai-gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        capabilities=[ModelCapability.TEXT_GENERATION],
        api_endpoint="https://api.openai.com/v1/chat/completions",
        max_tokens=4096
    ),

    # Gemini Models
    "gemini-pro": ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-pro",
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
        api_endpoint="https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
        max_tokens=8192
    ),
    "gemini-pro-vision": ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-pro-vision",
        capabilities=[ModelCapability.VISION_LANGUAGE, ModelCapability.TEXT_GENERATION],
        api_endpoint="https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent",
        supports_images=True,
        max_tokens=4096
    ),

    # DeepSeek Models
    "deepseek-chat": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
        api_endpoint="https://api.deepseek.com/v1/chat/completions",
        max_tokens=4096
    ),
    "deepseek-coder": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-coder",
        capabilities=[ModelCapability.CODE_GENERATION, ModelCapability.TEXT_GENERATION],
        api_endpoint="https://api.deepseek.com/v1/chat/completions",
        max_tokens=8192
    ),

    # Local Models
    "llama-7b": ModelConfig(
        provider=ModelProvider.LLAMA,
        model_name="llama-7b",
        capabilities=[ModelCapability.TEXT_GENERATION],
        requires_auth=False,
        max_tokens=2048
    ),
    "llama-13b": ModelConfig(
        provider=ModelProvider.LLAMA,
        model_name="llama-13b",
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
        requires_auth=False,
        max_tokens=4096
    ),
    "docowl": ModelConfig(
        provider=ModelProvider.DOCOWL,
        model_name="docowl",
        capabilities=[ModelCapability.DOCUMENT_UNDERSTANDING, ModelCapability.VISION_LANGUAGE],
        requires_auth=False,
        supports_images=True,
        supports_documents=True,
        max_tokens=2048
    ),
    "smoldocling": ModelConfig(
        provider=ModelProvider.SMOLDOCLING,
        model_name="smoldocling",
        capabilities=[ModelCapability.DOCUMENT_UNDERSTANDING, ModelCapability.TEXT_GENERATION],
        requires_auth=False,
        supports_documents=True,
        max_tokens=1024
    ),
    "donut": ModelConfig(
        provider=ModelProvider.DONUT,
        model_name="donut",
        capabilities=[ModelCapability.DOCUMENT_UNDERSTANDING, ModelCapability.VISION_LANGUAGE],
        requires_auth=False,
        supports_images=True,
        supports_documents=True,
        max_tokens=1024
    ),
    "deepseek-local-vlm": ModelConfig(
        provider=ModelProvider.DEEPSEEK_LOCAL,
        model_name="deepseek-local-vlm",
        capabilities=[ModelCapability.VISION_LANGUAGE, ModelCapability.TEXT_GENERATION],
        requires_auth=False,
        supports_images=True,
        max_tokens=4096
    ),
}


def process_images(image_path: Optional[str]) -> List[Dict[str, Any]]:
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
                    logger.warning(f"Could not load image {img_file}: {e}")
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
class EnhancedPrompt:
    """Enhanced data class for structured prompt configuration with model-specific features."""
    system_input: str
    user_prompt: str
    training_example: Optional[Union[str, List[str]]] = None
    output_format: str = 'text'
    image_path: Optional[str] = None
    document_path: Optional[str] = None
    task_type: Optional[ModelCapability] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt to dictionary format."""
        return {
            "system_input": self.system_input,
            "training_example": self.training_example,
            "user_prompt": self.user_prompt,
            "output_format": self.output_format,
            "image_path": self.image_path,
            "document_path": self.document_path,
            "task_type": self.task_type.value if self.task_type else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class EnhancedModelBase(ABC):
    """Enhanced abstract base class for all VLM models."""

    def __init__(self, model_id: str, config: ModelConfig):
        self.model_id = model_id
        self.config = config
        self.model_name = config.model_name
        self.provider = config.provider

    @abstractmethod
    def generate_response(self, prompt: EnhancedPrompt, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Generate response from the model based on the prompt and images."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and properly configured."""
        pass

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if the model supports a specific capability."""
        return capability in self.config.capabilities

    def optimize_prompt_for_model(self, prompt: EnhancedPrompt) -> EnhancedPrompt:
        """Optimize prompt based on model type and capabilities."""
        # Create a copy to avoid modifying the original
        optimized = EnhancedPrompt(
            system_input=prompt.system_input,
            user_prompt=prompt.user_prompt,
            training_example=prompt.training_example,
            output_format=prompt.output_format,
            image_path=prompt.image_path,
            document_path=prompt.document_path,
            task_type=prompt.task_type,
            temperature=prompt.temperature or self.config.default_temperature,
            max_tokens=prompt.max_tokens or self.config.max_tokens
        )

        # Model-specific prompt optimization
        if self.provider == ModelProvider.OPENAI:
            optimized.system_input = f"You are a helpful assistant. {optimized.system_input}"
        elif self.provider == ModelProvider.GEMINI:
            optimized.system_input = f"[System Context] {optimized.system_input}"
        elif self.provider == ModelProvider.DEEPSEEK:
            optimized.system_input = f"<system>{optimized.system_input}</system>"
        elif self.provider in [ModelProvider.DOCOWL, ModelProvider.SMOLDOCLING, ModelProvider.DONUT]:
            if ModelCapability.DOCUMENT_UNDERSTANDING in self.config.capabilities:
                optimized.system_input = f"Document Analysis Task: {optimized.system_input}"

        return optimized


class EnhancedAPIModel(EnhancedModelBase):
    """Enhanced model implementation for API-based models."""

    def __init__(self, model_id: str, config: ModelConfig, api_key: str, base_url: Optional[str] = None):
        super().__init__(model_id, config)
        self.api_key = api_key
        self.base_url = base_url or config.api_endpoint
        self._validate_api_key()

    def _validate_api_key(self):
        """Validate that the API key is provided."""
        if self.config.requires_auth and not self.api_key:
            raise ValueError(f"API key is required for model {self.model_id}")

    def generate_response(self, prompt: EnhancedPrompt, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Generate response using API call."""
        logger.info(f"Generating response for {self.model_id} via API")

        # Process images if provided
        if images is None and prompt.image_path:
            images = process_images(prompt.image_path)

        # Optimize prompt for this model
        optimized_prompt = self.optimize_prompt_for_model(prompt)

        # Validate model capabilities
        if images and not self.config.supports_images:
            raise ValueError(f"Model {self.model_id} does not support image inputs")

        # This would be implemented based on the specific API
        # For now, return a mock response with model-specific formatting
        response_data = {
            "model": self.model_id,
            "provider": self.provider.value,
            "prompt": optimized_prompt.to_dict(),
            "images_count": len(images) if images else 0,
            "capabilities": [cap.value for cap in self.config.capabilities],
            "response": self._generate_mock_response(optimized_prompt, images),
            "metadata": {
                "api_endpoint": self.base_url,
                "temperature": optimized_prompt.temperature,
                "max_tokens": optimized_prompt.max_tokens,
                **kwargs
            }
        }

        if optimized_prompt.output_format == "json":
            return response_data
        elif optimized_prompt.output_format == "text":
            return response_data["response"]
        else:
            raise ValueError(f"Unsupported output format: {optimized_prompt.output_format}")

    def _generate_mock_response(self, prompt: EnhancedPrompt, images: Optional[List[Dict[str, Any]]]) -> str:
        """Generate a mock response based on model type and capabilities."""
        image_info = f" with {len(images)} images" if images else ""

        if self.provider == ModelProvider.OPENAI:
            return f"OpenAI {self.model_name} response to: '{prompt.user_prompt}'{image_info}"
        elif self.provider == ModelProvider.GEMINI:
            return f"Gemini {self.model_name} analysis: '{prompt.user_prompt}'{image_info}"
        elif self.provider == ModelProvider.DEEPSEEK:
            return f"DeepSeek {self.model_name} generated response for: '{prompt.user_prompt}'{image_info}"
        else:
            return f"API response from {self.model_id}{image_info}"

    def is_available(self) -> bool:
        """Check if API is available."""
        return bool(self.api_key) if self.config.requires_auth else True


class EnhancedLocalModel(EnhancedModelBase):
    """Enhanced model implementation for local models."""

    def __init__(self, model_id: str, config: ModelConfig, model_path: str):
        super().__init__(model_id, config)
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
        logger.info(f"Loading local model {self.model_id} from {self.model_path}")

        # This would be implemented based on the specific model type
        # For now, just store the path and mark as loaded
        self.model = {
            "path": self.model_path,
            "loaded": True,
            "model_type": self.provider.value,
            "capabilities": [cap.value for cap in self.config.capabilities]
        }

    def generate_response(self, prompt: EnhancedPrompt, images: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """Generate response using local model."""
        if not self.model:
            raise RuntimeError(f"Model {self.model_id} is not loaded")

        logger.info(f"Generating response for {self.model_id} locally")

        # Process images if provided
        if images is None and prompt.image_path:
            images = process_images(prompt.image_path)

        # Optimize prompt for this model
        optimized_prompt = self.optimize_prompt_for_model(prompt)

        # Validate model capabilities
        if images and not self.config.supports_images:
            raise ValueError(f"Model {self.model_id} does not support image inputs")

        # This would be implemented based on the specific model
        # For now, return a mock response with model-specific formatting
        response_data = {
            "model": self.model_id,
            "provider": self.provider.value,
            "model_path": self.model_path,
            "prompt": optimized_prompt.to_dict(),
            "images_count": len(images) if images else 0,
            "capabilities": [cap.value for cap in self.config.capabilities],
            "response": self._generate_mock_response(optimized_prompt, images),
            "metadata": {
                "model_loaded": True,
                "temperature": optimized_prompt.temperature,
                "max_tokens": optimized_prompt.max_tokens,
                **kwargs
            }
        }

        if optimized_prompt.output_format == "json":
            return response_data
        elif optimized_prompt.output_format == "text":
            return response_data["response"]
        else:
            raise ValueError(f"Unsupported output format: {optimized_prompt.output_format}")

    def _generate_mock_response(self, prompt: EnhancedPrompt, images: Optional[List[Dict[str, Any]]]) -> str:
        """Generate a mock response based on model type and capabilities."""
        image_info = f" with {len(images)} images" if images else ""

        if self.provider == ModelProvider.LLAMA:
            return f"Llama {self.model_name} local response: '{prompt.user_prompt}'{image_info}"
        elif self.provider == ModelProvider.DOCOWL:
            return f"DocOWL document analysis: '{prompt.user_prompt}'{image_info}"
        elif self.provider == ModelProvider.SMOLDOCLING:
            return f"SmolDocLing document processing: '{prompt.user_prompt}'{image_info}"
        elif self.provider == ModelProvider.DONUT:
            return f"Donut document understanding: '{prompt.user_prompt}'{image_info}"
        elif self.provider == ModelProvider.DEEPSEEK_LOCAL:
            return f"DeepSeek Local VLM response: '{prompt.user_prompt}'{image_info}"
        else:
            return f"Local response from {self.model_id}{image_info}"

    def is_available(self) -> bool:
        """Check if local model is available."""
        return self.model is not None and os.path.exists(self.model_path)


class ModelFactory:
    """Factory class for creating and configuring models."""

    @staticmethod
    def create_model(model_id: str, api_key: Optional[str] = None,
                    model_path: Optional[str] = None, base_url: Optional[str] = None) -> EnhancedModelBase:
        """
        Create a model instance based on the model ID.

        Args:
            model_id (str): The model identifier (e.g., 'openai-gpt-4', 'llama-7b')
            api_key (str, optional): API key for API-based models
            model_path (str, optional): Local path for local models
            base_url (str, optional): Custom API endpoint

        Returns:
            EnhancedModelBase: Configured model instance

        Raises:
            ValueError: If model is not supported or configuration is invalid
        """
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_id}. Available models: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_id]

        # Determine if this is an API or local model based on configuration
        if config.api_endpoint:
            # API-based model
            if config.requires_auth and not api_key:
                raise ValueError(f"API key is required for model {model_id}")
            return EnhancedAPIModel(model_id, config, api_key or "", base_url)
        else:
            # Local model
            if not model_path:
                raise ValueError(f"Model path is required for local model {model_id}")
            return EnhancedLocalModel(model_id, config, model_path)

    @staticmethod
    def get_available_models() -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return MODEL_CONFIGS.copy()

    @staticmethod
    def get_models_by_capability(capability: ModelCapability) -> List[str]:
        """Get all models that support a specific capability."""
        return [
            model_id for model_id, config in MODEL_CONFIGS.items()
            if capability in config.capabilities
        ]

    @staticmethod
    def get_models_by_provider(provider: ModelProvider) -> List[str]:
        """Get all models from a specific provider."""
        return [
            model_id for model_id, config in MODEL_CONFIGS.items()
            if config.provider == provider
        ]


class EnhancedVLMInterface:
    """Enhanced main interface for Vision-Language Models supporting multiple providers and capabilities."""

    def __init__(self):
        self.models: Dict[str, EnhancedModelBase] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def configure_model(self, model_id: str, api_key: Optional[str] = None,
                       model_path: Optional[str] = None, base_url: Optional[str] = None) -> EnhancedModelBase:
        """
        Configure a model for either API-based or local usage.

        Args:
            model_id (str): The model identifier (e.g., 'openai-gpt-4', 'llama-7b')
            api_key (str, optional): API key for API-based models
            model_path (str, optional): Local path for local models
            base_url (str, optional): Custom API endpoint

        Returns:
            EnhancedModelBase: Configured model instance

        Raises:
            ValueError: If model configuration is invalid
        """
        try:
            model = ModelFactory.create_model(model_id, api_key, model_path, base_url)
            self.models[model_id] = model
            self.logger.info(f"Successfully configured model: {model_id}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to configure model {model_id}: {e}")
            raise

    def create_prompt(self, system_input: str, user_prompt: str,
                     training_example: Optional[Union[str, List[str]]] = None,
                     output_format: str = 'text',
                     image_path: Optional[str] = None,
                     document_path: Optional[str] = None,
                     task_type: Optional[ModelCapability] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> EnhancedPrompt:
        """
        Create an enhanced prompt with advanced configuration options.

        Args:
            system_input (str): System context for the model
            user_prompt (str): User's query
            training_example (Union[str, List[str]], optional): Few-shot examples
            output_format (str): Desired output format ('text' or 'json')
            image_path (str, optional): Path to image file or folder
            document_path (str, optional): Path to document file
            task_type (ModelCapability, optional): Type of task for optimization
            temperature (float, optional): Generation temperature
            max_tokens (int, optional): Maximum tokens to generate

        Returns:
            EnhancedPrompt: Configured prompt object
        """
        return EnhancedPrompt(
            system_input=system_input,
            user_prompt=user_prompt,
            training_example=training_example,
            output_format=output_format,
            image_path=image_path,
            document_path=document_path,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def execute_query(self, model_id: str, prompt: EnhancedPrompt, **kwargs) -> Tuple[Any, float]:
        """
        Execute a query on the specified model with comprehensive error handling.

        Args:
            model_id (str): The configured model identifier
            prompt (EnhancedPrompt): The prompt to execute
            **kwargs: Additional parameters

        Returns:
            Tuple[Any, float]: Model response and execution time

        Raises:
            KeyError: If model is not configured
            RuntimeError: If model is not available
            ValueError: If prompt is incompatible with model capabilities
        """
        # Validate model exists
        if model_id not in self.models:
            available_models = list(self.models.keys())
            self.logger.error(f"Model '{model_id}' not configured. Available: {available_models}")
            raise KeyError(f"Model '{model_id}' is not configured. Call configure_model() first.")

        model = self.models[model_id]

        # Check model availability
        if not model.is_available():
            self.logger.error(f"Model '{model_id}' is not available")
            raise RuntimeError(f"Model '{model_id}' is not available.")

        # Validate model capabilities against prompt requirements
        self._validate_prompt_compatibility(model, prompt)

        # Track execution time
        start_time = time.time()

        try:
            self.logger.info(f"Executing query on {model_id}")

            # Process images if provided in the prompt
            images = None
            if prompt.image_path:
                self.logger.info(f"Processing images from: {prompt.image_path}")
                images = process_images(prompt.image_path)

            # Generate response
            response = model.generate_response(prompt, images, **kwargs)

            end_time = time.time()
            execution_time = end_time - start_time

            self.logger.info(f"Query completed in {execution_time:.3f} seconds")
            return response, execution_time

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.error(f"Query failed after {execution_time:.3f} seconds: {e}")
            raise

    def _validate_prompt_compatibility(self, model: EnhancedModelBase, prompt: EnhancedPrompt):
        """Validate that the prompt is compatible with the model's capabilities."""
        # Check image support
        if prompt.image_path and not model.config.supports_images:
            raise ValueError(f"Model {model.model_id} does not support image inputs")

        # Check document support
        if prompt.document_path and not model.config.supports_documents:
            raise ValueError(f"Model {model.model_id} does not support document inputs")

        # Check task type compatibility
        if prompt.task_type and not model.supports_capability(prompt.task_type):
            raise ValueError(f"Model {model.model_id} does not support {prompt.task_type.value} tasks")

    def get_configured_models(self) -> List[str]:
        """Get list of configured model identifiers."""
        return list(self.models.keys())

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a configured model."""
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' is not configured")

        model = self.models[model_id]
        return {
            "model_id": model_id,
            "provider": model.provider.value,
            "capabilities": [cap.value for cap in model.config.capabilities],
            "supports_images": model.config.supports_images,
            "supports_documents": model.config.supports_documents,
            "max_tokens": model.config.max_tokens,
            "is_available": model.is_available(),
            "model_type": "API" if hasattr(model, 'api_key') else "Local"
        }

    def remove_model(self, model_id: str):
        """Remove a configured model."""
        if model_id in self.models:
            self.logger.info(f"Removing model: {model_id}")
            del self.models[model_id]

    def get_available_model_configs(self) -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return ModelFactory.get_available_models()

    def find_models_for_task(self, capability: ModelCapability) -> List[str]:
        """Find all available models that support a specific capability."""
        return ModelFactory.get_models_by_capability(capability)

    def batch_configure_models(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        Configure multiple models at once.

        Args:
            model_configs: Dict mapping model_id to configuration parameters

        Returns:
            Dict mapping model_id to success status
        """
        results = {}
        for model_id, config in model_configs.items():
            try:
                self.configure_model(model_id, **config)
                results[model_id] = True
            except Exception as e:
                self.logger.error(f"Failed to configure {model_id}: {e}")
                results[model_id] = False
        return results


# Legacy compatibility functions
def configure_model(model_id: str, api_key: Optional[str] = None,
                   model_path: Optional[str] = None, base_url: Optional[str] = None) -> EnhancedModelBase:
    """Legacy function for configuring a single model."""
    vlm = EnhancedVLMInterface()
    return vlm.configure_model(model_id, api_key, model_path, base_url)


def create_prompt(system_input: str, user_prompt: str,
                 training_example: Optional[Union[str, List[str]]] = None,
                 output_format: str = 'text',
                 image_path: Optional[str] = None,
                 document_path: Optional[str] = None,
                 task_type: Optional[ModelCapability] = None) -> EnhancedPrompt:
    """Legacy function for creating a prompt."""
    vlm = EnhancedVLMInterface()
    return vlm.create_prompt(system_input, user_prompt, training_example, output_format,
                            image_path, document_path, task_type)


def execute_query(model: EnhancedModelBase, prompt: EnhancedPrompt, **kwargs) -> Tuple[Any, float]:
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