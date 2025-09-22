"""
VLM Interface Package

A unified interface for querying Vision-Language Models (VLM),
supporting both API-based closed models and on-device models with
enhanced multi-provider support and advanced capabilities.
"""

# Original VLM Interface
from .vlm_interface import (
    VLMInterface,
    Prompt,
    ModelBase,
    APIModel,
    LocalModel,
    configure_model,
    create_prompt,
    execute_query,
    process_images
)

# Enhanced VLM Interface
from .enhanced_vlm_interface import (
    EnhancedVLMInterface,
    EnhancedPrompt,
    EnhancedModelBase,
    EnhancedAPIModel,
    EnhancedLocalModel,
    ModelCapability,
    ModelProvider,
    ModelFactory,
    ModelConfig
)

__version__ = "2.0.0"

# Original interface exports
__all__ = [
    # Original VLM Interface
    "VLMInterface",
    "Prompt",
    "ModelBase",
    "APIModel",
    "LocalModel",
    "configure_model",
    "create_prompt",
    "execute_query",
    "process_images",

    # Enhanced VLM Interface
    "EnhancedVLMInterface",
    "EnhancedPrompt",
    "EnhancedModelBase",
    "EnhancedAPIModel",
    "EnhancedLocalModel",
    "ModelCapability",
    "ModelProvider",
    "ModelFactory",
    "ModelConfig"
]