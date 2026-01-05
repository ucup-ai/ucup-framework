"""
Model Interfaces for UCUP Framework Testing.

Provides standardized interfaces for model testing and validation.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ModelMetadata:
    """Metadata about a machine learning model."""

    name: str
    version: str
    provider: str
    model_type: str
    parameters: Optional[Dict[str, Any]] = None
    training_data: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    limitations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if not self.provider:
            raise ValueError("Model provider cannot be empty")


@dataclass
class ModelPrediction:
    """Container for model prediction results."""

    value: Any
    confidence: float
    alternatives: List[Any] = field(default_factory=list)
    timing: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class UniversalInput:
    """Universal input format for multimodal models."""

    text: Optional[str] = None
    image: Optional[Any] = None  # Can be PIL Image, numpy array, base64 string, etc.
    audio: Optional[Any] = None
    video: Optional[Any] = None
    structured_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_modality(self, modality: str) -> bool:
        """Check if input contains a specific modality."""
        return getattr(self, modality, None) is not None

    def get_available_modalities(self) -> List[str]:
        """Get list of available modalities in this input."""
        modalities = []
        for mod in ['text', 'image', 'audio', 'video']:
            if self.has_modality(mod):
                modalities.append(mod)
        return modalities


class UniversalModelInterface(ABC):
    """Abstract interface for universal model interactions."""

    @abstractmethod
    async def predict(self, input_data: Any, **kwargs) -> ModelPrediction:
        """Make a prediction using the model."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelMetadata:
        """Get metadata about the model."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Set[str]:
        """Get the set of capabilities supported by this model."""
        pass

    @abstractmethod
    def supports_modality(self, modality: str) -> bool:
        """Check if the model supports a specific modality."""
        pass

    def get_supported_modalities(self) -> List[str]:
        """Get list of supported modalities."""
        modalities = ["text", "image", "audio", "video"]
        return [mod for mod in modalities if self.supports_modality(mod)]

    async def predict_universal(self, universal_input: UniversalInput, **kwargs) -> ModelPrediction:
        """Make a prediction using universal input format."""
        return await self.predict(universal_input, **kwargs)
