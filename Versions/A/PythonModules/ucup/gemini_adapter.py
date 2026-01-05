"""
Google Gemini Adapter for UCUP Framework.

This module provides integration with Google Gemini AI models,
enabling advanced multimodal AI capabilities within the UCUP ecosystem.

Copyright (c) 2025 UCUP Framework Contributors. All rights reserved.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    # Fallback for newer versions
    import google.genai as genai
    # Define fallback classes for newer API
    class HarmCategory:
        HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
        HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
        HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"

    class HarmBlockThreshold:
        BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"

from .testing import ModelMetadata, ModelPrediction, UniversalInput, UniversalModelInterface

logger = logging.getLogger(__name__)


class GeminiAdapter(UniversalModelInterface):
    """
    Universal adapter for Google Gemini AI models.

    Provides a standardized interface for interacting with Google Gemini models,
    supporting text generation, multimodal processing, and probabilistic reasoning.

    Supports both Gemini 1.5 Flash and Gemini 1.5 Pro models with automatic
    fallback and error handling.
    """

    # Supported Gemini models
    SUPPORTED_MODELS = {
        "gemini-1.5-flash": {
            "name": "gemini-1.5-flash",
            "display_name": "Gemini 1.5 Flash",
            "max_tokens": 1048576,
            "context_window": 1048576,
            "multimodal": True,
            "capabilities": {"text_generation", "multimodal", "code_generation", "reasoning"}
        },
        "gemini-1.5-pro": {
            "name": "gemini-1.5-pro",
            "display_name": "Gemini 1.5 Pro",
            "max_tokens": 2097152,
            "context_window": 2097152,
            "multimodal": True,
            "capabilities": {"text_generation", "multimodal", "code_generation", "reasoning", "advanced_reasoning"}
        }
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        safety_settings: Optional[Dict] = None,
        generation_config: Optional[Dict] = None
    ):
        """
        Initialize Gemini adapter.

        Args:
            api_key: Google Gemini API key. If None, reads from GOOGLE_AI_API_KEY env var
            model: Model name ('gemini-1.5-flash' or 'gemini-1.5-pro')
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate. If None, uses model default
            timeout: Request timeout in seconds
            safety_settings: Custom safety settings for content filtering
            generation_config: Additional generation configuration
        """
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_AI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            available_models = list(self.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unsupported model '{model}'. Available: {available_models}")

        self.model_name = model
        self.model_config = self.SUPPORTED_MODELS[model]
        self.temperature = temperature
        self.max_tokens = max_tokens or min(4096, self.model_config["max_tokens"])
        self.timeout = timeout

        # Initialize safety settings
        self.safety_settings = safety_settings or self._get_default_safety_settings()

        # Initialize generation config
        self.generation_config = generation_config or self._get_default_generation_config()

        # Initialize Gemini client
        self._client = None
        self._initialize_client()

        logger.info(f"Initialized GeminiAdapter with model: {model}")

    def _initialize_client(self):
        """Initialize the Google Gemini client."""
        try:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            logger.debug("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def _get_default_safety_settings(self) -> List[Dict]:
        """Get default safety settings for content filtering."""
        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        ]

    def _get_default_generation_config(self) -> Dict:
        """Get default generation configuration."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": 0.8,
            "top_k": 10,
        }

    async def predict(self, input_data: Any, **kwargs) -> ModelPrediction:
        """
        Generate a prediction using the Gemini model.

        Args:
            input_data: Input data (UniversalInput or raw text)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ModelPrediction with response, confidence, and metadata
        """
        start_time = time.time()

        try:
            # Convert input to Gemini format
            gemini_input = self._prepare_input(input_data, **kwargs)

            # Make API call with timeout
            response = await asyncio.wait_for(
                self._call_gemini_api(gemini_input),
                timeout=self.timeout
            )

            # Process response
            prediction = self._process_response(response, start_time)

            logger.debug(f"Gemini prediction completed in {prediction.timing:.2f}s")
            return prediction

        except asyncio.TimeoutError:
            logger.error(f"Gemini API request timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Return error prediction
            return ModelPrediction(
                value="",
                confidence=0.0,
                timing=time.time() - start_time,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )

    def _prepare_input(self, input_data: Any, **kwargs) -> List[Dict]:
        """
        Prepare input data for Gemini API call.

        Args:
            input_data: UniversalInput or raw text/image data
            **kwargs: Additional parameters

        Returns:
            List of content parts for Gemini API
        """
        # Handle UniversalInput
        if isinstance(input_data, UniversalInput):
            return self._convert_universal_input_to_gemini(input_data)

        # Handle raw text
        if isinstance(input_data, str):
            return [{"text": input_data}]

        # Handle other formats
        return [{"text": str(input_data)}]

    def _convert_universal_input_to_gemini(self, universal_input: UniversalInput) -> List[Dict]:
        """
        Convert UniversalInput to Gemini content format.

        Args:
            universal_input: UniversalInput object

        Returns:
            List of content parts for Gemini
        """
        content_parts = []

        # Add text content
        if universal_input.text:
            content_parts.append({"text": universal_input.text})

        # Add image content (if supported)
        if universal_input.image and self.supports_modality("image"):
            # Handle different image formats
            if isinstance(universal_input.image, str):
                # Assume base64 encoded image
                content_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",  # Default assumption
                        "data": universal_input.image
                    }
                })
            elif hasattr(universal_input.image, 'tobytes'):
                # PIL Image or numpy array
                import base64
                image_data = base64.b64encode(universal_input.image.tobytes()).decode()
                content_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                })

        # Add structured data as text
        if universal_input.structured_data:
            json_text = json.dumps(universal_input.structured_data, indent=2)
            content_parts.append({"text": f"Structured Data:\n{json_text}"})

        return content_parts

    async def _call_gemini_api(self, content_parts: List[Dict]) -> Any:
        """
        Make the actual API call to Gemini.

        Args:
            content_parts: Content parts for the API call

        Returns:
            Gemini API response
        """
        # Create content for Gemini
        content = content_parts

        # Make the API call
        response = await self._client.generate_content_async(content)

        return response

    def _process_response(self, response: Any, start_time: float) -> ModelPrediction:
        """
        Process Gemini API response into ModelPrediction.

        Args:
            response: Raw Gemini API response
            start_time: Request start time for timing calculation

        Returns:
            Processed ModelPrediction
        """
        timing = time.time() - start_time

        # Extract text content
        try:
            text_content = response.text
        except Exception:
            text_content = ""

        # Extract usage information if available
        usage_info = {}
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            usage_info = {
                "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                "response_tokens": getattr(usage, 'candidates_token_count', 0),
                "total_tokens": getattr(usage, 'total_token_count', 0),
            }

        # Calculate confidence (simplified - could be improved)
        confidence = self._estimate_confidence(response, text_content)

        # Extract candidates for alternatives
        alternatives = []
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates[1:]:  # Skip first (main response)
                try:
                    alt_text = candidate.content.parts[0].text
                    alternatives.append(alt_text)
                except (AttributeError, IndexError):
                    continue

        # Build metadata
        metadata = {
            "model": self.model_name,
            "timing": timing,
            "usage": usage_info,
            "finish_reason": getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'unknown',
            "safety_ratings": self._extract_safety_ratings(response),
        }

        return ModelPrediction(
            value=text_content,
            confidence=confidence,
            alternatives=alternatives,
            timing=timing,
            metadata=metadata
        )

    def _estimate_confidence(self, response: Any, text_content: str) -> float:
        """
        Estimate confidence score from Gemini response.

        This is a simplified implementation. In practice, you might use
        multiple generations, log probabilities, or other signals.
        """
        base_confidence = 0.8  # Default high confidence for Gemini

        # Adjust based on response characteristics
        if not text_content.strip():
            return 0.0

        # Adjust based on safety filtering
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason.name == 'SAFETY':
                    base_confidence *= 0.7  # Reduce confidence for safety-filtered responses

        # Adjust based on response length (longer responses might be more confident)
        length_factor = min(len(text_content.split()) / 100, 1.0)
        base_confidence *= (0.7 + 0.3 * length_factor)

        return round(base_confidence, 3)

    def _extract_safety_ratings(self, response: Any) -> List[Dict]:
        """Extract safety ratings from response."""
        ratings = []

        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        ratings.append({
                            "category": rating.category.name if hasattr(rating.category, 'name') else str(rating.category),
                            "probability": rating.probability.name if hasattr(rating.probability, 'name') else str(rating.probability),
                            "blocked": getattr(rating, 'blocked', False)
                        })
        except Exception:
            pass

        return ratings

    def get_model_info(self) -> ModelMetadata:
        """Get metadata about the current model."""
        return ModelMetadata(
            name=self.model_name,
            version="1.5",  # Simplified version
            provider="Google",
            model_type="multimodal",
            parameters=self.model_config.get("parameters"),
            training_data="Mixed multimodal data (text, images, etc.)",
            capabilities=self.model_config["capabilities"],
            limitations={
                "max_tokens": self.model_config["max_tokens"],
                "context_window": self.model_config["context_window"],
                "rate_limits": "Varies by API tier",
            }
        )

    def get_capabilities(self) -> Set[str]:
        """Get the set of capabilities supported by this model."""
        return self.model_config["capabilities"].copy()

    def supports_modality(self, modality: str) -> bool:
        """Check if the model supports a specific modality."""
        modality_capabilities = {
            "text": "text_generation",
            "image": "multimodal",
            "audio": "multimodal",  # Gemini supports audio through multimodal
            "video": "multimodal",  # Gemini supports video through multimodal
        }

        required_capability = modality_capabilities.get(modality)
        return required_capability in self.get_capabilities() if required_capability else False

    def get_supported_modalities(self) -> List[str]:
        """Get list of supported modalities."""
        modalities = []
        for modality in ["text", "image", "audio", "video"]:
            if self.supports_modality(modality):
                modalities.append(modality)
        return modalities

    async def predict_universal(
        self, universal_input: UniversalInput, **kwargs
    ) -> ModelPrediction:
        """
        Predict using universal input format.

        This is a convenience method that converts UniversalInput to
        the model's native format before prediction.
        """
        return await self.predict(universal_input, **kwargs)

    def _convert_universal_input(self, universal_input: UniversalInput) -> Any:
        """
        Convert UniversalInput to Gemini's native format.

        This is a default implementation that can be overridden by subclasses.
        """
        return self._convert_universal_input_to_gemini(universal_input)

    # Additional utility methods
    def update_generation_config(self, **kwargs):
        """Update generation configuration parameters."""
        self.generation_config.update(kwargs)
        self._initialize_client()  # Re-initialize with new config

    def update_safety_settings(self, safety_settings: List[Dict]):
        """Update safety settings."""
        self.safety_settings = safety_settings
        self._initialize_client()  # Re-initialize with new settings

    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        return list(self.SUPPORTED_MODELS.keys())

    def switch_model(self, model_name: str):
        """Switch to a different Gemini model."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self._initialize_client()

        logger.info(f"Switched to model: {model_name}")


# Convenience function for creating Gemini adapters
def create_gemini_adapter(
    api_key: Optional[str] = None,
    model: str = "gemini-1.5-flash",
    **kwargs
) -> GeminiAdapter:
    """
    Create a Gemini adapter with sensible defaults.

    Args:
        api_key: Google Gemini API key
        model: Model name
        **kwargs: Additional parameters for GeminiAdapter

    Returns:
        Configured GeminiAdapter instance
    """
    return GeminiAdapter(api_key=api_key, model=model, **kwargs)
