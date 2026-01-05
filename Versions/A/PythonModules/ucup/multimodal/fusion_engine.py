"""
Advanced multimodal fusion engine for UCUP Framework.

This module provides sophisticated multimodal fusion capabilities that intelligently
combine text, visual, audio, and sensor data for comprehensive analysis and reasoning.
"""

import asyncio
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..errors import ProbabilisticError, RuntimeError, get_error_handler
from ..probabilistic import AlternativePath, ProbabilisticResult
from ..validation import validate_data


@dataclass
class MultimodalInputs:
    """Enhanced container for multi-modal input data with temporal tracking."""

    text_content: Optional[str] = None
    image_data: Optional[np.ndarray] = None
    audio_stream: Optional[np.ndarray] = None
    video_frames: Optional[List[np.ndarray]] = None
    sensor_data: Optional[Dict[str, float]] = None
    temporal_sequence: Optional[List[Dict[str, Any]]] = None  # Time-series data
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_available_modalities(self) -> List[str]:
        """Get list of available modalities in this input."""
        modalities = []
        if self.text_content is not None:
            modalities.append("text")
        if self.image_data is not None:
            modalities.append("image")
        if self.audio_stream is not None:
            modalities.append("audio")
        if self.video_frames is not None:
            modalities.append("video")
        if self.sensor_data is not None:
            modalities.append("sensor")
        if self.temporal_sequence is not None:
            modalities.append("temporal")
        return modalities

    def get_modality_data(self, modality: str) -> Any:
        """Get data for a specific modality."""
        mapping = {
            "text": self.text_content,
            "image": self.image_data,
            "audio": self.audio_stream,
            "video": self.video_frames,
            "sensor": self.sensor_data,
            "temporal": self.temporal_sequence,
        }
        return mapping.get(modality)


@dataclass
class FusedAnalysis:
    """Comprehensive result from multimodal fusion."""

    confidence_score: float
    semantic_embedding: np.ndarray
    cross_modal_relations: Dict[str, Dict[str, float]]
    uncertainty_estimate: float
    processing_time_ms: float
    modality_contributions: Dict[str, float]  # How much each modality contributed
    fusion_confidence: float  # Confidence in the fusion process itself
    alternative_fusions: List["FusedAnalysis"] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionWeights:
    """Dynamic weights for multimodal fusion."""

    modality_weights: Dict[str, float] = field(default_factory=dict)
    relation_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)
    temporal_weights: Dict[str, float] = field(default_factory=dict)
    confidence_weights: Dict[str, float] = field(default_factory=dict)

    def update_from_context(self, context: Dict[str, Any]):
        """Update weights based on contextual factors."""
        # Adjust weights based on context
        if context.get("task_type") == "visual_analysis":
            self.modality_weights.update({"image": 0.8, "video": 0.7, "text": 0.4})
        elif context.get("task_type") == "audio_analysis":
            self.modality_weights.update({"audio": 0.8, "text": 0.4})
        elif context.get("task_type") == "data_analysis":
            self.modality_weights.update({"sensor": 0.7, "temporal": 0.6})


class ModalityEmbeddingCache:
    """Cache system for modality embeddings to improve performance."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self._access_counter = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        if key in self.cache:
            self.access_times[key] = self._access_counter
            self._access_counter += 1
            return self.cache[key]
        return None

    def put(self, key: str, embedding: np.ndarray):
        """Cache an embedding."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]

        self.cache[key] = embedding
        self.access_times[key] = self._access_counter
        self._access_counter += 1


class MultimodalFusionEngine:
    """
    Advanced multimodal fusion engine that intelligently combines
    text, visual, audio, and sensor data for comprehensive analysis.

    Key features:
    - Dynamic fusion weighting based on context
    - Cross-modal attention mechanisms
    - Temporal sequence processing
    - Uncertainty quantification
    - Real-time adaptation capabilities
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.embedding_cache = ModalityEmbeddingCache(
            max_size=self.config.get("cache_size", 1000)
        )
        self.fusion_history = []
        self.performance_stats = defaultdict(list)

        # Initialize fusion models
        self.fusion_weights = FusionWeights()
        self._init_fusion_models()

    def _init_fusion_models(self):
        """Initialize modality-specific fusion models."""
        # These would be sophisticated ML models in production
        self.models = {
            "text_image_fusion": self._create_text_image_fusion_model(),
            "audio_visual_fusion": self._create_audio_visual_fusion_model(),
            "sensor_integration": self._create_sensor_integration_model(),
            "temporal_fusion": self._create_temporal_fusion_model(),
        }

    def _create_text_image_fusion_model(self):
        """Create text-image fusion model (placeholder)."""
        return {"type": "attention_fusion", "dim": 512}

    def _create_audio_visual_fusion_model(self):
        """Create audio-visual fusion model (placeholder)."""
        return {"type": "multimodal_attention", "dim": 1024}

    def _create_sensor_integration_model(self):
        """Create sensor integration model (placeholder)."""
        return {"type": "late_fusion", "dim": 256}

    def _create_temporal_fusion_model(self):
        """Create temporal fusion model (placeholder)."""
        return {"type": "sequence_attention", "dim": 768}

    async def fuse_multimodal_inputs(
        self, inputs: MultimodalInputs, context: Dict[str, Any] = None
    ) -> FusedAnalysis:
        """
        Fuse multiple modalities into unified semantic representation.

        Args:
            inputs: MultimodalInputs containing available data types
            context: Optional context for fusion adaptation

        Returns:
            FusedAnalysis with integrated understanding and uncertainty quantification
        """
        start_time = datetime.now()
        context = context or {}

        try:
            # Step 1: Extract embeddings for available modalities
            embeddings = await self._extract_embeddings(inputs)

            # Step 2: Update fusion weights based on context
            self.fusion_weights.update_from_context(context)

            # Step 3: Compute cross-modal relations
            relations = await self._compute_cross_modal_relations(embeddings, inputs)

            # Step 4: Apply fusion strategy
            fusion_result = await self._apply_fusion_strategy(
                embeddings, relations, inputs, context
            )

            # Step 5: Estimate uncertainty
            uncertainty = await self._estimate_fusion_uncertainty(
                embeddings, relations, fusion_result, inputs
            )

            # Step 6: Generate alternative fusions
            alternatives = await self._generate_alternative_fusions(
                embeddings, relations, context, max_alternatives=3
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Track performance
            self._track_performance(processing_time, len(embeddings), uncertainty)

            return FusedAnalysis(
                confidence_score=self._calculate_overall_confidence(
                    fusion_result, relations, uncertainty
                ),
                semantic_embedding=fusion_result,
                cross_modal_relations=relations,
                uncertainty_estimate=uncertainty,
                processing_time_ms=processing_time,
                modality_contributions=self._calculate_modality_contributions(
                    embeddings, relations
                ),
                fusion_confidence=self._assess_fusion_quality(embeddings, relations),
                alternative_fusions=alternatives,
                reasoning_chain=[],  # Would be populated with actual reasoning steps
                metadata={
                    "input_modalities": inputs.get_available_modalities(),
                    "fusion_strategy": self._select_fusion_strategy(inputs, context),
                    "num_embeddings": len(embeddings),
                    "embedding_dims": {k: v.shape[0] for k, v in embeddings.items()},
                },
            )

        except Exception as e:
            # Handle fusion errors gracefully
            error_handler = get_error_handler()
            await error_handler.handle_error(
                ProbabilisticError(
                    message=f"Multimodal fusion failed: {str(e)}",
                    reasoning_component="fusion_engine",
                    confidence_level=0.0,
                ),
                {"inputs": inputs, "context": context},
            )

            # Return minimal fallback result
            return FusedAnalysis(
                confidence_score=0.0,
                semantic_embedding=np.zeros(512),  # Minimal embedding
                cross_modal_relations={},
                uncertainty_estimate=1.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                modality_contributions={},
                fusion_confidence=0.0,
                alternative_fusions=[],
                reasoning_chain=["Fusion failed - returning minimal result"],
                metadata={"error": str(e)},
            )

    async def _extract_embeddings(
        self, inputs: MultimodalInputs
    ) -> Dict[str, np.ndarray]:
        """Extract semantic embeddings for each available modality."""
        embeddings = {}
        available_modalities = inputs.get_available_modalities()

        extraction_tasks = []
        for modality in available_modalities:
            if modality != "metadata":  # Skip metadata
                task = self._extract_modality_embedding(
                    modality, inputs.get_modality_data(modality)
                )
                extraction_tasks.append((modality, task))

        # Process embeddings concurrently
        results = await asyncio.gather(
            *[task for _, task in extraction_tasks], return_exceptions=True
        )

        # Organize results
        for (modality, _), result in zip(extraction_tasks, results):
            if isinstance(result, Exception):
                # Log error and use zero embedding as fallback
                print(f"Failed to extract {modality} embedding: {result}")
                embeddings[modality] = np.zeros(512)  # Fallback embedding
            else:
                embeddings[modality] = result

        return embeddings

    async def _extract_modality_embedding(self, modality: str, data: Any) -> np.ndarray:
        """Extract embedding for a specific modality."""
        # Create cache key
        cache_key = f"{modality}_{hash(str(data))}"

        # Check cache first
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        # Extract embedding based on modality
        if modality == "text":
            embedding = await self._extract_text_embedding(data)
        elif modality == "image":
            embedding = await self._extract_image_embedding(data)
        elif modality == "audio":
            embedding = await self._extract_audio_embedding(data)
        elif modality == "video":
            embedding = await self._extract_video_embedding(data)
        elif modality == "sensor":
            embedding = await self._extract_sensor_embedding(data)
        elif modality == "temporal":
            embedding = await self._extract_temporal_embedding(data)
        else:
            embedding = np.zeros(512)  # Unknown modality fallback

        # Cache the result
        self.embedding_cache.put(cache_key, embedding)

        return embedding

    async def _extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract text embedding (placeholder - would use actual NLP model)."""
        if not text:
            return np.zeros(768)

        # Mock embedding based on text properties
        text_len = len(text)
        word_count = len(text.split())
        has_numbers = any(c.isdigit() for c in text)

        # Create deterministic but varied embedding
        base_embedding = np.random.RandomState(hash(text) % 2**32).randn(768)
        base_embedding *= word_count / max(text_len, 1)  # Scale by complexity

        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            return scaler.fit_transform(base_embedding.reshape(1, -1)).flatten()
        else:
            return base_embedding / np.linalg.norm(base_embedding)

    async def _extract_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract image embedding (placeholder - would use actual vision model)."""
        if image is None or image.size == 0:
            return np.zeros(512)

        # Extract features from image array
        height, width = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
        channels = image.shape[2] if len(image.shape) >= 3 else 1

        # Calculate basic statistics
        mean_intensity = np.mean(image.astype(float))
        std_intensity = np.std(image.astype(float))
        colorfulness = self._calculate_image_colorfulness(image)

        # Create embedding from image features
        features = np.array(
            [
                height / 1000.0,  # Normalized dimensions
                width / 1000.0,
                channels / 3.0,  # Normalized channels
                mean_intensity / 255.0,
                std_intensity / 255.0,
                colorfulness,
                self._calculate_image_entropy(image),
                self._calculate_image_complexity(image),
            ]
        )

        # Expand to embedding dimensions using simple repetition + noise
        base_embedding = np.tile(features, 512 // len(features) + 1)[:512]
        base_embedding += np.random.RandomState(image.size % 2**32).randn(512) * 0.1

        return base_embedding / np.linalg.norm(base_embedding)

    def _calculate_image_colorfulness(self, image: np.ndarray) -> float:
        """Calculate colorfulness metric for image."""
        if len(image.shape) < 3 or image.shape[2] < 3:
            return 0.0  # Grayscale image

        # Convert to float
        img = image.astype(float)

        # Calculate colorfulness as described in research
        rg = img[:, :, 0] - img[:, :, 1]
        yb = 0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2]

        rg_mean, rg_std = np.mean(rg), np.std(rg)
        yb_mean, yb_std = np.mean(yb), np.std(yb)

        std_root = np.sqrt(rg_std**2 + yb_std**2)
        mean_root = np.sqrt(rg_mean**2 + yb_mean**2)

        return std_root + (0.3 * mean_root)

    def _calculate_image_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy / 8.0  # Normalize

    def _calculate_image_complexity(self, image: np.ndarray) -> float:
        """Calculate image complexity based on gradient magnitude."""
        if len(image.shape) == 2:
            gray = image
        else:
            gray = np.mean(image, axis=2)

        # Calculate gradients
        dx = np.gradient(gray, axis=1)
        dy = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        return np.mean(gradient_magnitude) / 255.0

    async def _extract_audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio embedding (placeholder)."""
        if audio is None or audio.size == 0:
            return np.zeros(256)

        # Calculate basic audio features
        energy = np.sqrt(np.mean(audio**2))
        zero_crossings = np.sum(np.diff(np.sign(audio))) / (2 * len(audio))
        spectral_centroid = self._calculate_spectral_centroid(audio)
        rms = np.sqrt(np.mean(audio**2))

        features = np.array(
            [
                energy,
                zero_crossings,
                spectral_centroid,
                rms,
                np.max(np.abs(audio)),  # Peak amplitude
                len(audio) / 44100.0,  # Duration in seconds (assuming 44.1kHz)
            ]
        )

        # Expand to embedding dimensions
        base_embedding = np.tile(features, 256 // len(features) + 1)[:256]
        base_embedding += np.random.RandomState(audio.size % 2**32).randn(256) * 0.05

        return base_embedding / np.linalg.norm(base_embedding)

    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid of audio."""
        if len(audio) == 0:
            return 0.0

        # Simple FFT for spectral centroid
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio))

        # Focus on positive frequencies
        pos_mask = freqs > 0
        magnitude = np.abs(fft[pos_mask])
        freqs_pos = freqs[pos_mask]

        if np.sum(magnitude) == 0:
            return 0.0

        centroid = np.sum(freqs_pos * magnitude) / np.sum(magnitude)
        return centroid

    async def _extract_video_embedding(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract video embedding by combining frame embeddings."""
        if not frames:
            return np.zeros(1024)

        # Extract embeddings for key frames (first, middle, last, and a few others)
        key_frame_indices = self._select_key_frames(frames)
        frame_embeddings = []

        for idx in key_frame_indices:
            frame_embedding = await self._extract_image_embedding(frames[idx])
            frame_embeddings.append(frame_embedding)

        if not frame_embeddings:
            return np.zeros(1024)

        # Combine frame embeddings using simple pooling
        combined = np.mean(frame_embeddings, axis=0)

        # Add temporal dynamics
        temporal_features = self._calculate_temporal_features(frame_embeddings)

        # Concatenate and ensure correct dimension
        video_embedding = np.concatenate([combined, temporal_features])
        if len(video_embedding) > 1024:
            video_embedding = video_embedding[:1024]
        elif len(video_embedding) < 1024:
            video_embedding = np.pad(video_embedding, (0, 1024 - len(video_embedding)))

        return video_embedding / np.linalg.norm(video_embedding)

    def _select_key_frames(
        self, frames: List[np.ndarray], max_frames: int = 8
    ) -> List[int]:
        """Select key frames from video for embedding extraction."""
        if len(frames) <= max_frames:
            return list(range(len(frames)))

        # Select evenly spaced frames
        indices = []
        for i in range(max_frames):
            idx = int(i * (len(frames) - 1) / (max_frames - 1))
            indices.append(idx)

        return indices

    def _calculate_temporal_features(
        self, frame_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate temporal dynamics features."""
        if len(frame_embeddings) < 2:
            return np.zeros(128)

        # Calculate differences between consecutive frames
        diffs = []
        for i in range(1, len(frame_embeddings)):
            diff = frame_embeddings[i] - frame_embeddings[i - 1]
            diffs.append(diff)

        # Temporal features
        motion_magnitude = np.mean([np.linalg.norm(d) for d in diffs])
        motion_variance = np.var([np.linalg.norm(d) for d in diffs])
        direction_consistency = np.mean(
            [
                np.dot(d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2))
                for d1, d2 in zip(diffs[:-1], diffs[1:])
            ]
        )

        temporal_features = np.array(
            [
                motion_magnitude,
                motion_variance,
                direction_consistency,
                len(diffs),  # Number of transitions
            ]
        )

        return np.tile(temporal_features, 128 // len(temporal_features) + 1)[:128]

    async def _extract_sensor_embedding(
        self, sensor_data: Dict[str, float]
    ) -> np.ndarray:
        """Extract sensor data embedding."""
        if not sensor_data or len(sensor_data) == 0:
            return np.zeros(128)

        # Convert sensor readings to embedding
        sensor_values = list(sensor_data.values())
        sensor_names = list(sensor_data.keys())

        # Basic statistical features
        values_array = np.array(sensor_values)
        features = [
            np.mean(values_array),
            np.std(values_array),
            np.min(values_array),
            np.max(values_array),
            np.median(values_array),
            len(sensor_data),  # Number of sensors
            self._calculate_sensor_correlation(values_array),
            self._calculate_sensor_diversity(sensor_names),
        ]

        # Add individual sensor values (up to limit)
        for value in sensor_values[:16]:  # Limit to 16 sensors
            features.append(value)

        # Expand to embedding dimensions
        base_embedding = np.tile(features, 128 // len(features) + 1)[:128]
        base_embedding += (
            np.random.RandomState(len(sensor_data) % 2**32).randn(128) * 0.05
        )

        return base_embedding / np.linalg.norm(base_embedding)

    def _calculate_sensor_correlation(self, values: np.ndarray) -> float:
        """Calculate correlation strength between sensors."""
        if len(values) < 2:
            return 0.0

        # Simple correlation measure
        corr_matrix = np.corrcoef(values)
        return np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))

    def _calculate_sensor_diversity(self, names: List[str]) -> float:
        """Calculate diversity based on sensor types."""
        # Simple diversity based on name prefixes
        prefixes = set()
        for name in names:
            prefix = name.split("_")[0] if "_" in name else name[:3]
            prefixes.add(prefix.lower())

        return len(prefixes) / max(len(names), 1)

    async def _extract_temporal_embedding(
        self, temporal_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract embedding from temporal sequence data."""
        if not temporal_data or len(temporal_data) == 0:
            return np.zeros(768)

        # Extract features from temporal sequence
        sequence_length = len(temporal_data)
        features_over_time = []

        # Process each time point
        for data_point in temporal_data[:50]:  # Limit for performance
            point_embedding = await self._extract_sensor_embedding(data_point)
            features_over_time.append(point_embedding)

        if not features_over_time:
            return np.zeros(768)

        # Calculate temporal patterns
        sequence_array = np.array(features_over_time)
        temporal_features = [
            np.mean(sequence_array, axis=0),  # Average over time
            np.std(sequence_array, axis=0),  # Variability over time
            sequence_array[0]
            if len(sequence_array) > 0
            else np.zeros_like(sequence_array[0]),  # Initial state
            sequence_array[-1]
            if len(sequence_array) > 0
            else np.zeros_like(sequence_array[0]),  # Final state
            self._calculate_trend(sequence_array),  # Overall trend
        ]

        # Flatten and combine
        combined_features = np.concatenate(temporal_features)

        # Ensure correct dimension
        if len(combined_features) > 768:
            combined_features = combined_features[:768]
        elif len(combined_features) < 768:
            combined_features = np.pad(
                combined_features, (0, 768 - len(combined_features))
            )

        return combined_features / np.linalg.norm(combined_features)

    def _calculate_trend(self, sequence: np.ndarray) -> np.ndarray:
        """Calculate trend for each feature dimension."""
        if len(sequence) < 2:
            return np.zeros(sequence.shape[1])

        # Simple linear trend calculation
        trends = []
        for dim in range(sequence.shape[1]):
            values = sequence[:, dim]
            if len(np.unique(values)) > 1:  # Avoid constant sequences
                trend = np.polyfit(range(len(values)), values, 1)[0]
            else:
                trend = 0.0
            trends.append(trend)

        return np.array(trends)

    async def _compute_cross_modal_relations(
        self, embeddings: Dict[str, np.ndarray], inputs: MultimodalInputs
    ) -> Dict[str, Dict[str, float]]:
        """Compute semantic relations between different modalities."""
        relations = {}
        modality_keys = list(embeddings.keys())

        for i, mod1 in enumerate(modality_keys):
            relations[mod1] = {}
            for mod2 in modality_keys[i + 1 :]:
                relation_score = await self._compute_relation_score(
                    mod1, embeddings[mod1], mod2, embeddings[mod2], inputs
                )
                relations[mod1][mod2] = relation_score

        return relations

    async def _compute_relation_score(
        self,
        mod1: str,
        emb1: np.ndarray,
        mod2: str,
        emb2: np.ndarray,
        inputs: MultimodalInputs,
    ) -> float:
        """Compute similarity score between two modality embeddings."""
        # Base similarity calculation
        if SKLEARN_AVAILABLE and len(emb1) == len(emb2):
            similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][
                0
            ]
        else:
            # Fallback to simple dot product similarity
            min_len = min(len(emb1), len(emb2))
            similarity = np.dot(emb1[:min_len], emb2[:min_len]) / (
                np.linalg.norm(emb1[:min_len]) * np.linalg.norm(emb2[:min_len])
            )

        # Apply modality-specific adjustment factors
        adjustment_factors = {
            ("text", "image"): 0.9,
            ("text", "audio"): 0.7,
            ("text", "video"): 0.8,
            ("text", "sensor"): 0.6,
            ("text", "temporal"): 0.65,
            ("image", "audio"): 0.8,
            ("image", "video"): 0.95,
            ("image", "sensor"): 0.7,
            ("image", "temporal"): 0.75,
            ("audio", "video"): 0.85,
            ("audio", "sensor"): 0.75,
            ("audio", "temporal"): 0.8,
            ("video", "sensor"): 0.8,
            ("video", "temporal"): 0.85,
            ("sensor", "temporal"): 0.9,
        }

        key = tuple(sorted([mod1, mod2]))
        factor = adjustment_factors.get(key, 0.5)

        # Context-based adjustment
        context_factor = self._get_context_factor(mod1, mod2, inputs.metadata)
        final_factor = factor * context_factor

        return float(max(0.0, min(1.0, similarity * final_factor)))

    def _get_context_factor(
        self, mod1: str, mod2: str, metadata: Dict[str, Any]
    ) -> float:
        """Get context-based adjustment factor for modality pair."""
        # Adjust based on metadata context
        if metadata.get("task_type") == "analysis":
            if {mod1, mod2} <= {"text", "sensor", "temporal"}:
                return 1.2  # Boost analytical modalities
        elif metadata.get("task_type") == "creative":
            if "image" in {mod1, mod2} or "audio" in {mod1, mod2}:
                return 1.1  # Boost creative modalities

        return 1.0  # No adjustment

    def _select_fusion_strategy(
        self, inputs: MultimodalInputs, context: Dict[str, Any]
    ) -> str:
        """Select appropriate fusion strategy based on inputs and context."""
        available_modalities = inputs.get_available_modalities()
        num_modalities = len(available_modalities)

        # Strategy selection logic
        if num_modalities == 1:
            return "single_modality"
        elif num_modalities == 2:
            return "early_fusion"
        elif num_modalities <= 4:
            return "attention_fusion"
        else:
            return "hierarchical_fusion"

    async def _apply_fusion_strategy(
        self,
        embeddings: Dict[str, np.ndarray],
        relations: Dict[str, Dict[str, float]],
        inputs: MultimodalInputs,
        context: Dict[str, Any],
    ) -> np.ndarray:
        """Apply the selected fusion strategy to combine embeddings."""

        if len(embeddings) == 0:
            return np.zeros(512)

        if len(embeddings) == 1:
            # Single modality - just return the embedding
            return list(embeddings.values())[0]

        strategy = self._select_fusion_strategy(inputs, context)

        if strategy == "early_fusion":
            return await self._apply_early_fusion(embeddings)
        elif strategy == "attention_fusion":
            return await self._apply_attention_fusion(embeddings, relations)
        elif strategy == "hierarchical_fusion":
            return await self._apply_hierarchical_fusion(embeddings, relations)
        else:
            # Default to attention fusion
            return await self._apply_attention_fusion(embeddings, relations)

    async def _apply_early_fusion(
        self, embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply early fusion by concatenating and transforming embeddings."""
        if not embeddings:
            return np.zeros(512)

        # Concatenate all embeddings
        concatenated = np.concatenate(list(embeddings.values()))

        # If too long, use PCA-like dimensionality reduction (simple approach)
        if len(concatenated) > 512:
            # Simple dimensionality reduction by averaging chunks
            chunk_size = len(concatenated) // 512 + 1
            reduced = []
            for i in range(0, len(concatenated), chunk_size):
                chunk = concatenated[i : i + chunk_size]
                reduced.append(np.mean(chunk))
            concatenated = np.array(reduced)[:512]
        elif len(concatenated) < 512:
            # Pad with zeros
            concatenated = np.pad(concatenated, (0, 512 - len(concatenated)))

        return concatenated / np.linalg.norm(concatenated)

    async def _apply_attention_fusion(
        self, embeddings: Dict[str, np.ndarray], relations: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """Apply attention-based fusion using relation scores."""
        if not embeddings:
            return np.zeros(512)

        modality_names = list(embeddings.keys())

        # Compute attention weights based on relations
        attention_weights = {}
        for mod in modality_names:
            # Weight = sum of relations + modality preference
            relation_sum = 0.0
            for other_mod, score in relations.get(mod, {}).items():
                relation_sum += score
            for other_mod in modality_names:
                if (
                    other_mod != mod
                    and other_mod in relations
                    and mod in relations[other_mod]
                ):
                    relation_sum += relations[other_mod][mod]

            base_weight = self.fusion_weights.modality_weights.get(mod, 0.5)
            attention_weights[mod] = base_weight + relation_sum

        # Normalize weights
        total_weight = sum(attention_weights.values())
        if total_weight > 0:
            for mod in attention_weights:
                attention_weights[mod] /= total_weight

        # Weighted combination
        fused = np.zeros_like(list(embeddings.values())[0])
        total_weight = 0

        for mod, embedding in embeddings.items():
            weight = attention_weights[mod]
            # Align embedding dimensions
            aligned_emb = self._align_embedding_dimensions(embedding, fused.shape[0])
            fused += aligned_emb * weight
            total_weight += weight

        return fused / max(total_weight, 1.0)

    def _align_embedding_dimensions(
        self, embedding: np.ndarray, target_dim: int
    ) -> np.ndarray:
        """Align embedding to target dimension."""
        current_dim = len(embedding)

        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate
            return embedding[:target_dim]
        else:
            # Pad with zeros
            return np.pad(embedding, (0, target_dim - current_dim))

    async def _apply_hierarchical_fusion(
        self, embeddings: Dict[str, np.ndarray], relations: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """Apply hierarchical fusion by grouping related modalities."""
        if len(embeddings) <= 3:
            # Use attention fusion for smaller sets
            return await self._apply_attention_fusion(embeddings, relations)

        # Group modalities by similarity
        groups = self._group_related_modalities(embeddings, relations)

        # Fuse within groups first
        group_fusions = {}
        for group_id, group_modalities in groups.items():
            group_embeddings = {mod: embeddings[mod] for mod in group_modalities}
            group_fusions[group_id] = await self._apply_attention_fusion(
                group_embeddings, relations
            )

        # Then fuse across groups
        final_fusion = await self._apply_attention_fusion(group_fusions, {})

        return final_fusion

    def _group_related_modalities(
        self,
        embeddings: Dict[str, np.ndarray],
        relations: Dict[str, Dict[str, float]],
        threshold: float = 0.7,
    ) -> Dict[int, List[str]]:
        """Group modalities based on relation strengths."""
        modality_names = list(embeddings.keys())
        groups = {}
        assigned = set()

        group_id = 0
        for mod in modality_names:
            if mod in assigned:
                continue

            group = [mod]
            assigned.add(mod)

            # Find strongly related modalities
            for other_mod in modality_names:
                if other_mod not in assigned:
                    # Check relation in both directions
                    relation_score = (
                        relations.get(mod, {}).get(other_mod, 0.0)
                        + relations.get(other_mod, {}).get(mod, 0.0)
                    ) / 2

                    if relation_score >= threshold:
                        group.append(other_mod)
                        assigned.add(other_mod)

            groups[group_id] = group
            group_id += 1

        return groups

    async def _estimate_fusion_uncertainty(
        self,
        embeddings: Dict[str, np.ndarray],
        relations: Dict[str, Dict[str, float]],
        fused_embedding: np.ndarray,
        inputs: MultimodalInputs,
    ) -> float:
        """Estimate uncertainty in the fused representation."""
        if not embeddings:
            return 1.0

        # Multiple uncertainty sources
        uncertainties = []

        # 1. Embedding quality uncertainty
        embedding_vars = []
        for embedding in embeddings.values():
            embedding_vars.append(np.var(embedding))
        embedding_uncertainty = np.mean(embedding_vars) if embedding_vars else 0.5
        uncertainties.append(embedding_uncertainty)

        # 2. Relation consistency uncertainty
        relation_scores = []
        for mod_relations in relations.values():
            relation_scores.extend(mod_relations.values())

        if relation_scores:
            relation_variance = np.var(relation_scores)
            relation_uncertainty = min(relation_variance * 2, 1.0)
        else:
            relation_uncertainty = 0.5
        uncertainties.append(relation_uncertainty)

        # 3. Modality count uncertainty
        num_modalities = len(embeddings)
        modality_uncertainty = max(
            0.0, (6 - num_modalities) / 6.0
        )  # Penalize few modalities
        uncertainties.append(modality_uncertainty)

        # 4. Temporal consistency uncertainty (if applicable)
        temporal_uncertainty = await self._calculate_temporal_uncertainty(inputs)
        uncertainties.append(temporal_uncertainty)

        # Combine uncertainties
        combined_uncertainty = np.mean(uncertainties)

        # Adjust based on confidence scores
        confidence_adjustment = 1.0 - np.mean(
            [
                self.fusion_weights.confidence_weights.get(mod, 0.5)
                for mod in embeddings.keys()
            ]
        )
        combined_uncertainty *= confidence_adjustment

        return min(combined_uncertainty, 1.0)

    async def _calculate_temporal_uncertainty(self, inputs: MultimodalInputs) -> float:
        """Calculate uncertainty based on temporal aspects."""
        if not inputs.temporal_sequence:
            return 0.1  # Low uncertainty if no temporal data

        # Check temporal consistency
        sequence_length = len(inputs.temporal_sequence)
        if sequence_length < 3:
            return 0.3  # Moderate uncertainty for short sequences

        # Check for gaps or anomalies in temporal data
        timestamps = []
        for point in inputs.temporal_sequence:
            if "timestamp" in point:
                timestamps.append(point["timestamp"])

        if len(timestamps) < 2:
            return 0.4

        # Calculate temporal regularity
        time_diffs = np.diff([float(t) for t in timestamps])
        regularity_score = 1.0 - min(np.std(time_diffs) / np.mean(time_diffs), 1.0)

        return max(0.0, 1.0 - regularity_score)

    def _calculate_overall_confidence(
        self,
        fused_embedding: np.ndarray,
        relations: Dict[str, Dict[str, float]],
        uncertainty: float,
    ) -> float:
        """Calculate overall confidence score for the fusion."""
        # Base confidence from embedding magnitude and uniformity
        embedding_score = min(np.linalg.norm(fused_embedding) / 10.0, 1.0)

        # Relation consistency score
        relation_scores = []
        for mod_relations in relations.values():
            relation_scores.extend(mod_relations.values())

        if relation_scores:
            relation_score = np.mean(relation_scores)
        else:
            relation_score = 0.5

        # Combine scores and adjust for uncertainty
        combined_score = embedding_score * 0.6 + relation_score * 0.4
        final_confidence = combined_score * (1.0 - uncertainty)

        return max(0.0, min(final_confidence, 1.0))

    def _calculate_modality_contributions(
        self, embeddings: Dict[str, np.ndarray], relations: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate contribution of each modality to the final fusion."""
        contributions = {}
        modality_names = list(embeddings.keys())

        for mod in modality_names:
            # Contribution based on relation strengths and embedding magnitude
            relation_strength = 0.0
            for other_mod in modality_names:
                if other_mod != mod:
                    relation_strength += relations.get(mod, {}).get(other_mod, 0.0)
                    relation_strength += relations.get(other_mod, {}).get(mod, 0.0)

            embedding_magnitude = np.linalg.norm(embeddings[mod])
            base_weight = self.fusion_weights.modality_weights.get(mod, 0.5)

            contribution = (
                embedding_magnitude * 0.3 + relation_strength * 0.4 + base_weight * 0.3
            )

            contributions[mod] = float(contribution)

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            for mod in contributions:
                contributions[mod] /= total

        return contributions

    def _assess_fusion_quality(
        self, embeddings: Dict[str, np.ndarray], relations: Dict[str, Dict[str, float]]
    ) -> float:
        """Assess the quality of the fusion process itself."""
        if len(embeddings) <= 1:
            return 1.0  # Perfect quality for single modality

        # Quality based on relation diversity and consistency
        all_relations = []
        for mod_relations in relations.values():
            all_relations.extend(mod_relations.values())

        if not all_relations:
            return 0.3  # Low quality if no relations

        # Diversity: avoid concentrated relations
        relation_entropy = self._calculate_entropy(all_relations)

        # Consistency: low variance in relation scores
        relation_variance = np.var(all_relations)

        # Balance: all modalities should contribute reasonably
        contributions = self._calculate_modality_contributions(embeddings, relations)
        balance_score = 1.0 - np.std(list(contributions.values()))

        quality = (
            relation_entropy * 0.4
            + (1.0 - min(relation_variance, 1.0)) * 0.4
            + balance_score * 0.2
        )

        return max(0.0, min(quality, 1.0))

    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy (diversity) of a list of values."""
        if not values:
            return 0.0

        # Normalize values
        values = np.array(values)
        values = values / np.sum(values) if np.sum(values) > 0 else values

        # Calculate entropy
        entropy = -np.sum(values * np.log2(values + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(values))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    async def _generate_alternative_fusions(
        self,
        embeddings: Dict[str, np.ndarray],
        relations: Dict[str, Dict[str, float]],
        context: Dict[str, Any],
        max_alternatives: int = 3,
    ) -> List["FusedAnalysis"]:
        """Generate alternative fusion approaches for robustness."""
        alternatives = []

        if len(embeddings) <= 1:
            return alternatives

        # Different fusion strategies to try
        strategies = ["early_fusion", "attention_fusion"]

        for strategy in strategies[:max_alternatives]:
            try:
                alt_embedding = np.zeros(512)  # Placeholder

                if strategy == "early_fusion":
                    alt_embedding = await self._apply_early_fusion(embeddings)
                elif strategy == "attention_fusion":
                    alt_embedding = await self._apply_attention_fusion(
                        embeddings, relations
                    )

                # Create alternative analysis
                alt_analysis = FusedAnalysis(
                    confidence_score=self._calculate_overall_confidence(
                        alt_embedding, relations, 0.2
                    ),
                    semantic_embedding=alt_embedding,
                    cross_modal_relations=relations,
                    uncertainty_estimate=0.2,  # Assume lower uncertainty for alternatives
                    processing_time_ms=0.0,  # Not measured for alternatives
                    modality_contributions=self._calculate_modality_contributions(
                        embeddings, relations
                    ),
                    fusion_confidence=0.8,
                    alternative_fusions=[],  # No recursive alternatives
                    reasoning_chain=[f"Alternative fusion using {strategy}"],
                    metadata={"fusion_strategy": strategy},
                )

                alternatives.append(alt_analysis)

            except Exception as e:
                print(f"Failed to generate {strategy} alternative: {e}")

        return alternatives

    def _track_performance(
        self, processing_time: float, num_modalities: int, uncertainty: float
    ):
        """Track performance metrics for monitoring and optimization."""
        self.performance_stats["processing_times"].append(processing_time)
        self.performance_stats["num_modalities"].append(num_modalities)
        self.performance_stats["uncertainties"].append(uncertainty)
        self.performance_stats["total_fusions"] = (
            self.performance_stats.get("total_fusions", 0) + 1
        )

        # Keep only recent stats
        max_histories = 1000
        for key in ["processing_times", "num_modalities", "uncertainties"]:
            if len(self.performance_stats[key]) > max_histories:
                self.performance_stats[key] = self.performance_stats[key][
                    -max_histories:
                ]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = dict(self.performance_stats)

        # Calculate aggregates
        if stats.get("processing_times"):
            stats["avg_processing_time"] = np.mean(stats["processing_times"])
            stats["median_processing_time"] = np.median(stats["processing_times"])
            stats["processing_time_percentiles"] = {
                "90th": np.percentile(stats["processing_times"], 90),
                "95th": np.percentile(stats["processing_times"], 95),
                "99th": np.percentile(stats["processing_times"], 99),
            }

        if stats.get("uncertainties"):
            stats["avg_uncertainty"] = np.mean(stats["uncertainties"])

        return stats

    async def adapt_to_feedback(self, feedback: Dict[str, Any]):
        """Adapt fusion parameters based on feedback."""
        if feedback.get("success", False):
            # Adjust weights toward successful patterns
            successful_modalities = feedback.get("successful_modalities", [])
            for modality in successful_modalities:
                current_weight = self.fusion_weights.modality_weights.get(modality, 0.5)
                self.fusion_weights.modality_weights[modality] = min(
                    1.0, current_weight + 0.1
                )

        # More adaptation logic would be implemented here

    def reset_cache(self):
        """Reset the embedding cache."""
        self.embedding_cache = ModalityEmbeddingCache()


# Convenience functions
async def fuse_multimodal(
    inputs: MultimodalInputs,
    config: Dict[str, Any] = None,
    context: Dict[str, Any] = None,
) -> FusedAnalysis:
    """Convenience function for multimodal fusion."""
    engine = MultimodalFusionEngine(config)
    return await engine.fuse_multimodal_inputs(inputs, context)


def create_fusion_engine(config: Dict[str, Any] = None) -> MultimodalFusionEngine:
    """Create and return a configured multimodal fusion engine."""
    return MultimodalFusionEngine(config)
