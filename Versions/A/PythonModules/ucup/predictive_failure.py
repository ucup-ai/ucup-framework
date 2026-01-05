"""
Predictive Failure Detection Module for UCUP Framework

This module provides ML-based failure prediction capabilities for the UCUP framework,
enabling proactive failure detection and prevention in agent systems.
"""

import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .reliability import FailureDetector, FailurePattern

logger = logging.getLogger(__name__)


@dataclass
class FailurePrediction:
    """Prediction result for potential failure."""

    timestamp: datetime
    failure_probability: float
    failure_type: str
    confidence_score: float
    predicted_time_to_failure: Optional[timedelta] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class FeatureVector:
    """ML feature vector for failure prediction."""

    timestamp: datetime
    features: Dict[str, Union[float, int, bool]]
    label: Optional[bool] = None  # True if failure occurred
    session_id: Optional[str] = None

    def to_numpy(self) -> np.ndarray:
        """Convert features to numpy array."""
        return np.array(list(self.features.values()))

    def feature_names(self) -> List[str]:
        """Get feature names in order."""
        return list(self.features.keys())


class FailureFeatureExtractor:
    """Extracts features from agent sessions for failure prediction."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            "confidence_mean",
            "confidence_std",
            "confidence_trend",
            "response_time_mean",
            "response_time_std",
            "response_time_trend",
            "memory_usage",
            "cpu_usage",
            "error_rate",
            "coordination_complexity",
            "task_load",
            "communication_frequency",
            "uncertainty_level",
            "decision_confidence",
            "fallback_usage",
        ]

    def extract_features(self, session_data: Dict[str, Any]) -> FeatureVector:
        """Extract features from session data."""
        features = {}

        # Confidence metrics
        confidence_history = session_data.get("confidence_history", [])
        if confidence_history:
            features["confidence_mean"] = np.mean(confidence_history)
            features["confidence_std"] = np.std(confidence_history)
            features["confidence_trend"] = self._calculate_trend(confidence_history)
        else:
            features["confidence_mean"] = 0.5
            features["confidence_std"] = 0.0
            features["confidence_trend"] = 0.0

        # Performance metrics
        response_times = session_data.get("response_times", [])
        if response_times:
            features["response_time_mean"] = np.mean(response_times)
            features["response_time_std"] = np.std(response_times)
            features["response_time_trend"] = self._calculate_trend(response_times)
        else:
            features["response_time_mean"] = 1.0
            features["response_time_std"] = 0.0
            features["response_time_trend"] = 0.0

        # System metrics
        features["memory_usage"] = session_data.get("memory_usage", 0.5)
        features["cpu_usage"] = session_data.get("cpu_usage", 0.5)
        features["error_rate"] = session_data.get("error_rate", 0.0)

        # Coordination metrics
        features["coordination_complexity"] = session_data.get(
            "coordination_complexity", 0.0
        )
        features["task_load"] = session_data.get("task_load", 0.0)
        features["communication_frequency"] = session_data.get(
            "communication_frequency", 0.0
        )

        # Uncertainty metrics
        features["uncertainty_level"] = session_data.get("uncertainty_level", 0.5)
        features["decision_confidence"] = session_data.get("decision_confidence", 0.5)
        features["fallback_usage"] = session_data.get("fallback_usage", 0.0)

        return FeatureVector(
            timestamp=datetime.now(),
            features=features,
            session_id=session_data.get("session_id"),
        )

    def _calculate_trend(self, values: List[float], window: int = 5) -> float:
        """Calculate trend using linear regression on recent values."""
        if len(values) < window:
            return 0.0

        recent_values = values[-window:]
        x = np.arange(len(recent_values))
        y = np.array(recent_values)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope


class FailurePredictionModel(abc.ABC):
    """Abstract base class for failure prediction models."""

    @abc.abstractmethod
    def train(self, feature_vectors: List[FeatureVector]) -> None:
        """Train the model on feature vectors."""
        pass

    @abc.abstractmethod
    def predict(self, feature_vector: FeatureVector) -> FailurePrediction:
        """Predict failure probability for a feature vector."""
        pass

    @abc.abstractmethod
    def predict_batch(
        self, feature_vectors: List[FeatureVector]
    ) -> List[FailurePrediction]:
        """Predict failure probabilities for multiple feature vectors."""
        pass

    @abc.abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class GradientBoostingFailurePredictor(FailurePredictionModel):
    """Gradient boosting based failure predictor using sklearn."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def train(self, feature_vectors: List[FeatureVector]) -> None:
        """Train the gradient boosting model."""
        if not feature_vectors:
            raise ValueError("No feature vectors provided for training")

        # Extract features and labels
        X = []
        y = []
        self.feature_names = feature_vectors[0].feature_names()

        for fv in feature_vectors:
            if fv.label is None:
                continue
            X.append(fv.to_numpy())
            y.append(int(fv.label))

        if not X:
            raise ValueError("No labeled feature vectors found for training")

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Log performance
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        logger.info(
            f"GradientBoosting model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}"
        )

    def predict(self, feature_vector: FeatureVector) -> FailurePrediction:
        """Predict failure probability."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X = feature_vector.to_numpy().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0][
            1
        ]  # Probability of positive class
        prediction = self.model.predict(X_scaled)[0]

        return FailurePrediction(
            timestamp=datetime.now(),
            failure_probability=proba,
            failure_type="general_failure" if prediction else "no_failure",
            confidence_score=max(proba, 1 - proba),
            feature_importance=self.get_feature_importance(),
        )

    def predict_batch(
        self, feature_vectors: List[FeatureVector]
    ) -> List[FailurePrediction]:
        """Predict for multiple feature vectors."""
        return [self.predict(fv) for fv in feature_vectors]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if not self.is_trained:
            return {}

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class NeuralFailurePredictor(FailurePredictionModel):
    """Neural network based failure predictor using PyTorch."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.model = self._build_model(input_dim, hidden_dims)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def _build_model(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build the neural network model."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def train(
        self,
        feature_vectors: List[FeatureVector],
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        """Train the neural network model."""
        if not feature_vectors:
            raise ValueError("No feature vectors provided for training")

        # Extract features and labels
        X = []
        y = []
        self.feature_names = feature_vectors[0].feature_names()

        for fv in feature_vectors:
            if fv.label is None:
                continue
            X.append(fv.to_numpy())
            y.append(float(fv.label))

        if not X:
            raise ValueError("No labeled feature vectors found for training")

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create dataset
        dataset = FailureDataset(X_scaled, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}"
                )

        self.is_trained = True
        logger.info("Neural network model trained successfully")

    def predict(self, feature_vector: FeatureVector) -> FailurePrediction:
        """Predict failure probability using neural network."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            X = feature_vector.to_numpy().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)

            output = self.model(X_tensor)
            probability = output.item()

        return FailurePrediction(
            timestamp=datetime.now(),
            failure_probability=probability,
            failure_type="general_failure" if probability > 0.5 else "no_failure",
            confidence_score=max(probability, 1 - probability),
        )

    def predict_batch(
        self, feature_vectors: List[FeatureVector]
    ) -> List[FailurePrediction]:
        """Predict for multiple feature vectors."""
        return [self.predict(fv) for fv in feature_vectors]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance using integrated gradients approximation."""
        if not self.is_trained:
            return {}

        # Simple approximation using input gradients
        # In a production system, you'd use more sophisticated methods
        importance_scores = np.random.random(len(self.feature_names))  # Placeholder
        importance_scores = importance_scores / importance_scores.sum()

        return dict(zip(self.feature_names, importance_scores))


class FailureDataset(Dataset):
    """PyTorch dataset for failure prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PredictiveFailureDetector:
    """Main class integrating predictive failure detection with existing FailureDetector."""

    def __init__(
        self,
        failure_detector: FailureDetector,
        model: Optional[FailurePredictionModel] = None,
    ):
        self.failure_detector = failure_detector
        self.model = model or GradientBoostingFailurePredictor()
        self.feature_extractor = FailureFeatureExtractor()
        self.prediction_history: List[FailurePrediction] = []
        self.monitoring_active = False

    async def start_monitoring(self, monitoring_interval: float = 60.0) -> None:
        """Start continuous monitoring for potential failures."""
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                await self._perform_prediction_check()
                await asyncio.sleep(monitoring_interval)
            except Exception as e:
                logger.error(f"Error during predictive monitoring: {e}")
                await asyncio.sleep(monitoring_interval)

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False

    async def _perform_prediction_check(self) -> None:
        """Perform a single prediction check."""
        # Get current session data from failure detector
        session_data = await self._gather_session_data()

        # Extract features
        feature_vector = self.feature_extractor.extract_features(session_data)

        # Make prediction
        prediction = self.model.predict(feature_vector)
        self.prediction_history.append(prediction)

        # Check for high-risk predictions
        if prediction.failure_probability > 0.8:
            await self._handle_high_risk_prediction(prediction)

        # Log prediction
        logger.info(
            f"Failure prediction: {prediction.failure_type} "
            f"(prob: {prediction.failure_probability:.3f})"
        )

    async def _gather_session_data(self) -> Dict[str, Any]:
        """Gather current session data for feature extraction."""
        # This would integrate with the existing failure detector
        # and other system components to gather relevant metrics
        return {
            "confidence_history": [0.8, 0.7, 0.9, 0.6],  # Example data
            "response_times": [0.1, 0.2, 0.15, 0.3],
            "memory_usage": 0.6,
            "cpu_usage": 0.4,
            "error_rate": 0.02,
            "coordination_complexity": 0.3,
            "task_load": 0.7,
            "communication_frequency": 0.5,
            "uncertainty_level": 0.4,
            "decision_confidence": 0.75,
            "fallback_usage": 0.1,
            "session_id": f"session_{int(time.time())}",
        }

    async def _handle_high_risk_prediction(self, prediction: FailurePrediction) -> None:
        """Handle predictions indicating high failure risk."""
        logger.warning(f"High failure risk detected: {prediction}")

        # Could trigger preventive actions like:
        # - Increasing monitoring frequency
        # - Triggering self-healing mechanisms
        # - Alerting administrators
        # - Scaling up resources

    def get_recent_predictions(self, limit: int = 10) -> List[FailurePrediction]:
        """Get recent failure predictions."""
        return self.prediction_history[-limit:]

    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about predictions."""
        if not self.prediction_history:
            return {}

        probabilities = [p.failure_probability for p in self.prediction_history]
        return {
            "total_predictions": len(self.prediction_history),
            "avg_failure_probability": np.mean(probabilities),
            "max_failure_probability": np.max(probabilities),
            "high_risk_predictions": sum(
                1 for p in self.prediction_history if p.failure_probability > 0.8
            ),
        }


class FailureTrainingPipeline:
    """Pipeline for training and updating failure prediction models."""

    def __init__(self, model: FailurePredictionModel):
        self.model = model
        self.training_data: List[FeatureVector] = []
        self.last_training_time: Optional[datetime] = None

    def add_training_sample(self, feature_vector: FeatureVector, label: bool) -> None:
        """Add a labeled sample to the training dataset."""
        feature_vector.label = label
        self.training_data.append(feature_vector)

    def add_training_samples(
        self, feature_vectors: List[FeatureVector], labels: List[bool]
    ) -> None:
        """Add multiple labeled samples."""
        if len(feature_vectors) != len(labels):
            raise ValueError("Number of feature vectors must match number of labels")

        for fv, label in zip(feature_vectors, labels):
            self.add_training_sample(fv, label)

    def train_model(self, min_samples: int = 100) -> bool:
        """Train the model if sufficient data is available."""
        if len(self.training_data) < min_samples:
            logger.info(
                f"Insufficient training data: {len(self.training_data)}/{min_samples}"
            )
            return False

        try:
            self.model.train(self.training_data)
            self.last_training_time = datetime.now()
            logger.info(
                f"Model trained successfully on {len(self.training_data)} samples"
            )
            return True
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def should_retrain(
        self, max_age_hours: int = 24, min_new_samples: int = 50
    ) -> bool:
        """Check if the model should be retrained."""
        if self.last_training_time is None:
            return True

        time_since_training = datetime.now() - self.last_training_time

        # Retrain if too old or if we have new samples
        return (
            time_since_training.total_seconds() > max_age_hours * 3600
            or len(self.training_data) >= min_new_samples
        )

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training data."""
        if not self.training_data:
            return {"total_samples": 0}

        labels = [fv.label for fv in self.training_data if fv.label is not None]
        positive_samples = sum(labels)
        negative_samples = len(labels) - positive_samples

        return {
            "total_samples": len(self.training_data),
            "labeled_samples": len(labels),
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "positive_ratio": positive_samples / len(labels) if labels else 0,
            "last_training_time": self.last_training_time.isoformat()
            if self.last_training_time
            else None,
        }
