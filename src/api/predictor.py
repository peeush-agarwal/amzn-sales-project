"""Model predictor service for loading and running predictions."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd


class ModelPredictor:
    """Service for loading ML models and making predictions."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        transformer_path: Optional[Path] = None,
    ):
        """Initialize the predictor.

        Args:
            model_path: Path to the trained model file
            meta_path: Path to the model metadata JSON file
            transformer_path: Path to the data transformer file
        """
        self.model_path = model_path
        self.meta_path = meta_path
        self.transformer_path = transformer_path
        self.model: Optional[Any] = None
        self.transformer: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.load_time: Optional[float] = None
        self.prediction_count: int = 0
        self.total_prediction_time: float = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(self) -> None:
        """Load the trained model and metadata."""
        start_time = time.time()

        try:
            # Load model
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Model loaded successfully: {type(self.model).__name__}")

            # Load metadata
            if self.meta_path and self.meta_path.exists():
                with open(self.meta_path, "r") as f:
                    self.metadata = json.load(f)
                self.logger.info(
                    f"Metadata loaded: {self.metadata.get('best_name', 'unknown')}"
                )
            else:
                self.logger.warning(f"Metadata file not found at {self.meta_path}")
                self.metadata = {}

            if self.transformer_path and self.transformer_path.exists():
                self.logger.info(f"Loading transformer from {self.transformer_path}")
                self.transformer = joblib.load(self.transformer_path)
                self.logger.info("Transformer loaded successfully")
            else:
                self.logger.warning(
                    f"Transformer file not found at {self.transformer_path}. "
                    "Raw feature transformation will not be available."
                )
                self.transformer = None

            self.load_time = time.time() - start_time
            self.logger.info(f"Model loading completed in {self.load_time:.3f}s")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on input features.

        Args:
            features: Input feature array of shape (n_samples, n_features)

        Returns:
            Predictions array of shape (n_samples,)

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input validation fails
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Validate input
        self._validate_input(features)

        # Make prediction
        start_time = time.time()
        try:
            predictions = self.model.predict(features)
            prediction_time = time.time() - start_time

            # Update metrics
            self.prediction_count += len(features)
            self.total_prediction_time += prediction_time

            # Log prediction metrics
            avg_time = prediction_time / len(features) * 1000  # ms per sample
            self.logger.info(
                f"Prediction completed: {len(features)} samples in "
                f"{prediction_time:.3f}s ({avg_time:.2f}ms per sample)"
            )

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_single(self, features: list) -> float:
        """Make a prediction for a single sample.

        Args:
            features: List of feature values

        Returns:
            Predicted discount percentage
        """
        X = np.array([features])
        predictions = self.predict(X)
        return float(predictions[0])

    def predict_batch(self, samples: list) -> np.ndarray:
        """Make predictions for multiple samples.

        Args:
            samples: List of feature lists

        Returns:
            Array of predictions
        """
        X = np.array(samples)
        return self.predict(X)

    def predict_from_raw(self, raw_data: Dict[str, Any]) -> float:
        """Make a prediction from raw product data.

        Args:
            raw_data: Dictionary with raw product fields

        Returns:
            Predicted discount percentage

        Raises:
            RuntimeError: If transformer is not loaded
        """
        if self.transformer is None:
            raise RuntimeError("Transformer is not loaded. Cannot process raw data.")

        # Convert raw data dict to DataFrame
        df = pd.DataFrame([raw_data])

        # Transform using the loaded transformer
        try:
            X, _ = self.transformer.transform(df)
            predictions = self.predict(X)
            return float(predictions[0])
        except Exception as e:
            self.logger.error(f"Failed to transform raw data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to transform raw data: {str(e)}")

    def predict_batch_from_raw(self, raw_samples: list) -> np.ndarray:
        """Make predictions from multiple raw product data samples.

        Args:
            raw_samples: List of dictionaries with raw product fields

        Returns:
            Array of predictions

        Raises:
            RuntimeError: If transformer is not loaded
        """
        if self.transformer is None:
            raise RuntimeError("Transformer is not loaded. Cannot process raw data.")

        # Convert raw data list to DataFrame
        df = pd.DataFrame(raw_samples)

        # Transform using the loaded transformer
        try:
            X, _ = self.transformer.transform(df)
            return self.predict(X)
        except Exception as e:
            self.logger.error(f"Failed to transform raw data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to transform raw data: {str(e)}")

    def _validate_input(self, features: np.ndarray) -> None:
        """Validate input features.

        Args:
            features: Input feature array

        Raises:
            ValueError: If validation fails
        """
        # Check type
        if not isinstance(features, np.ndarray):
            raise ValueError(f"Features must be numpy array, got {type(features)}")

        # Check shape
        if features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array (n_samples, n_features), got shape {features.shape}"
            )

        # Check for NaN/Inf
        if np.any(np.isnan(features)):
            raise ValueError("Features contain NaN values")

        if np.any(np.isinf(features)):
            raise ValueError("Features contain infinite values")

        # Check feature count (if metadata available)
        if self.metadata and "n_features" in self.metadata:
            expected_features = self.metadata["n_features"]
            if features.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {features.shape[1]}"
                )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.

        Returns:
            Dictionary with model info
        """
        info = {
            "model_loaded": self.model is not None,
            "model_type": type(self.model).__name__ if self.model else None,
            "load_time_seconds": self.load_time,
            "prediction_count": self.prediction_count,
            "total_prediction_time_seconds": self.total_prediction_time,
        }

        # Add metadata
        if self.metadata:
            info["model_name"] = self.metadata.get("best_name")
            info["run_id"] = self.metadata.get("best_run_id")
            info["metrics"] = self.metadata.get("metrics", {})
            info["timestamp"] = self.metadata.get("timestamp")

        # Calculate average prediction time
        if self.prediction_count > 0:
            info["avg_prediction_time_ms"] = (
                self.total_prediction_time / self.prediction_count * 1000
            )

        return info

    def is_ready(self) -> bool:
        """Check if predictor is ready to make predictions.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
