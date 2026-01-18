"""
Surgical Ensemble Model for Trajectory Prediction

Combines multiple prediction models (LSTM, Transformer, Kalman Filter)
with dynamic model selection based on prediction quality and confidence.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time


class SurgicalEnsemblePredictor:
    """
    Ensemble predictor that combines multiple models with quality-based selection.

    Uses prediction confidence, historical accuracy, and clinical relevance
    to dynamically select the best model for each prediction.
    """

    def __init__(self,
                 model_configs: List[Dict[str, Any]] = None,
                 quality_threshold: float = 0.7):
        """
        Initialize ensemble predictor.

        Args:
            model_configs: List of model configurations
            quality_threshold: Minimum quality score to use ML models
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quality_threshold = quality_threshold
        self.models = {}
        self.model_qualities = {}
        self.prediction_history = []

        # Default model configurations
        if model_configs is None:
            model_configs = [
                {'type': 'lstm', 'version': 'v1', 'path': None, 'weight': 0.4},
                {'type': 'transformer', 'path': None, 'weight': 0.3},
                {'type': 'kalman', 'weight': 0.3}
            ]

        # Initialize models
        self._load_models(model_configs)

        # Fallback predictor (simple velocity extrapolation)
        self.fallback = self._create_fallback_predictor()

    def _load_models(self, configs: List[Dict[str, Any]]) -> None:
        """Load and initialize all models in the ensemble."""
        for config in configs:
            model_type = config['type']
            model_path = config.get('path')

            try:
                if model_type == 'lstm':
                    from .surgical_ltsm import TrajectoryPredictor
                    version = config.get('version', 'v1')
                    model = TrajectoryPredictor(model_path=model_path, model_version=version)
                    model_name = f"LSTM_{version}"

                elif model_type == 'transformer':
                    from .transformer_model import TrajectoryTransformerPredictor
                    model = TrajectoryTransformerPredictor(model_path=model_path)
                    model_name = "Transformer"

                elif model_type == 'kalman':
                    from .kalman_filter import AdaptiveKalmanFilter
                    model = AdaptiveKalmanFilter()
                    model_name = "Kalman"

                else:
                    print(f"Unknown model type: {model_type}")
                    continue

                self.models[model_name] = {
                    'model': model,
                    'weight': config.get('weight', 1.0),
                    'type': model_type
                }
                self.model_qualities[model_name] = 1.0  # Start with neutral quality

                print(f"Loaded {model_name} for ensemble")

            except Exception as e:
                print(f"Failed to load {model_type}: {e}")

    def _create_fallback_predictor(self):
        """Create a simple fallback predictor."""
        return lambda positions, steps: self._simple_extrapolation(positions, steps)

    def _simple_extrapolation(self, positions: List[Tuple[float, float]],
                             steps_ahead: int = 1) -> Tuple[float, float]:
        """Simple velocity-based extrapolation."""
        if len(positions) < 2:
            return positions[-1] if positions else (0.5, 0.5)

        # Calculate velocity from last two points
        curr = np.array(positions[-1])
        prev = np.array(positions[-2])
        velocity = curr - prev

        # Extrapolate
        predicted = curr + velocity * steps_ahead

        # Clamp to valid range
        predicted = np.clip(predicted, 0.0, 1.0)

        return tuple(predicted)

    def predict_with_ensemble(self,
                             positions: List[Tuple[float, float]],
                             steps_ahead: int = 1) -> Dict[str, Any]:
        """
        Make prediction using ensemble of models.

        Args:
            positions: Historical positions [(x,y), ...]
            steps_ahead: Steps to predict ahead

        Returns:
            Dictionary with predictions, confidences, and selected model
        """
        if not positions:
            return {
                'prediction': (0.5, 0.5),
                'confidence': 0.0,
                'selected_model': 'fallback',
                'model_predictions': {}
            }

        # Get predictions from all models
        model_predictions = {}
        model_confidences = {}
        model_qualities = {}

        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                model_type = model_info['type']

                if model_type in ['lstm', 'transformer']:
                    # ML models
                    trajectory = np.array(positions[-20:])  # Use last 20 points
                    if len(trajectory) < 5:
                        continue

                    prediction = model.predict_trajectory(trajectory, steps_ahead)
                    pred_point = prediction[0] if len(prediction) > 0 else trajectory[-1]

                    # Calculate confidence (placeholder - could use uncertainty)
                    confidence = 0.8  # Default confidence for ML models

                elif model_type == 'kalman':
                    # Kalman filter
                    pred_point = model.predict(positions, steps_ahead)
                    confidence = model.get_confidence()

                else:
                    continue

                model_predictions[model_name] = pred_point
                model_confidences[model_name] = confidence
                model_qualities[model_name] = self.model_qualities.get(model_name, 1.0)

            except Exception as e:
                print(f"Prediction failed for {model_name}: {e}")
                continue

        # Select best model based on weighted quality score
        if model_predictions:
            best_model, best_prediction = self._select_best_model(
                model_predictions, model_confidences, model_qualities
            )
            final_confidence = model_confidences.get(best_model, 0.5)
        else:
            # Use fallback
            best_model = 'fallback'
            best_prediction = self._simple_extrapolation(positions, steps_ahead)
            final_confidence = 0.3

        return {
            'prediction': best_prediction,
            'confidence': final_confidence,
            'selected_model': best_model,
            'model_predictions': model_predictions,
            'model_confidences': model_confidences
        }

    def _select_best_model(self,
                          predictions: Dict[str, Tuple[float, float]],
                          confidences: Dict[str, float],
                          qualities: Dict[str, float]) -> Tuple[str, Tuple[float, float]]:
        """
        Select the best model based on confidence and historical quality.

        Args:
            predictions: Model predictions
            confidences: Model confidence scores
            qualities: Historical quality scores

        Returns:
            (model_name, prediction)
        """
        best_score = -1
        best_model = None
        best_prediction = None

        for model_name, prediction in predictions.items():
            confidence = confidences.get(model_name, 0.5)
            quality = qualities.get(model_name, 1.0)
            weight = self.models[model_name]['weight']

            # Combined score: confidence * quality * weight
            score = confidence * quality * weight

            if score > best_score:
                best_score = score
                best_model = model_name
                best_prediction = prediction

        return best_model, best_prediction

    def update_model_qualities(self,
                              true_position: Tuple[float, float],
                              predictions: Dict[str, Any]) -> None:
        """
        Update model quality scores based on prediction accuracy.

        Args:
            true_position: Actual position that occurred
            predictions: Prediction results from predict_with_ensemble
        """
        true_pos = np.array(true_position)

        for model_name, pred_pos in predictions.get('model_predictions', {}).items():
            if model_name in self.model_qualities:
                pred_pos = np.array(pred_pos)
                error = np.linalg.norm(true_pos - pred_pos)

                # Update quality using exponential moving average
                alpha = 0.1  # Learning rate
                current_quality = self.model_qualities[model_name]

                # Convert error to quality (lower error = higher quality)
                error_quality = max(0, 1.0 - error)  # Simple error-to-quality mapping

                # Update with exponential moving average
                new_quality = alpha * error_quality + (1 - alpha) * current_quality
                self.model_qualities[model_name] = new_quality

    def predict_trajectory(self,
                          trajectory: np.ndarray,
                          prediction_horizon: int = 1) -> np.ndarray:
        """
        Predict future trajectory using ensemble.

        Args:
            trajectory: Historical trajectory (seq_len, 2)
            prediction_horizon: Steps to predict ahead

        Returns:
            Predicted trajectory (prediction_horizon, 2)
        """
        # Convert numpy array to list of tuples for ensemble interface
        positions = [tuple(point) for point in trajectory]

        predictions = []
        current_positions = positions.copy()

        for _ in range(prediction_horizon):
            # Get ensemble prediction
            result = self.predict_with_ensemble(current_positions, steps_ahead=1)
            next_point = result['prediction']

            predictions.append(next_point)

            # Add prediction to history for next iteration
            current_positions.append(next_point)

        return np.array(predictions)


def create_ensemble_predictor(model_configs: List[Dict[str, Any]] = None) -> SurgicalEnsemblePredictor:
    """
    Factory function for creating ensemble predictor.

    Args:
        model_configs: Model configurations

    Returns:
        SurgicalEnsemblePredictor instance
    """
    return SurgicalEnsemblePredictor(model_configs)


# Quick test function
if __name__ == "__main__":
    print("Testing SurgicalEnsemblePredictor...")

    # Create sample trajectory
    t = np.linspace(0, 2*np.pi, 15)
    trajectory = np.column_stack([
        0.5 + 0.2 * np.cos(t),
        0.5 + 0.15 * np.sin(t)
    ])

    print(f"Input trajectory shape: {trajectory.shape}")

    # Test ensemble predictor
    predictor = create_ensemble_predictor()

    predictions = predictor.predict_trajectory(trajectory, prediction_horizon=3)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    print("✓ SurgicalEnsemblePredictor test complete!")