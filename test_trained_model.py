"""
Test the trained CholecystectomyLSTM model.
"""

import torch
import numpy as np
from models.surgical_ltsm import TrajectoryPredictor

def test_model():
    print("Testing trained CholecystectomyLSTM model...")

    # Load the trained model
    model_path = "checkpoints/surgical_lstm_v1_20260117_182335/best_model.pth"
    predictor = TrajectoryPredictor(model_path=model_path, model_version='v1')

    # Create a simple test trajectory (similar to training data)
    t = np.linspace(0, 2*np.pi, 20)
    test_trajectory = np.column_stack([
        0.5 + 0.2 * np.cos(t),
        0.5 + 0.15 * np.sin(t)
    ])

    print(f"Test trajectory shape: {test_trajectory.shape}")
    print(f"First few points: {test_trajectory[:5]}")

    # Make prediction
    predictions, uncertainties = predictor.predict_trajectory(
        position_history=test_trajectory,
        prediction_horizon=5
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")
    print(f"Uncertainties:\n{uncertainties}")

    # Calculate some metrics
    if len(predictions) > 1:
        # Simple smoothness metric (lower is smoother)
        pred_diff = np.diff(predictions, axis=0)
        smoothness = np.mean(np.abs(pred_diff))
        print(f"Smoothness metric: {smoothness:.4f}")

    # Test inference speed
    import time
    start_time = time.time()
    for _ in range(10):
        _, _ = predictor.predict_trajectory(test_trajectory, prediction_horizon=1)
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    print(f"Average inference time: {avg_time:.4f}s")
    print("Model test complete!")

if __name__ == "__main__":
    test_model()