"""
Predictor Interface - Predicts future instrument position

This file contains:
1. DummyPredictor - Simple velocity extrapolation (works immediately)
2. LSTMPredictor - Real AI model (swap in when Person B is ready)

The interface is the same, so the demo code doesn't need to change.
"""

from typing import List, Tuple, Optional, Protocol
from abc import ABC, abstractmethod
import numpy as np


class PredictorInterface(ABC):
    """Abstract interface that all predictors must implement."""
    
    @abstractmethod
    def predict(self, positions: List[Tuple[float, float]], 
                steps_ahead: int = 15) -> Tuple[float, float]:
        """
        Predict future position.
        
        Args:
            positions: List of recent (x, y) positions, oldest first
            steps_ahead: How many frames ahead to predict (15 = 500ms at 30fps)
            
        Returns:
            Predicted (x, y) position
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return predictor name for display."""
        pass


class DummyPredictor(PredictorInterface):
    """
    Simple velocity-based prediction.
    Uses linear extrapolation with optional acceleration.
    
    This is your fallback - works without any training!
    """
    
    def __init__(self, use_acceleration: bool = True):
        """
        Args:
            use_acceleration: If True, account for acceleration in prediction
        """
        self.use_acceleration = use_acceleration
    
    def predict(self, positions: List[Tuple[float, float]], 
                steps_ahead: int = 15) -> Tuple[float, float]:
        """
        Predict using linear/quadratic extrapolation.
        """
        if len(positions) < 2:
            # Not enough data, return last known position
            return positions[-1] if positions else (0.5, 0.5)
        
        # Get current position and velocity
        curr = np.array(positions[-1])
        prev = np.array(positions[-2])
        velocity = curr - prev
        
        if self.use_acceleration and len(positions) >= 3:
            # Calculate acceleration
            prev2 = np.array(positions[-3])
            prev_velocity = prev - prev2
            acceleration = velocity - prev_velocity
            
            # Quadratic extrapolation: p = p0 + v*t + 0.5*a*t^2
            t = steps_ahead
            predicted = curr + velocity * t + 0.5 * acceleration * t * t
        else:
            # Linear extrapolation: p = p0 + v*t
            predicted = curr + velocity * steps_ahead
        
        # Clamp to valid range [0, 1]
        predicted = np.clip(predicted, 0.0, 1.0)
        
        return (float(predicted[0]), float(predicted[1]))
    
    def get_name(self) -> str:
        if self.use_acceleration:
            return "Quadratic Extrapolation"
        return "Linear Extrapolation"


class LSTMPredictor(PredictorInterface):
    """
    Real LSTM-based predictor.
    
    TODO: Person B will implement this!
    For now, this is a placeholder that falls back to DummyPredictor.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to trained PyTorch model (.pth file)
        """
        self.model_path = model_path
        self.model = None
        self.fallback = DummyPredictor()
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load the trained LSTM model."""
        try:
            import torch
            # TODO: Person B - implement model loading
            # self.model = YourLSTMModel()
            # self.model.load_state_dict(torch.load(self.model_path))
            # self.model.eval()
            print(f"[LSTMPredictor] Would load model from: {self.model_path}")
            print("[LSTMPredictor] Model not implemented yet, using fallback")
        except Exception as e:
            print(f"[LSTMPredictor] Failed to load model: {e}")
            print("[LSTMPredictor] Using fallback predictor")
    
    def predict(self, positions: List[Tuple[float, float]], 
                steps_ahead: int = 15) -> Tuple[float, float]:
        """
        Predict using LSTM model, or fallback if not loaded.
        """
        if self.model is None:
            return self.fallback.predict(positions, steps_ahead)
        
        # TODO: Person B - implement LSTM inference
        # with torch.no_grad():
        #     input_tensor = self._prepare_input(positions)
        #     output = self.model(input_tensor)
        #     return self._process_output(output)
        
        return self.fallback.predict(positions, steps_ahead)
    
    def get_name(self) -> str:
        if self.model is not None:
            return "LSTM Neural Network"
        return f"Fallback ({self.fallback.get_name()})"


# Factory function to get the right predictor
def create_predictor(model_path: Optional[str] = None) -> PredictorInterface:
    """
    Create the appropriate predictor.
    
    Args:
        model_path: Path to LSTM model, or None for dummy predictor
        
    Returns:
        A predictor instance
    """
    if model_path:
        return LSTMPredictor(model_path)
    return DummyPredictor(use_acceleration=True)


# Quick test
if __name__ == "__main__":
    print("Testing Predictors...")
    
    # Generate a curved trajectory
    positions = []
    for i in range(15):
        angle = i * 0.15
        x = 0.5 + 0.1 * np.cos(angle)
        y = 0.5 + 0.08 * np.sin(angle)
        positions.append((x, y))
    
    print(f"Input: {len(positions)} positions")
    print(f"Last position: ({positions[-1][0]:.3f}, {positions[-1][1]:.3f})")
    
    # Test dummy predictor
    dummy = DummyPredictor()
    pred = dummy.predict(positions, steps_ahead=15)
    print(f"\nDummy prediction (15 steps): ({pred[0]:.3f}, {pred[1]:.3f})")
    print(f"Predictor name: {dummy.get_name()}")
    
    # Test LSTM predictor (will use fallback)
    lstm = LSTMPredictor(model_path=None)
    pred2 = lstm.predict(positions, steps_ahead=15)
    print(f"\nLSTM prediction (15 steps): ({pred2[0]:.3f}, {pred2[1]:.3f})")
    print(f"Predictor name: {lstm.get_name()}")
    
    # Test factory
    predictor = create_predictor()
    print(f"\nFactory created: {predictor.get_name()}")
    
    print("\n✓ Predictor test complete!")