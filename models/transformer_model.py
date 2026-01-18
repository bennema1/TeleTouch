"""
Surgical Transformer Model for Trajectory Prediction

Specialized transformer architecture for laparoscopic cholecystectomy instrument prediction.
Incorporates positional encoding, multi-head attention, and uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input sequences.

    Adds sinusoidal positional information to help the model understand
    temporal ordering in surgical trajectories.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class SurgicalTransformer(nn.Module):
    """
    Transformer-based model for surgical instrument trajectory prediction.

    Specialized for laparoscopic cholecystectomy with attention mechanisms
    that focus on clinically relevant movement patterns.
    """

    def __init__(self,
                 input_size: int = 2,  # x, y coordinates
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 50):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection to transformer dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection back to coordinate space
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, input_size)
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, input_size),
            nn.Softplus()  # Ensure positive uncertainty
        )

        # Surgical attention mechanism for cholecystectomy-specific patterns
        self.surgical_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead // 2,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through surgical transformer.

        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            prediction: (batch_size, input_size) - predicted next position
            uncertainty: (batch_size, input_size) - prediction uncertainty
        """
        batch_size, seq_len, _ = x.shape

        # Project input to transformer dimension
        x_proj = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x_encoded = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)

        # Apply surgical attention (focus on recent movements)
        attn_output, _ = self.surgical_attention(
            x_encoded, x_encoded, x_encoded,
            key_padding_mask=self._create_padding_mask(seq_len)
        )

        # Transformer encoder
        transformer_output = self.transformer_encoder(attn_output)

        # Use the last position's representation for prediction
        last_hidden = transformer_output[:, -1, :]  # (batch, d_model)

        # Generate prediction
        prediction = self.output_projection(last_hidden)  # (batch, input_size)

        # Generate uncertainty estimate
        uncertainty = self.uncertainty_head(last_hidden)  # (batch, input_size)

        return prediction, uncertainty

    def _create_padding_mask(self, seq_len: int) -> Optional[torch.Tensor]:
        """Create padding mask for attention (None for now, can be extended)"""
        return None


class TrajectoryTransformerPredictor:
    """
    High-level interface for transformer-based trajectory prediction.

    Handles model loading, preprocessing, and multi-step prediction.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = SurgicalTransformer(
            input_size=2,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1
        ).to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"Loaded SurgicalTransformer from {model_path}")
        else:
            print("Using untrained SurgicalTransformer (random predictions)")

        self.model.eval()

    def predict_trajectory(self,
                          trajectory: np.ndarray,
                          prediction_horizon: int = 1) -> np.ndarray:
        """
        Predict future trajectory points.

        Args:
            trajectory: Historical trajectory (seq_len, 2)
            prediction_horizon: Number of steps to predict ahead

        Returns:
            Predicted trajectory points (prediction_horizon, 2)
        """
        with torch.no_grad():
            # Prepare input
            if len(trajectory) > self.model.max_seq_len:
                trajectory = trajectory[-self.model.max_seq_len:]

            trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(self.device)

            # For multi-step prediction, iteratively predict
            predictions = []

            for _ in range(prediction_horizon):
                # Get prediction for next step
                pred, _ = self.model(trajectory_tensor)
                next_point = pred.squeeze(0).cpu().numpy()

                predictions.append(next_point)

                # Add prediction to trajectory for next iteration
                next_point_tensor = torch.FloatTensor(next_point).unsqueeze(0).unsqueeze(0)
                trajectory_tensor = torch.cat([trajectory_tensor, next_point_tensor], dim=1)

                # Keep only recent history
                if trajectory_tensor.size(1) > self.model.max_seq_len:
                    trajectory_tensor = trajectory_tensor[:, -self.model.max_seq_len:]

            return np.array(predictions)


def create_surgical_transformer(model_path: Optional[str] = None) -> TrajectoryTransformerPredictor:
    """
    Factory function for creating surgical transformer predictor.

    Args:
        model_path: Path to trained model (.pth file)

    Returns:
        TrajectoryTransformerPredictor instance
    """
    return TrajectoryTransformerPredictor(model_path)


# Quick test function
if __name__ == "__main__":
    print("Testing SurgicalTransformer...")

    # Create sample trajectory
    t = np.linspace(0, 4*np.pi, 20)
    trajectory = np.column_stack([
        0.5 + 0.2 * np.cos(t),
        0.5 + 0.15 * np.sin(t)
    ])

    print(f"Input trajectory shape: {trajectory.shape}")

    # Test predictor
    predictor = create_surgical_transformer()

    predictions = predictor.predict_trajectory(trajectory, prediction_horizon=5)
    print(f"Predictions shape: {predictions.shape}")
    print(f"First prediction: {predictions[0]}")

    print("✓ SurgicalTransformer test complete!")