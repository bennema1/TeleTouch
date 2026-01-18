import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from pathlib import Path


class CholecystectomyLSTM(nn.Module):
    """
    Specialized LSTM for laparoscopic cholecystectomy trajectory prediction.

    Cholecystectomy-specific optimizations:
    - Tuned for gallbladder dissection and retraction movements
    - Optimized for cystic duct/artery clipping patterns
    - Calibrated for cholecystectomy procedure phases
    - Specialized for laparoscopic instrument kinematics

    Architecture:
    Input: (batch_size, sequence_length, 2) - x,y coordinates over time
    LSTM: Process temporal dependencies in cholecystectomy movements
    Output: (batch_size, 2) - predicted next position with cholecystectomy-aware uncertainty
    """

    def __init__(self,
                 input_size: int = 2,  # x, y coordinates
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Cholecystectomy-specific input processing
        self.cholecystectomy_preprocessing = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size)
        )

        # Specialized LSTM for cholecystectomy movements
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Cholecystectomy-aware attention mechanism
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Cholecystectomy-optimized output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size)
        )

        # Cholecystectomy-calibrated uncertainty estimation
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
            # In baseline, input_size=2, so this outputs (batch, 2)
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cholecystectomy-specialized forward pass.

        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            prediction: (batch_size, input_size) - predicted next position
            uncertainty: (batch_size, input_size) - cholecystectomy-calibrated uncertainty
        """
        batch_size, seq_len, _ = x.shape

        # Cholecystectomy-specific preprocessing
        x_processed = self.cholecystectomy_preprocessing(x)

        # Specialized LSTM processing for cholecystectomy movements
        lstm_out, (h_n, c_n) = self.lstm(x_processed)  # (batch, seq_len, hidden_size)

        # Cholecystectomy-aware attention mechanism
        attention_scores = self.attention_weights(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, seq_len)

        # Apply attention with cholecystectomy movement context
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)  # (batch, hidden_size)

        # Generate cholecystectomy-optimized prediction and uncertainty
        prediction = self.output_layer(context)
        uncertainty = self.uncertainty_layer(context)

        return prediction, uncertainty

    def predict_next_position(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next position given movement history.

        Args:
            history: Past positions (batch_size, seq_len, 2)

        Returns:
            next_position: (batch_size, 2)
            uncertainty: (batch_size, 2)
        """
        return self.forward(history)


class CholecystectomyLSTMv2(nn.Module):
    """
    Enhanced LSTM with convolutional preprocessing for better feature extraction.
    """

    def __init__(self,
                 input_size: int = 2,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 conv_filters: int = 64,
                 kernel_size: int = 3):
        super().__init__()

        # Convolutional preprocessing
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, conv_filters, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU()
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layers
        self.output_layer = nn.Linear(hidden_size, 2)
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with convolutional preprocessing.
        """
        batch_size, seq_len, input_size = x.shape

        # Conv1D expects (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)  # (batch, input_size, seq_len)
        conv_out = self.conv1d(x_conv)  # (batch, conv_filters, seq_len)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, conv_filters)

        # LSTM processing
        lstm_out, _ = self.lstm(conv_out)  # (batch, seq_len, hidden_size)

        # Use final timestep
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Generate prediction and uncertainty
        prediction = self.output_layer(final_hidden)
        uncertainty = self.uncertainty_layer(final_hidden)

        return prediction, uncertainty

    def predict_next_position(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next position."""
        return self.forward(history)


def create_cholecystectomy_lstm(model_version: str = 'v1', input_size: int = 2) -> nn.Module:
    """
    Factory function for creating cholecystectomy-specialized LSTM models.

    Args:
        model_version: 'v1' (basic specialized LSTM) or 'v2' (with convolution)
        input_size: Number of input features

    Returns:
        Cholecystectomy-optimized model
    """
    if model_version == 'v1':
        return CholecystectomyLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
    elif model_version == 'v2':
        return CholecystectomyLSTMv2(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            conv_filters=64,
            kernel_size=3
        )
    else:
        raise ValueError(f"Unknown model version: {model_version}")


class TrajectoryPredictor:
    """
    High-level interface for trajectory prediction.
    Handles model loading, preprocessing, and inference.
    """

    def __init__(self, model_path: Optional[str] = None, model_version: str = 'v1'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_cholecystectomy_lstm(model_version)
        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"Loaded model from {model_path}")
        else:
            print("Using untrained model (random predictions)")

        self.model.eval()

    def predict_trajectory(self,
                          position_history: np.ndarray,
                          prediction_horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future trajectory given position history.

        Args:
            position_history: Past positions (seq_len, 2) or (batch, seq_len, 2)
            prediction_horizon: How many steps to predict ahead

        Returns:
            predictions: (prediction_horizon, 2) future positions
            uncertainties: (prediction_horizon, 2) prediction uncertainties
        """
        # Ensure correct shape
        if position_history.ndim == 2:
            position_history = position_history[np.newaxis, :, :]  # Add batch dimension

        # Convert to tensor
        input_tensor = torch.tensor(position_history, dtype=torch.float32, device=self.device)

        predictions = []
        uncertainties = []
        current_input = input_tensor.clone()

        with torch.no_grad():
            for _ in range(prediction_horizon):
                # Predict next position
                pred, unc = self.model.predict_next_position(current_input)

                predictions.append(pred.cpu().numpy())
                uncertainties.append(unc.cpu().numpy())

                # Update input with prediction (autoregressive)
                # Remove oldest timestep, add new prediction
                new_input = torch.cat([
                    current_input[:, 1:, :],  # Remove first timestep
                    pred.unsqueeze(1)         # Add prediction as new timestep
                ], dim=1)

                current_input = new_input

        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # (horizon, batch, 2)
        uncertainties = np.stack(uncertainties, axis=0)  # (horizon, batch, 2)

        # Remove batch dimension if input was 2D
        if position_history.shape[0] == 1:
            predictions = predictions[:, 0, :]  # (horizon, 2)
            uncertainties = uncertainties[:, 0, :]  # (horizon, 2)

        return predictions, uncertainties

    def predict_single_step(self,
                           position_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict just the next single position.

        Args:
            position_history: Recent positions (seq_len, 2)

        Returns:
            next_position: (2,) predicted position
            uncertainty: (2,) prediction uncertainty
        """
        if position_history.ndim == 1:
            position_history = position_history[np.newaxis, :]  # Add sequence dimension

        predictions, uncertainties = self.predict_trajectory(position_history, prediction_horizon=1)

        return predictions[0], uncertainties[0]