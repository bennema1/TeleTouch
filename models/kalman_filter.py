"""
Kalman Filter Models for Surgical Trajectory Prediction

Implements Kalman filtering for robust trajectory prediction with
adaptive noise estimation for surgical instrument tracking.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import warnings


class SurgicalKalmanFilter:
    """
    Kalman filter specialized for surgical instrument trajectory prediction.

    Uses constant velocity model with process and measurement noise
    tuned for laparoscopic instrument movements.
    """

    def __init__(self,
                 dt: float = 1.0/30.0,  # 30 FPS
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Initialize Kalman filter for 2D position tracking.

        Args:
            dt: Time step (seconds)
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        self.dt = dt
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy], Measurement: [x, y]

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ])

        # Measurement matrix (only position measurements)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        # Process noise (tuned for surgical movements)
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise)

        # Measurement noise
        self.kf.R = np.eye(2) * measurement_noise

        # Initial state covariance
        self.kf.P = np.eye(4) * 1.0

        self.initialized = False
        self.last_prediction_time = 0
        self.confidence = 0.5

    def initialize(self, initial_position: Tuple[float, float]) -> None:
        """
        Initialize filter with first position measurement.

        Args:
            initial_position: (x, y) starting position
        """
        x, y = initial_position
        self.kf.x = np.array([x, 0, y, 0])  # [x, vx, y, vy]
        self.initialized = True
        self.confidence = 0.7

    def predict(self, positions: List[Tuple[float, float]],
                steps_ahead: int = 1) -> Tuple[float, float]:
        """
        Predict future position using Kalman filter.

        Args:
            positions: Recent position history
            steps_ahead: Steps to predict ahead

        Returns:
            Predicted (x, y) position
        """
        if not positions:
            return (0.5, 0.5)

        # Initialize if needed
        if not self.initialized:
            self.initialize(positions[-1])

        # Update with latest measurements
        for pos in positions[-5:]:  # Use last 5 measurements
            self.kf.update(np.array(pos))

        # Predict steps ahead
        predicted_state = self.kf.x.copy()

        for _ in range(steps_ahead):
            # Predict one step
            predicted_state = self.kf.F @ predicted_state

        # Extract position
        x_pred, y_pred = predicted_state[0], predicted_state[2]

        # Update confidence based on state covariance
        position_covariance = self.kf.P[[0, 2]][:, [0, 2]]
        uncertainty = np.trace(position_covariance)
        self.confidence = max(0.1, min(1.0, 1.0 / (1.0 + uncertainty)))

        # Clamp to valid range
        x_pred = np.clip(x_pred, 0.0, 1.0)
        y_pred = np.clip(y_pred, 0.0, 1.0)

        return (float(x_pred), float(y_pred))

    def get_confidence(self) -> float:
        """Get current prediction confidence."""
        return self.confidence

    def reset(self) -> None:
        """Reset filter state."""
        self.initialized = False
        self.kf.x = np.zeros(4)
        self.kf.P = np.eye(4) * 1.0
        self.confidence = 0.5


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman filter that adjusts noise parameters based on
    movement characteristics and prediction errors.
    """

    def __init__(self,
                 dt: float = 1.0/30.0,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive Kalman filter.

        Args:
            dt: Time step
            adaptation_rate: How quickly to adapt noise parameters
        """
        self.base_filter = SurgicalKalmanFilter(dt=dt)
        self.adaptation_rate = adaptation_rate

        # Adaptive parameters
        self.process_noise_scale = 1.0
        self.measurement_noise_scale = 1.0

        # Error tracking for adaptation
        self.prediction_errors = []
        self.max_error_history = 20

        # Movement pattern detection
        self.movement_smoothness = 1.0
        self.last_velocities = []

    def _detect_movement_pattern(self, positions: List[Tuple[float, float]]) -> float:
        """
        Detect movement smoothness to adapt filter parameters.

        Returns:
            Smoothness score (0 = jerky, 1 = smooth)
        """
        if len(positions) < 3:
            return 0.5

        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)

        if len(velocities) < 2:
            return 0.5

        # Calculate velocity changes (jerk)
        velocity_changes = np.abs(np.diff(velocities))

        # Smoothness = 1 / (1 + average_jerk)
        avg_jerk = np.mean(velocity_changes)
        smoothness = 1.0 / (1.0 + avg_jerk)

        return smoothness

    def _adapt_noise_parameters(self, positions: List[Tuple[float, float]]) -> None:
        """
        Adapt process and measurement noise based on movement patterns.
        """
        if len(positions) < 3:
            return

        # Detect movement smoothness
        smoothness = self._detect_movement_pattern(positions)

        # Adapt process noise: higher for jerky movements
        if smoothness < 0.3:  # Jerky movement
            self.process_noise_scale = min(2.0, self.process_noise_scale * 1.2)
        elif smoothness > 0.7:  # Smooth movement
            self.process_noise_scale = max(0.5, self.process_noise_scale * 0.9)

        # Adapt measurement noise based on prediction errors
        if self.prediction_errors:
            avg_error = np.mean(self.prediction_errors[-5:])
            if avg_error > 0.05:  # High error
                self.measurement_noise_scale = min(2.0, self.measurement_noise_scale * 1.1)
            elif avg_error < 0.02:  # Low error
                self.measurement_noise_scale = max(0.5, self.measurement_noise_scale * 0.95)

    def predict(self, positions: List[Tuple[float, float]],
                steps_ahead: int = 1) -> Tuple[float, float]:
        """
        Adaptive prediction with parameter adjustment.
        """
        if not positions:
            return (0.5, 0.5)

        # Adapt parameters based on movement
        self._adapt_noise_parameters(positions)

        # Update base filter with adapted parameters
        self.base_filter.kf.Q = Q_discrete_white_noise(
            dim=2, dt=self.base_filter.dt,
            var=0.01 * self.process_noise_scale
        )
        self.base_filter.kf.R = np.eye(2) * (0.1 * self.measurement_noise_scale)

        # Make prediction
        prediction = self.base_filter.predict(positions, steps_ahead)

        return prediction

    def get_confidence(self) -> float:
        """Get current prediction confidence."""
        return self.base_filter.get_confidence()

    def update_error(self, true_position: Tuple[float, float],
                     predicted_position: Tuple[float, float]) -> None:
        """
        Update error history for adaptation.

        Args:
            true_position: Actual position
            predicted_position: Predicted position
        """
        error = np.linalg.norm(np.array(true_position) - np.array(predicted_position))
        self.prediction_errors.append(error)

        # Keep limited history
        if len(self.prediction_errors) > self.max_error_history:
            self.prediction_errors.pop(0)

    def reset(self) -> None:
        """Reset filter and adaptation state."""
        self.base_filter.reset()
        self.prediction_errors.clear()
        self.process_noise_scale = 1.0
        self.measurement_noise_scale = 1.0
        self.movement_smoothness = 1.0
        self.last_velocities.clear()


class VelocityExtrapolation:
    """
    Simple velocity-based predictor as baseline/fallback.
    """

    def __init__(self, use_acceleration: bool = True):
        """
        Initialize velocity extrapolator.

        Args:
            use_acceleration: Whether to use acceleration in prediction
        """
        self.use_acceleration = use_acceleration
        self.confidence = 0.6

    def predict(self, positions: List[Tuple[float, float]],
                steps_ahead: int = 1) -> Tuple[float, float]:
        """
        Predict using velocity/acceleration extrapolation.
        """
        if len(positions) < 2:
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

            # Quadratic extrapolation
            predicted = curr + velocity * steps_ahead + 0.5 * acceleration * steps_ahead**2
        else:
            # Linear extrapolation
            predicted = curr + velocity * steps_ahead

        # Clamp to valid range
        predicted = np.clip(predicted, 0.0, 1.0)

        return tuple(predicted)

    def get_confidence(self) -> float:
        """Get prediction confidence."""
        return self.confidence


# Quick test functions
if __name__ == "__main__":
    print("Testing Kalman Filter models...")

    # Create sample trajectory with some noise
    t = np.linspace(0, 2*np.pi, 20)
    clean_trajectory = np.column_stack([
        0.5 + 0.2 * np.cos(t),
        0.5 + 0.15 * np.sin(t)
    ])

    # Add noise
    np.random.seed(42)
    noisy_trajectory = clean_trajectory + np.random.normal(0, 0.02, clean_trajectory.shape)

    positions = [tuple(point) for point in noisy_trajectory]

    print(f"Input trajectory length: {len(positions)}")

    # Test SurgicalKalmanFilter
    kf = SurgicalKalmanFilter()
    pred_kf = kf.predict(positions, steps_ahead=5)
    print(f"KalmanFilter prediction: {pred_kf}, confidence: {kf.get_confidence():.3f}")

    # Test AdaptiveKalmanFilter
    akf = AdaptiveKalmanFilter()
    pred_akf = akf.predict(positions, steps_ahead=5)
    print(f"AdaptiveKalmanFilter prediction: {pred_akf}, confidence: {akf.get_confidence():.3f}")

    # Test VelocityExtrapolation
    ve = VelocityExtrapolation()
    pred_ve = ve.predict(positions, steps_ahead=5)
    print(f"VelocityExtrapolation prediction: {pred_ve}, confidence: {ve.get_confidence():.3f}")

    print("✓ Kalman Filter models test complete!")