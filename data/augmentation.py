"""
Data augmentation and feature extraction for surgical trajectories.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import random


class SurgicalDataAugmenter:
    """
    Applies surgical-aware augmentations to trajectory data.
    """

    def __init__(self, time_warp_sigma: float = 0.1, spatial_scale_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.01, mirror_prob: float = 0.3):
        self.time_warp_sigma = time_warp_sigma
        self.spatial_scale_range = spatial_scale_range
        self.noise_std = noise_std
        self.mirror_prob = mirror_prob

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to a batch of data."""
        inputs = batch['input']
        # Apply augmentations to each trajectory in the batch
        augmented_inputs = []
        for trajectory in inputs:
            augmented = self(trajectory.numpy() if torch.is_tensor(trajectory) else trajectory)
            augmented_inputs.append(torch.tensor(augmented, dtype=torch.float32))

        batch['input'] = torch.stack(augmented_inputs)
        return batch

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a trajectory.

        Args:
            trajectory: (seq_len, 2) trajectory

        Returns:
            Augmented trajectory
        """
        augmented = trajectory.copy()

        # Random time warping
        if random.random() < 0.5:
            augmented = self._time_warp(augmented)

        # Random spatial scaling
        if random.random() < 0.5:
            augmented = self._spatial_scale(augmented)

        # Add noise
        if random.random() < 0.5:
            augmented = self._add_noise(augmented)

        # Random mirroring
        if random.random() < self.mirror_prob:
            augmented = self._mirror_trajectory(augmented)

        return augmented

    def _time_warp(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply time warping augmentation."""
        seq_len = len(trajectory)
        # Create smooth time warping function
        warp_factors = np.random.normal(1.0, self.time_warp_sigma, seq_len)
        warp_factors = np.clip(warp_factors, 0.5, 1.5)  # Reasonable bounds

        # Apply cumulative warping
        indices = np.cumsum(warp_factors)
        indices = (indices - indices.min()) / (indices.max() - indices.min()) * (seq_len - 1)

        # Interpolate
        warped = np.zeros_like(trajectory)
        for i in range(2):  # x, y coordinates
            warped[:, i] = np.interp(np.arange(seq_len), indices, trajectory[:, i])

        return warped

    def _spatial_scale(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply spatial scaling augmentation."""
        scale_factor = random.uniform(*self.spatial_scale_range)
        center = trajectory.mean(axis=0)

        # Scale around center
        scaled = center + (trajectory - center) * scale_factor
        return scaled

    def _add_noise(self, trajectory: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, trajectory.shape)
        return trajectory + noise

    def _mirror_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Mirror trajectory horizontally or vertically."""
        if random.random() < 0.5:
            # Horizontal mirror
            return np.array([[1 - x, y] for x, y in trajectory])
        else:
            # Vertical mirror
            return np.array([[x, 1 - y] for x, y in trajectory])


class TemporalFeatureExtractor:
    """
    Extracts temporal features from surgical trajectories.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def __call__(self, trajectory: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract temporal features.

        Args:
            trajectory: (seq_len, 2) trajectory

        Returns:
            Dictionary of feature arrays
        """
        positions = trajectory[:, :2]

        features = {}

        # Velocity
        velocity = np.gradient(positions, axis=0)
        features['velocity'] = velocity

        # Acceleration
        acceleration = np.gradient(velocity, axis=0)
        features['acceleration'] = acceleration

        # Speed
        speed = np.linalg.norm(velocity, axis=1)
        features['speed'] = speed

        # Curvature (simplified)
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**(3/2)
        features['curvature'] = curvature

        # Jerk (rate of change of acceleration)
        jerk = np.linalg.norm(np.gradient(acceleration, axis=0), axis=1)
        features['jerk'] = jerk

        # Direction changes
        direction = np.arctan2(dy, dx)
        direction_changes = np.abs(np.gradient(direction))
        features['direction_changes'] = direction_changes

        return features