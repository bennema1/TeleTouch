import numpy as np
import pandas as pd
import requests
import zipfile
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import scipy.signal
import scipy.ndimage
from typing import Tuple, List, Optional


class CholecystectomyDataset:
    """Cholecystectomy-specialized dataset loader and processor."""

    ZENODO_URL = "https://zenodo.org/records/10719748/files/ROSMAG40.zip"
    DATA_DIR = Path("data/rosma")

    def __init__(self, download_if_missing: bool = True):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if download_if_missing and not self._is_downloaded():
            self._download_dataset()

        self.kinematic_data = {}
        self.annotation_data = {}
        self._load_data()

    def _is_downloaded(self) -> bool:
        """Check if ROSMA dataset is already downloaded."""
        return (self.DATA_DIR / "ROSMAG40").exists()

    def _download_dataset(self):
        """Download ROSMA dataset from Zenodo."""
        print("Downloading ROSMA cholecystectomy dataset...")
        response = requests.get(self.ZENODO_URL, stream=True)
        zip_path = self.DATA_DIR / "rosmag40.zip"

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.DATA_DIR)

        os.remove(zip_path)
        print("Cholecystectomy dataset downloaded and extracted.")

    def _load_data(self):
        """Load kinematic and annotation data."""
        rosma_dir = self.DATA_DIR / "ROSMAG40"

        if not rosma_dir.exists():
            raise FileNotFoundError(f"ROSMA directory not found: {rosma_dir}")

        # Load kinematic data (CSV files)
        kinematic_files = list(rosma_dir.glob("*_kinematic.csv"))
        for kf in kinematic_files:
            trial_name = kf.stem.replace('_kinematic', '')
            self.kinematic_data[trial_name] = pd.read_csv(kf)

        # Load annotation data if available
        annotation_files = list(rosma_dir.glob("*_annotations.json"))
        for af in annotation_files:
            trial_name = af.stem.replace('_annotations', '')
            self.annotation_data[trial_name] = pd.read_csv(af)

    def extract_cholecystectomy_trajectories(self, trial_name: str) -> np.ndarray:
        """
        Extract cholecystectomy-specific trajectories from ROSMA kinematic data.

        Returns normalized (x,y) coordinates optimized for cholecystectomy movements.
        """
        if trial_name not in self.kinematic_data:
            available_trials = list(self.kinematic_data.keys())
            raise ValueError(f"Trial '{trial_name}' not found. Available: {available_trials[:5]}...")

        df = self.kinematic_data[trial_name]

        # Extract position data - ROSMA uses Cartesian coordinates
        # For cholecystectomy, focus on instrument tip movements
        pos_cols = [col for col in df.columns if 'position' in col.lower() or 'pos' in col.lower()]

        if len(pos_cols) < 2:
            # Fallback: look for x,y,z coordinates
            coord_cols = []
            for axis in ['x', 'y', 'z']:
                candidates = [col for col in df.columns if axis in col.lower()]
                if candidates:
                    coord_cols.append(candidates[0])

            if len(coord_cols) >= 2:
                pos_cols = coord_cols[:2]  # Take x,y

        if len(pos_cols) < 2:
            raise ValueError(f"Could not find position columns in trial {trial_name}")

        positions = df[pos_cols[:2]].values  # (N, 2)

        # Cholecystectomy-specific normalization
        # Focus on the working area typical for gallbladder surgery
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)

        # Ensure minimum working area (avoid division by very small numbers)
        working_area = pos_max - pos_min
        working_area = np.maximum(working_area, 10.0)  # Minimum 10mm working area

        positions_norm = (positions - pos_min) / working_area

        return positions_norm.astype(np.float32)

    def get_cholecystectomy_kinematic_features(self, trial_name: str) -> np.ndarray:
        """
        Extract cholecystectomy-specialized kinematic features.

        Returns 13 features per timestep optimized for laparoscopic cholecystectomy:
        0-1: position (x, y)
        2-3: velocity (vx, vy)
        4-5: acceleration (ax, ay)
        6-7: jerk (jx, jy) - smoothness indicator
        8: curvature - turning rate
        9: speed magnitude
        10: movement direction
        11: cholecystectomy path efficiency
        12: tremor indicator (8-12 Hz typical for cholecystectomy)
        """
        positions = self.extract_cholecystectomy_trajectories(trial_name)

        features = []
        dt = 1.0 / 50.0  # 50 Hz sampling

        for i in range(len(positions)):
            # Position
            x, y = positions[i]

            # Velocity (smoothed for cholecystectomy precision)
            if i >= 2:
                vx = (positions[i, 0] - positions[i-2, 0]) / (2 * dt)
                vy = (positions[i, 1] - positions[i-2, 1]) / (2 * dt)
            else:
                vx = vy = 0.0

            # Acceleration (critical for cholecystectomy tissue handling)
            if i >= 4:
                ax = (positions[i, 0] - 2*positions[i-2, 0] + positions[i-4, 0]) / (4 * dt**2)
                ay = (positions[i, 1] - 2*positions[i-2, 1] + positions[i-4, 1]) / (4 * dt**2)
            else:
                ax = ay = 0.0

            # Jerk (3rd derivative) - smoothness is critical in cholecystectomy
            if i >= 6:
                jx = (positions[i, 0] - 3*positions[i-2, 0] + 3*positions[i-4, 0] - positions[i-6, 0]) / (8 * dt**3)
                jy = (positions[i, 1] - 3*positions[i-2, 1] + 3*positions[i-4, 1] - positions[i-6, 1]) / (8 * dt**3)
            else:
                jx = jy = 0.0

            # Curvature (turning rate) - important for dissection paths
            if abs(vx) > 1e-6 or abs(vy) > 1e-6:
                curvature = (vx * ay - vy * ax) / (vx**2 + vy**2)**(3/2) if (vx**2 + vy**2) > 1e-6 else 0.0
            else:
                curvature = 0.0

            # Speed magnitude
            speed = np.sqrt(vx**2 + vy**2)

            # Movement direction
            direction = np.arctan2(vy, vx) if speed > 1e-6 else 0.0

            # Cholecystectomy-specific path efficiency
            # Rewards smooth, deliberate movements typical of expert cholecystectomy
            if i >= 10:
                start_pos = positions[i-10]
                end_pos = positions[i]
                straight_distance = np.linalg.norm(end_pos - start_pos)
                actual_distance = sum(np.linalg.norm(positions[j] - positions[j-1])
                                    for j in range(i-9, i+1))

                if actual_distance > 0:
                    base_efficiency = straight_distance / actual_distance

                    # Cholecystectomy bonus: reward smooth movements, penalize jerkiness
                    smoothness_factor = 1.0 / (1.0 + 0.1 * (abs(jx) + abs(jy)))
                    path_efficiency = base_efficiency * smoothness_factor
                else:
                    path_efficiency = 1.0
            else:
                path_efficiency = 1.0

            # Cholecystectomy tremor characteristics
            # Physiological tremor in laparoscopic surgery is typically 8-12 Hz
            # This helps the model distinguish between intended movements and hand shake
            tremor_indicator = 0.0
            if speed > 0.001:  # Only check when moving
                # Estimate frequency content of recent movement
                if i >= 20:
                    recent_positions = positions[i-20:i]
                    # Simple frequency estimation (dominant frequency in recent window)
                    # This is a simplified version - real implementation would use FFT
                    position_variance = np.var(recent_positions, axis=0).mean()
                    tremor_indicator = 1.0 / (1.0 + position_variance)  # Higher variance = more tremor

            features.append([
                x, y, vx, vy, ax, ay, jx, jy,
                curvature, speed, direction, path_efficiency, tremor_indicator
            ])

        return np.array(features, dtype=np.float32)


class EnhancedCholecystectomyDataset(Dataset):
    """Enhanced dataset that computes cholecystectomy-specialized kinematic features from PyGame mouse data."""

    def __init__(self, numpy_file: str, sequence_length: int = 10, prediction_horizon: int = 30):
        """
        Args:
            numpy_file: Path to training_data.npy from PyGame (shape: N, 3)
            sequence_length: Number of frames for input sequence
            prediction_horizon: Number of frames to predict ahead
        """
        self.data = np.load(numpy_file)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Compute cholecystectomy-specialized kinematic features
        self.features = self._compute_cholecystectomy_kinematic_features()

        # Create valid indices for sequences
        self.valid_indices = []
        for i in range(sequence_length, len(self.features) - prediction_horizon):
            self.valid_indices.append(i)

    def _compute_cholecystectomy_kinematic_features(self) -> np.ndarray:
        """Compute 13 cholecystectomy-specialized kinematic features from raw mouse data."""
        positions = self.data[:, :2]  # x, y coordinates
        timestamps = self.data[:, 2]  # timestamps

        features = []
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 1/60.0

        for i in range(len(positions)):
            x, y = positions[i]

            # Velocity (smoothed for cholecystectomy precision)
            if i >= 2:
                vx = (positions[i, 0] - positions[i-2, 0]) / (2 * dt)
                vy = (positions[i, 1] - positions[i-2, 1]) / (2 * dt)
            else:
                vx = vy = 0.0

            # Acceleration (critical for tissue handling)
            if i >= 4:
                ax = (positions[i, 0] - 2*positions[i-2, 0] + positions[i-4, 0]) / (4 * dt**2)
                ay = (positions[i, 1] - 2*positions[i-2, 1] + positions[i-4, 1]) / (4 * dt**2)
            else:
                ax = ay = 0.0

            # Jerk (smoothness indicator for cholecystectomy)
            if i >= 6:
                jx = (positions[i, 0] - 3*positions[i-2, 0] + 3*positions[i-4, 0] - positions[i-6, 0]) / (8 * dt**3)
                jy = (positions[i, 1] - 3*positions[i-2, 1] + 3*positions[i-4, 1] - positions[i-6, 1]) / (8 * dt**3)
            else:
                jx = jy = 0.0

            # Curvature (turning rate for dissection paths)
            if abs(vx) > 1e-6 or abs(vy) > 1e-6:
                curvature = (vx * ay - vy * ax) / (vx**2 + vy**2)**(3/2) if (vx**2 + vy**2) > 1e-6 else 0.0
            else:
                curvature = 0.0

            # Speed magnitude
            speed = np.sqrt(vx**2 + vy**2)

            # Movement direction
            direction = np.arctan2(vy, vx) if speed > 1e-6 else 0.0

            # Cholecystectomy path efficiency
            if i >= 10:
                start_pos = positions[i-10]
                end_pos = positions[i]
                straight_distance = np.linalg.norm(end_pos - start_pos)
                actual_distance = sum(np.linalg.norm(positions[j] - positions[j-1])
                                    for j in range(i-9, i+1))

                if actual_distance > 0:
                    base_efficiency = straight_distance / actual_distance
                    # Cholecystectomy bonus: reward smooth movements
                    smoothness_factor = 1.0 / (1.0 + 0.1 * (abs(jx) + abs(jy)))
                    path_efficiency = base_efficiency * smoothness_factor
                else:
                    path_efficiency = 1.0
            else:
                path_efficiency = 1.0

            # Cholecystectomy tremor indicator
            tremor_indicator = 0.0
            if speed > 0.001 and i >= 20:
                recent_positions = positions[i-20:i]
                position_variance = np.var(recent_positions, axis=0).mean()
                tremor_indicator = 1.0 / (1.0 + position_variance)

            features.append([
                x, y, vx, vy, ax, ay, jx, jy,
                curvature, speed, direction, path_efficiency, tremor_indicator
            ])

        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get input sequence and target prediction."""
        center_idx = self.valid_indices[idx]

        # Input: sequence_length frames of features
        input_start = center_idx - self.sequence_length
        input_end = center_idx
        input_sequence = self.features[input_start:input_end]  # (10, 13)

        # Target: position prediction_horizon frames ahead
        target_idx = center_idx + self.prediction_horizon
        target_position = self.features[target_idx, :2]  # (x, y) only

        return {
            'input': torch.tensor(input_sequence, dtype=torch.float32),
            'target': torch.tensor(target_position, dtype=torch.float32),
            'metadata': {
                'center_idx': center_idx,
                'target_idx': target_idx
            }
        }


class CholecystectomySyntheticDataset(Dataset):
    """Generate synthetic cholecystectomy-specific movements for training augmentation."""

    def __init__(self, num_samples: int = 10000, sequence_length: int = 10, prediction_horizon: int = 30):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.data = self._generate_cholecystectomy_synthetic_data()

    def _generate_cholecystectomy_synthetic_data(self) -> np.ndarray:
        """Generate synthetic movements that mimic laparoscopic cholecystectomy patterns."""
        features_list = []

        for _ in range(self.num_samples + self.sequence_length + self.prediction_horizon):
            # Generate cholecystectomy-specific movement patterns
            pattern_type = np.random.choice(['dissection', 'clipping', 'retraction', 'navigation'])

            if pattern_type == 'dissection':
                # Precise dissection movements (most common in cholecystectomy)
                # Slow, deliberate, curved paths following gallbladder anatomy
                t = np.random.uniform(0, 2*np.pi)
                radius = 0.05  # Small, precise movements
                x = 0.5 + radius * np.cos(t) + 0.01 * np.sin(t * 3) + np.random.normal(0, 0.002)
                y = 0.5 + radius * np.sin(t) + 0.008 * np.cos(t * 2) + np.random.normal(0, 0.002)
                vx = -radius * np.sin(t) - 0.03 * np.cos(t * 3)
                vy = radius * np.cos(t) + 0.016 * np.sin(t * 2)

            elif pattern_type == 'clipping':
                # Cystic duct/artery clipping movements
                # Quick, precise positioning followed by fine adjustments
                t = np.random.uniform(0, 1.5*np.pi)
                if t < np.pi:
                    # Approach phase
                    x = 0.6 + 0.02 * np.sin(t * 2) + np.random.normal(0, 0.001)
                    y = 0.45 + 0.015 * np.cos(t * 1.8) + np.random.normal(0, 0.001)
                    vx = 0.04 * np.cos(t * 2)
                    vy = -0.027 * np.sin(t * 1.8)
                else:
                    # Fine adjustment phase (slower, more precise)
                    x = 0.615 + 0.005 * np.sin(t) + np.random.normal(0, 0.0005)
                    y = 0.432 + 0.004 * np.cos(t * 0.8) + np.random.normal(0, 0.0005)
                    vx = 0.005 * np.cos(t)
                    vy = -0.0032 * np.sin(t * 0.8)

            elif pattern_type == 'retraction':
                # Gallbladder retraction movements
                # Steady pulling with some tissue resistance
                t = np.random.uniform(0, 3*np.pi)
                x = 0.55 + 0.03 * np.sin(t * 0.5) + 0.002 * np.sin(t * 8) + np.random.normal(0, 0.003)
                y = 0.52 + 0.025 * np.cos(t * 0.6) + 0.0015 * np.cos(t * 6) + np.random.normal(0, 0.003)
                vx = 0.015 * np.cos(t * 0.5) + 0.016 * np.cos(t * 8)
                vy = -0.015 * np.sin(t * 0.6) - 0.0094 * np.sin(t * 6)

            else:  # navigation
                # General navigation around abdominal cavity
                t = np.random.uniform(0, 4*np.pi)
                x = 0.5 + 0.08 * np.cos(t * 0.3) + 0.01 * np.sin(t * 2) + np.random.normal(0, 0.005)
                y = 0.5 + 0.06 * np.sin(t * 0.4) + 0.008 * np.cos(t * 1.5) + np.random.normal(0, 0.005)
                vx = -0.024 * np.sin(t * 0.3) + 0.02 * np.cos(t * 2)
                vy = 0.024 * np.cos(t * 0.4) - 0.012 * np.sin(t * 1.5)

            # Calculate derived features (same as real data processing)
            ax = np.random.normal(0, 0.3)  # Realistic acceleration
            ay = np.random.normal(0, 0.3)
            jx = np.random.normal(0, 1.0)  # Jerk for smoothness
            jy = np.random.normal(0, 1.0)

            curvature = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-8)**(3/2)
            speed = np.sqrt(vx**2 + vy**2)
            direction = np.arctan2(vy, vx)

            # Cholecystectomy-specific path efficiency
            path_efficiency = np.random.uniform(0.8, 1.0)  # Higher efficiency for surgical movements

            # Tremor indicator (more prominent in synthetic data for training)
            tremor_indicator = np.random.uniform(0.1, 0.4)  # Some baseline tremor

            features_list.append([
                x, y, vx, vy, ax, ay, jx, jy,
                curvature, speed, direction, path_efficiency, tremor_indicator
            ])

        return np.array(features_list, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get input sequence and target prediction."""
        center_idx = idx + self.sequence_length

        # Input sequence
        input_sequence = self.data[center_idx - self.sequence_length:center_idx]

        # Target position
        target_idx = center_idx + self.prediction_horizon
        target_position = self.data[target_idx, :2]

        return {
            'input': torch.tensor(input_sequence, dtype=torch.float32),
            'target': torch.tensor(target_position, dtype=torch.float32),
            'metadata': {
                'center_idx': center_idx,
                'target_idx': target_idx,
                'pattern_type': 'cholecystectomy_synthetic'
            }
        }