"""
Kinematic Data Source - Plays back real surgical trajectories for the demo.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

class KinematicDataSource:
    """
    Data source that plays back pre-recorded surgical trajectories.
    """
    
    def __init__(self, npy_path: str):
        """
        Args:
            npy_path: Path to .npy file with shape (N, 2)
        """
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"Kinematic data not found at {self.npy_path}")
            
        # Load the trajectory
        self.trajectory = np.load(self.npy_path)
        self.total_frames = len(self.trajectory)
        self.current_frame = 0
        
        print(f"[KinematicDataSource] Loaded {self.total_frames} frames from {self.npy_path.name}")
    
    def get_position(self, frame_number: int) -> Tuple[float, float]:
        """Get position for a specific frame."""
        idx = frame_number % self.total_frames
        return tuple(self.trajectory[idx])
    
    def get_current_position(self) -> Tuple[float, float]:
        """Get position for current frame and advance."""
        pos = self.get_position(self.current_frame)
        self.current_frame = (self.current_frame + 1) % self.total_frames
        return pos
    
    def reset(self) -> None:
        """Reset to frame 0."""
        self.current_frame = 0
        
    def __len__(self) -> int:
        return self.total_frames

    def get_name(self) -> str:
        return f"Real Kinematics ({self.npy_path.name})"
