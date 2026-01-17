"""
Synthetic Data Generator - Creates realistic instrument movement for testing

This lets you develop and test the demo WITHOUT needing:
- Real surgical videos
- Person A's annotation data
- Person B's trained model

The movement patterns simulate real surgical behavior:
- Smooth curves (following tissue contours)
- Occasional pauses (surgeon thinking)
- Quick precise movements (cutting/grasping)
- Random but realistic trajectories
"""

import numpy as np
from typing import Tuple, Generator, Optional
import time


class SyntheticTrajectoryGenerator:
    """Generates realistic surgical instrument movement patterns."""
    
    def __init__(self, fps: int = 30, seed: Optional[int] = None):
        """
        Args:
            fps: Frames per second (for timing calculations)
            seed: Random seed for reproducibility
        """
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        
        # Current state
        self.x = 0.5  # Start at center
        self.y = 0.5
        self.vx = 0.0  # Velocity
        self.vy = 0.0
        
        # Movement parameters
        self.max_speed = 0.015       # Max movement per frame (normalized)
        self.smoothness = 0.85       # How smooth movements are (0-1)
        self.pause_probability = 0.02  # Chance of pausing each frame
        self.is_paused = False
        self.pause_frames = 0
        
        # Boundaries (keep instrument in view)
        self.margin = 0.1
        
        # Pattern state
        self.pattern = "wander"
        self.pattern_progress = 0
        self.target_x = 0.5
        self.target_y = 0.5
    
    def _apply_boundary_force(self) -> Tuple[float, float]:
        """Generate force to keep instrument within boundaries."""
        fx, fy = 0.0, 0.0
        strength = 0.002
        
        if self.x < self.margin:
            fx = strength * (self.margin - self.x)
        elif self.x > 1 - self.margin:
            fx = -strength * (self.x - (1 - self.margin))
        
        if self.y < self.margin:
            fy = strength * (self.margin - self.y)
        elif self.y > 1 - self.margin:
            fy = -strength * (self.y - (1 - self.margin))
        
        return fx, fy
    
    def _choose_new_pattern(self) -> None:
        """Randomly choose a new movement pattern."""
        patterns = ["wander", "circle", "line", "approach"]
        self.pattern = self.rng.choice(patterns)
        self.pattern_progress = 0
        
        # Set target for some patterns
        if self.pattern == "approach":
            self.target_x = 0.3 + self.rng.random() * 0.4
            self.target_y = 0.3 + self.rng.random() * 0.4
    
    def _get_pattern_acceleration(self) -> Tuple[float, float]:
        """Get acceleration based on current movement pattern."""
        ax, ay = 0.0, 0.0
        
        if self.pattern == "wander":
            # Random wandering with occasional direction changes
            ax = self.rng.normal(0, 0.001)
            ay = self.rng.normal(0, 0.001)
            
            if self.pattern_progress > 60:  # Change pattern after ~2 seconds
                self._choose_new_pattern()
        
        elif self.pattern == "circle":
            # Circular motion
            angle = self.pattern_progress * 0.05
            center_x, center_y = 0.5, 0.5
            radius = 0.15
            
            # Target position on circle
            target_x = center_x + radius * np.cos(angle)
            target_y = center_y + radius * np.sin(angle)
            
            ax = (target_x - self.x) * 0.01
            ay = (target_y - self.y) * 0.01
            
            if self.pattern_progress > 120:
                self._choose_new_pattern()
        
        elif self.pattern == "line":
            # Move in a straight line toward a target
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0.01:
                ax = dx * 0.005
                ay = dy * 0.005
            else:
                self._choose_new_pattern()
        
        elif self.pattern == "approach":
            # Approach a target (like grasping)
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0.02:
                # Move toward target
                ax = dx * 0.008
                ay = dy * 0.008
            else:
                # Pause at target
                self.is_paused = True
                self.pause_frames = int(self.rng.integers(30, 90))  # 1-3 second pause
                self._choose_new_pattern()
        
        return ax, ay
    
    def get_next_position(self) -> Tuple[float, float]:
        """
        Generate the next position in the trajectory.
        
        Returns:
            (x, y) normalized position (0-1 range)
        """
        # Handle pausing
        if self.is_paused:
            self.pause_frames -= 1
            if self.pause_frames <= 0:
                self.is_paused = False
            return (self.x, self.y)
        
        # Random pause check
        if self.rng.random() < self.pause_probability:
            self.is_paused = True
            self.pause_frames = int(self.rng.integers(10, 45))  # 0.3-1.5 seconds
            return (self.x, self.y)
        
        # Get pattern-based acceleration
        ax, ay = self._get_pattern_acceleration()
        
        # Add boundary force
        bx, by = self._apply_boundary_force()
        ax += bx
        ay += by
        
        # Update velocity with smoothing
        self.vx = self.smoothness * self.vx + (1 - self.smoothness) * ax
        self.vy = self.smoothness * self.vy + (1 - self.smoothness) * ay
        
        # Clamp velocity
        speed = np.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > self.max_speed:
            self.vx = self.vx / speed * self.max_speed
            self.vy = self.vy / speed * self.max_speed
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Hard clamp to boundaries
        self.x = np.clip(self.x, self.margin, 1 - self.margin)
        self.y = np.clip(self.y, self.margin, 1 - self.margin)
        
        # Increment pattern progress
        self.pattern_progress += 1
        
        return (float(self.x), float(self.y))
    
    def reset(self, x: float = 0.5, y: float = 0.5) -> None:
        """Reset generator to a new starting position."""
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.is_paused = False
        self.pattern_progress = 0
        self._choose_new_pattern()
    
    def generate_trajectory(self, num_frames: int) -> np.ndarray:
        """
        Generate a complete trajectory.
        
        Args:
            num_frames: Number of frames to generate
            
        Returns:
            numpy array of shape (num_frames, 2) with (x, y) positions
        """
        self.reset()
        positions = []
        
        for _ in range(num_frames):
            pos = self.get_next_position()
            positions.append(pos)
        
        return np.array(positions)


class SyntheticDataSource:
    """
    Data source interface for the demo.
    Provides positions frame-by-frame, simulating real annotation data.
    """
    
    def __init__(self, fps: int = 30, duration_seconds: float = 60.0, 
                 seed: Optional[int] = None):
        """
        Args:
            fps: Frames per second
            duration_seconds: How long the "video" is
            seed: Random seed for reproducibility
        """
        self.fps = fps
        self.total_frames = int(fps * duration_seconds)
        
        # Generate the full trajectory upfront
        generator = SyntheticTrajectoryGenerator(fps=fps, seed=seed)
        self.trajectory = generator.generate_trajectory(self.total_frames)
        
        self.current_frame = 0
    
    def get_position(self, frame_number: int) -> Tuple[float, float]:
        """
        Get position for a specific frame.
        
        Args:
            frame_number: Frame index
            
        Returns:
            (x, y) normalized position
        """
        idx = frame_number % self.total_frames  # Loop
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
        """Total number of frames."""
        return self.total_frames


# Quick test
if __name__ == "__main__":
    print("Testing SyntheticTrajectoryGenerator...")
    
    # Generate a trajectory
    generator = SyntheticTrajectoryGenerator(fps=30, seed=42)
    
    # Get 100 positions
    positions = []
    for i in range(100):
        pos = generator.get_next_position()
        positions.append(pos)
        if i % 20 == 0:
            print(f"Frame {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    positions = np.array(positions)
    print(f"\nTrajectory stats:")
    print(f"  X range: {positions[:, 0].min():.3f} - {positions[:, 0].max():.3f}")
    print(f"  Y range: {positions[:, 1].min():.3f} - {positions[:, 1].max():.3f}")
    
    # Test data source
    print("\nTesting SyntheticDataSource...")
    source = SyntheticDataSource(fps=30, duration_seconds=10, seed=42)
    print(f"Total frames: {len(source)}")
    
    for i in range(5):
        pos = source.get_current_position()
        print(f"Frame {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    print("\n✓ Synthetic data test complete!")