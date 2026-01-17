"""
Position History - Tracks recent positions for AI prediction input
The AI needs the last N positions to predict where the instrument will be next
"""

from collections import deque
from typing import List, Tuple, Optional
import numpy as np


class PositionHistory:
    def __init__(self, max_length: int = 30):
        """
        Initialize position history tracker.
        
        Args:
            max_length: Maximum positions to store (default 30 = 1 second at 30fps)
        """
        self.max_length = max_length
        self.positions: deque = deque(maxlen=max_length)
    
    def push(self, position: Tuple[float, float]) -> None:
        """
        Add a new position to history.
        
        Args:
            position: (x, y) normalized coordinates (0-1 range)
        """
        self.positions.append(position)
    
    def get_last(self, n: int = 10) -> List[Tuple[float, float]]:
        """
        Get the last N positions for AI input.
        
        Args:
            n: Number of positions to return (default 10)
            
        Returns:
            List of (x, y) positions, oldest first
        """
        # Convert deque to list and get last n items
        pos_list = list(self.positions)
        return pos_list[-n:] if len(pos_list) >= n else pos_list
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Calculate current velocity (change between last two positions).
        Useful for simple prediction fallback.
        
        Returns:
            (vx, vy) velocity, or None if not enough data
        """
        if len(self.positions) < 2:
            return None
        
        pos_list = list(self.positions)
        prev = pos_list[-2]
        curr = pos_list[-1]
        
        return (curr[0] - prev[0], curr[1] - prev[1])
    
    def get_acceleration(self) -> Optional[Tuple[float, float]]:
        """
        Calculate current acceleration (change in velocity).
        Useful for improved simple prediction.
        
        Returns:
            (ax, ay) acceleration, or None if not enough data
        """
        if len(self.positions) < 3:
            return None
        
        pos_list = list(self.positions)
        p1, p2, p3 = pos_list[-3], pos_list[-2], pos_list[-1]
        
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        return (v2[0] - v1[0], v2[1] - v1[1])
    
    def clear(self) -> None:
        """Clear all history (use when restarting video)."""
        self.positions.clear()
    
    def __len__(self) -> int:
        """Return number of positions in history."""
        return len(self.positions)
    
    def is_ready(self, min_positions: int = 10) -> bool:
        """Check if we have enough history for AI prediction."""
        return len(self.positions) >= min_positions


# Quick test
if __name__ == "__main__":
    print("Testing PositionHistory...")
    
    history = PositionHistory(max_length=30)
    
    # Simulate a curved trajectory
    for i in range(15):
        # Circular motion
        angle = i * 0.2
        x = 0.5 + 0.1 * np.cos(angle)
        y = 0.5 + 0.1 * np.sin(angle)
        history.push((x, y))
        print(f"Frame {i}: ({x:.3f}, {y:.3f})")
    
    print(f"\nHistory length: {len(history)}")
    print(f"Ready for AI (need 10): {history.is_ready()}")
    
    last_10 = history.get_last(10)
    print(f"Last 10 positions: {len(last_10)} items")
    
    vel = history.get_velocity()
    print(f"Current velocity: ({vel[0]:.4f}, {vel[1]:.4f})")
    
    acc = history.get_acceleration()
    print(f"Current acceleration: ({acc[0]:.4f}, {acc[1]:.4f})")
    
    print("\n✓ PositionHistory test complete!")