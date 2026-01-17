"""
Lag Buffer - Simulates network latency by storing positions with timestamps
Think of it as a conveyor belt: positions go in one end, come out 500ms later
"""

from collections import deque
from typing import Tuple, Optional
import time


class LagBuffer:
    def __init__(self, delay_seconds: float = 0.5):
        """
        Initialize lag buffer with specified delay.
        
        Args:
            delay_seconds: How much delay to simulate (default 500ms)
        """
        self.delay = delay_seconds
        # Deque of (timestamp, position) tuples
        # Deque is efficient for adding/removing from both ends
        self.buffer: deque = deque()
    
    def push(self, position: Tuple[float, float], timestamp: float) -> None:
        """
        Add a new position to the buffer.
        
        Args:
            position: (x, y) normalized coordinates (0-1 range)
            timestamp: Time in seconds when this position was recorded
        """
        self.buffer.append((timestamp, position))
    
    def get_lagged(self, current_time: float) -> Optional[Tuple[float, float]]:
        """
        Get the position from [delay] seconds ago.
        
        Args:
            current_time: Current timestamp in seconds
            
        Returns:
            Position from [delay] seconds ago, or None if buffer is empty
        """
        target_time = current_time - self.delay
        
        # Find the position closest to target_time
        best_pos = None
        best_diff = float('inf')
        
        for ts, pos in self.buffer:
            diff = abs(ts - target_time)
            if diff < best_diff:
                best_diff = diff
                best_pos = pos
        
        return best_pos
    
    def cleanup(self, current_time: float) -> None:
        """
        Remove positions that are too old (older than delay + small margin).
        Call this periodically to prevent memory buildup.
        
        Args:
            current_time: Current timestamp in seconds
        """
        cutoff = current_time - self.delay - 0.1  # Keep 100ms extra margin
        
        # Remove old entries from the front of the deque
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
    
    def clear(self) -> None:
        """Clear all buffered positions (use when restarting video)."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Return number of positions in buffer."""
        return len(self.buffer)


# Quick test
if __name__ == "__main__":
    print("Testing LagBuffer...")
    
    buffer = LagBuffer(delay_seconds=0.5)
    
    # Simulate pushing positions over time
    for i in range(20):
        t = i * 0.1  # Every 100ms
        pos = (0.3 + i * 0.01, 0.4 + i * 0.005)  # Moving diagonally
        buffer.push(pos, t)
        print(f"t={t:.1f}s: Pushed {pos}")
    
    # Now get lagged position at t=2.0s (should return position from t=1.5s)
    current_time = 2.0
    lagged = buffer.get_lagged(current_time)
    print(f"\nAt t={current_time}s, lagged position (from t=1.5s): {lagged}")
    print(f"Buffer size: {len(buffer)}")
    
    # Cleanup old entries
    buffer.cleanup(current_time)
    print(f"After cleanup, buffer size: {len(buffer)}")
    
    print("\n✓ LagBuffer test complete!")