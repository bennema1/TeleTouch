"""
ROSMA Annotation Reader - Loads pre-annotated instrument positions from ROSMA dataset.

This is the RECOMMENDED approach - uses ground truth annotations instead of trying to detect instruments.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class AnnotationReader:
    """
    Reads ROSMA annotation files and provides instrument positions for each frame.
    """
    
    def __init__(self, annotation_path: str):
        """
        Initialize annotation reader.
        
        Args:
            annotation_path: Path to ROSMA annotation JSON file
        """
        self.annotation_path = Path(annotation_path)
        
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        # Load annotations
        with open(self.annotation_path, 'r') as f:
            self.data = json.load(f)
        
        # Build frame lookup dictionary for fast access
        self.frame_data = {}
        for frame_info in self.data.get('frames', []):
            frame_num = frame_info.get('frame_number', 0)
            self.frame_data[frame_num] = frame_info
        
        self.total_frames = max(self.frame_data.keys()) if self.frame_data else 0
        self.video_id = self.data.get('video_id', 'unknown')
        
        print(f"[AnnotationReader] Loaded annotations for {self.video_id}")
        print(f"  Total annotated frames: {len(self.frame_data)}")
        print(f"  Max frame number: {self.total_frames}")
    
    def get_frame(self, frame_number: int) -> List[Dict]:
        """
        Get instrument positions for a specific frame.
        
        Args:
            frame_number: Frame number (0-indexed)
            
        Returns:
            List of instrument dictionaries, each containing:
            - 'type': Instrument type (e.g., 'grasper', 'scissors')
            - 'tip_position': (x, y) pixel coordinates
            - 'shaft_position': (x, y) pixel coordinates (optional)
        """
        if frame_number in self.frame_data:
            return self.frame_data[frame_number].get('instruments', [])
        else:
            # Frame not annotated - return empty list
            return []
    
    def get_tip_positions(self, frame_number: int) -> List[Tuple[int, int]]:
        """
        Get just the tip positions for a frame (simplified interface).
        
        Args:
            frame_number: Frame number
            
        Returns:
            List of (x, y) tip positions
        """
        instruments = self.get_frame(frame_number)
        positions = []
        for inst in instruments:
            tip_pos = inst.get('tip_position')
            if tip_pos and len(tip_pos) >= 2:
                positions.append((int(tip_pos[0]), int(tip_pos[1])))
        return positions
    
    def get_all_positions(self, frame_number: int) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get all positions grouped by instrument type.
        
        Args:
            frame_number: Frame number
            
        Returns:
            Dictionary mapping instrument type to list of (x, y) positions
        """
        instruments = self.get_frame(frame_number)
        result = {}
        for inst in instruments:
            inst_type = inst.get('type', 'unknown')
            tip_pos = inst.get('tip_position')
            if tip_pos and len(tip_pos) >= 2:
                if inst_type not in result:
                    result[inst_type] = []
                result[inst_type].append((int(tip_pos[0]), int(tip_pos[1])))
        return result
    
    def has_frame(self, frame_number: int) -> bool:
        """Check if frame has annotations."""
        return frame_number in self.frame_data
    
    def __len__(self) -> int:
        """Total number of annotated frames."""
        return len(self.frame_data)
    
    def get_video_id(self) -> str:
        """Get video ID from annotations."""
        return self.video_id


class AnnotationDataSource:
    """
    Data source that uses ROSMA annotations to provide instrument positions.
    
    Implements the same interface as SyntheticDataSource and KinematicDataSource.
    """
    
    def __init__(self, annotation_path: str, video_path: str, instrument_index: int = 0):
        """
        Initialize annotation-based data source.
        
        Args:
            annotation_path: Path to ROSMA annotation JSON file
            video_path: Path to corresponding video file (for frame count)
            instrument_index: Which instrument to track (0 = first, 1 = second, etc.)
        """
        self.annotation_path = annotation_path
        self.video_path = video_path
        self.instrument_index = instrument_index
        
        # Load annotations
        self.reader = AnnotationReader(annotation_path)
        
        # Get video frame count for validation
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            self.video_frame_count = self.reader.total_frames
        
        # Current frame tracking
        self.current_frame = 0
        
        # Position history for smoothing
        self.position_history = []
        self.last_known_position = (0.5, 0.5)  # Normalized default
        
        print(f"[AnnotationDataSource] Using instrument {instrument_index + 1}")
        print(f"  Video frames: {self.video_frame_count}")
        print(f"  Annotated frames: {len(self.reader)}")
    
    def get_current_position(self) -> Tuple[float, float]:
        """
        Get position for current frame and advance.
        
        Returns:
            (x, y) normalized position (0-1)
        """
        # Get tip positions for current frame
        tip_positions = self.reader.get_tip_positions(self.current_frame)
        
        # If no annotations for this frame, try previous frame
        if not tip_positions:
            # Try previous frames (up to 5 frames back)
            for offset in range(1, 6):
                prev_frame = self.current_frame - offset
                if prev_frame >= 0:
                    tip_positions = self.reader.get_tip_positions(prev_frame)
                    if tip_positions:
                        break
        
        # Get the requested instrument position
        if tip_positions and self.instrument_index < len(tip_positions):
            x_pixel, y_pixel = tip_positions[self.instrument_index]
            
            # Normalize to 0-1 (we need video dimensions)
            # For now, assume standard dimensions or get from video
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                # Default dimensions if video can't be opened
                width, height = 1280, 720
            
            x_norm = x_pixel / width
            y_norm = y_pixel / height
            
            # Smooth position (simple moving average)
            self.position_history.append((x_norm, y_norm))
            if len(self.position_history) > 5:
                self.position_history.pop(0)
            
            # Average last few positions for smoothing
            if len(self.position_history) > 0:
                avg_x = np.mean([p[0] for p in self.position_history])
                avg_y = np.mean([p[1] for p in self.position_history])
                self.last_known_position = (avg_x, avg_y)
                return (avg_x, avg_y)
            else:
                self.last_known_position = (x_norm, y_norm)
                return (x_norm, y_norm)
        else:
            # No annotation found - use last known position
            self.current_frame += 1
            return self.last_known_position
    
    def get_position(self, frame_number: int) -> Tuple[float, float]:
        """
        Get position for a specific frame.
        
        Args:
            frame_number: Frame number
            
        Returns:
            (x, y) normalized position
        """
        tip_positions = self.reader.get_tip_positions(frame_number)
        
        if tip_positions and self.instrument_index < len(tip_positions):
            x_pixel, y_pixel = tip_positions[self.instrument_index]
            
            # Normalize (need video dimensions)
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                width, height = 1280, 720
            
            x_norm = x_pixel / width
            y_norm = y_pixel / height
            return (x_norm, y_norm)
        else:
            return self.last_known_position
    
    def reset(self) -> None:
        """Reset to beginning."""
        self.current_frame = 0
        self.position_history = []
    
    def __len__(self) -> int:
        """Total number of frames."""
        return self.video_frame_count
    
    def get_name(self) -> str:
        """Return data source name."""
        return f"ROSMA Annotations (Instrument {self.instrument_index + 1})"
