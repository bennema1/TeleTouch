"""
Color-Based Instrument Detector - Fallback method using HSV color space.

Use this if ROSMA annotations are not available.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class ColorDetector:
    """
    Detects surgical instruments using color-based detection in HSV color space.
    """
    
    def __init__(self, 
                 hue_range: Tuple[int, int] = (80, 130),
                 saturation_range: Tuple[int, int] = (0, 80),
                 value_range: Tuple[int, int] = (120, 255),
                 min_contour_area: int = 500,
                 min_elongation: float = 3.0):
        """
        Initialize color detector.
        
        Args:
            hue_range: HSV hue range for instruments (default: blue-gray metal)
            saturation_range: HSV saturation range (low = not colorful)
            value_range: HSV value range (high = bright/reflective)
            min_contour_area: Minimum contour area in pixels
            min_elongation: Minimum length/width ratio for instruments
        """
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range
        self.min_contour_area = min_contour_area
        self.min_elongation = min_elongation
        
        # Smoothing
        self.position_history = deque(maxlen=10)
        
        print(f"[ColorDetector] Initialized")
        print(f"  HSV Range: H={hue_range}, S={saturation_range}, V={value_range}")
        print(f"  Min area: {min_contour_area}, Min elongation: {min_elongation}")
    
    def detect_instruments(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect instrument tip positions in a frame.
        
        Args:
            frame: BGR video frame
            
        Returns:
            List of (x, y) tip positions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for instrument color range
        lower_bound = np.array([self.hue_range[0], self.saturation_range[0], self.value_range[0]])
        upper_bound = np.array([self.hue_range[1], self.saturation_range[1], self.value_range[1]])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tip_positions = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Filter by elongation (instruments are long and thin)
            if len(contour) < 5:
                continue
            
            # Fit rotated rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            
            elongation = max(width, height) / min(width, height)
            if elongation < self.min_elongation:
                continue
            
            # Find tip position (furthest point in direction of elongation)
            tip_pos = self._find_tip_position(contour, rect)
            if tip_pos:
                tip_positions.append(tip_pos)
        
        # Sort by distance from center (instruments usually in center of surgical field)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        tip_positions.sort(key=lambda p: np.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2))
        
        return tip_positions
    
    def _find_tip_position(self, contour: np.ndarray, rect: Tuple) -> Optional[Tuple[int, int]]:
        """
        Find the tip position of an instrument contour.
        
        Args:
            contour: Contour points
            rect: Minimum area rectangle
            
        Returns:
            (x, y) tip position or None
        """
        # Get rectangle center and angle
        center, (width, height), angle = rect
        
        # Find extreme points
        points = contour.reshape(-1, 2)
        
        # Calculate direction vector of instrument
        angle_rad = np.radians(angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Find point furthest in direction of instrument
        distances = np.dot(points - center, direction)
        max_idx = np.argmax(distances)
        
        tip_pos = tuple(points[max_idx].astype(int))
        return tip_pos
    
    def detect_with_smoothing(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect instruments with temporal smoothing.
        
        Args:
            frame: BGR video frame
            
        Returns:
            List of smoothed (x, y) tip positions
        """
        # Detect instruments
        positions = self.detect_instruments(frame)
        
        # Smooth positions using history
        if len(self.position_history) > 0 and len(positions) > 0:
            # Match new positions to previous ones (closest distance)
            smoothed = []
            for new_pos in positions:
                if len(self.position_history) > 0:
                    # Find closest previous position
                    prev_positions = list(self.position_history)[-1]
                    if prev_positions:
                        distances = [np.sqrt((new_pos[0] - p[0])**2 + (new_pos[1] - p[1])**2) 
                                   for p in prev_positions]
                        min_dist = min(distances) if distances else float('inf')
                        
                        # If too far from previous, might be noise
                        if min_dist > 50:  # Threshold: 50 pixels
                            continue
                        
                        # Smooth with previous position
                        closest_idx = np.argmin(distances)
                        prev_pos = prev_positions[closest_idx]
                        smoothed_x = int(0.7 * new_pos[0] + 0.3 * prev_pos[0])
                        smoothed_y = int(0.7 * new_pos[1] + 0.3 * prev_pos[1])
                        smoothed.append((smoothed_x, smoothed_y))
                    else:
                        smoothed.append(new_pos)
                else:
                    smoothed.append(new_pos)
            
            positions = smoothed
        
        # Update history
        self.position_history.append(positions)
        
        return positions


class ColorDetectionDataSource:
    """
    Data source that uses color-based detection to provide instrument positions.
    """
    
    def __init__(self, video_path: str, instrument_index: int = 0, 
                 detector_params: Optional[Dict] = None):
        """
        Initialize color detection data source.
        
        Args:
            video_path: Path to video file
            instrument_index: Which instrument to track (0 = first, 1 = second, etc.)
            detector_params: Optional parameters for ColorDetector
        """
        self.video_path = video_path
        self.instrument_index = instrument_index
        
        # Initialize detector
        if detector_params:
            self.detector = ColorDetector(**detector_params)
        else:
            self.detector = ColorDetector()
        
        # Get video properties
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            raise ValueError(f"Could not open video: {video_path}")
        
        # Current frame
        self.current_frame_num = 0
        self.video_capture = None
        
        print(f"[ColorDetectionDataSource] Tracking instrument {instrument_index + 1}")
        print(f"  Video: {self.width}x{self.height}")
    
    def get_current_position(self) -> Tuple[float, float]:
        """Get position for current frame and advance."""
        if self.video_capture is None:
            import cv2
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        
        ret, frame = self.video_capture.read()
        if not ret:
            # Loop video
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_num = 0
            ret, frame = self.video_capture.read()
            if not ret:
                return (0.5, 0.5)  # Default center
        
        # Detect instruments
        tip_positions = self.detector.detect_with_smoothing(frame)
        
        # Get requested instrument
        if tip_positions and self.instrument_index < len(tip_positions):
            x_pixel, y_pixel = tip_positions[self.instrument_index]
            x_norm = x_pixel / self.width
            y_norm = y_pixel / self.height
            self.current_frame_num += 1
            return (x_norm, y_norm)
        else:
            # No detection - use center or last known
            self.current_frame_num += 1
            return (0.5, 0.5)
    
    def get_position(self, frame_number: int) -> Tuple[float, float]:
        """Get position for specific frame."""
        # For now, just return current (would need to seek video)
        return self.get_current_position()
    
    def reset(self) -> None:
        """Reset to beginning."""
        self.current_frame_num = 0
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.detector.position_history.clear()
    
    def __len__(self) -> int:
        """Total frames."""
        return self.total_frames
    
    def get_name(self) -> str:
        """Return data source name."""
        return f"Color Detection (Instrument {self.instrument_index + 1})"
    
    def release(self) -> None:
        """Release resources."""
        if self.video_capture:
            self.video_capture.release()
