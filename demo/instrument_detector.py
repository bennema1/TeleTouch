"""
Color-Based Surgical Instrument Detector

Detects surgical instruments in video frames using HSV color space filtering
and contour analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path


class InstrumentDetector:
    """
    Detects surgical instruments using color-based segmentation.
    
    Uses HSV color space filtering to identify instruments, then extracts
    tip positions from detected contours.
    """
    
    def __init__(self, params: Optional[Dict] = None, params_file: Optional[str] = None):
        """
        Initialize detector with parameters.
        
        Args:
            params: Dictionary with detection parameters
            params_file: Path to JSON file with parameters
        """
        if params_file and Path(params_file).exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
        
        if params is None:
            # Default parameters (good starting point)
            params = {
                'hue': (80, 130),
                'sat': (0, 80),
                'val': (120, 255),
                'min_area': 500,
                'min_elongation': 3.0
            }
        
        self.hue_range = params['hue']
        self.sat_range = params['sat']
        self.val_range = params['val']
        self.min_area = params['min_area']
        self.min_elongation = params['min_elongation']
        
        # Tracking for temporal smoothing
        self.previous_positions = []
        self.alpha = 0.7  # Smoothing factor
    
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect instrument tips in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of (x, y) tip positions (normalized 0-1)
        """
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = self._create_mask(hsv_frame)
        
        # Find contours
        contours = self._find_contours(mask)
        
        # Filter and extract tips
        tips = self._extract_tips(contours, frame.shape)
        
        # Smooth with previous frame
        tips = self._smooth_positions(tips)
        
        return tips
    
    def _create_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Create binary mask from HSV frame."""
        mask = cv2.inRange(
            hsv_frame,
            np.array([self.hue_range[0], self.sat_range[0], self.val_range[0]]),
            np.array([self.hue_range[1], self.sat_range[1], self.val_range[1]])
        )
        
        # Clean mask: remove noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _find_contours(self, mask: np.ndarray) -> List:
        """Find contours in mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _extract_tips(self, contours: List, frame_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        Extract tip positions from contours.
        
        Args:
            contours: List of contour arrays
            frame_shape: (height, width) of frame
            
        Returns:
            List of normalized (x, y) tip positions
        """
        tips = []
        height, width = frame_shape[:2]
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Filter by elongation (instruments are elongated, not round)
            x, y, w, h = cv2.boundingRect(contour)
            elongation = max(w, h) / max(min(w, h), 1)
            
            if elongation < self.min_elongation:
                continue
            
            # Find tip: point furthest from centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Find furthest point from centroid (this is the tip)
            max_dist = 0
            tip_x, tip_y = cx, cy
            
            for point in contour:
                px, py = point[0]
                dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                if dist > max_dist:
                    max_dist = dist
                    tip_x, tip_y = px, py
            
            # Normalize to 0-1
            tip_x_norm = tip_x / width
            tip_y_norm = tip_y / height
            
            tips.append((tip_x_norm, tip_y_norm))
        
        return tips
    
    def _smooth_positions(self, current_tips: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Smooth positions using temporal filtering.
        
        Matches tips from current frame to previous frame by distance,
        then applies exponential smoothing.
        """
        if not self.previous_positions:
            # First frame: no smoothing
            self.previous_positions = current_tips
            return current_tips
        
        # Match current tips to previous tips by distance
        matched_tips = []
        used_prev = [False] * len(self.previous_positions)
        
        for curr_tip in current_tips:
            best_match_idx = None
            best_dist = float('inf')
            
            for i, prev_tip in enumerate(self.previous_positions):
                if used_prev[i]:
                    continue
                
                dist = np.sqrt(
                    (curr_tip[0] - prev_tip[0])**2 + 
                    (curr_tip[1] - prev_tip[1])**2
                )
                
                if dist < best_dist and dist < 0.1:  # Max distance threshold
                    best_dist = dist
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Smooth with previous position
                prev_tip = self.previous_positions[best_match_idx]
                smoothed_x = self.alpha * curr_tip[0] + (1 - self.alpha) * prev_tip[0]
                smoothed_y = self.alpha * curr_tip[1] + (1 - self.alpha) * prev_tip[1]
                matched_tips.append((smoothed_x, smoothed_y))
                used_prev[best_match_idx] = True
            else:
                # New tip (no match found)
                matched_tips.append(curr_tip)
        
        # Update previous positions
        self.previous_positions = matched_tips
        
        return matched_tips
    
    def reset(self):
        """Reset tracking state."""
        self.previous_positions = []


class SmoothedTracker:
    """
    Temporal smoothing tracker for instrument positions.
    
    Maintains history of positions and applies exponential smoothing
    to reduce jitter.
    """
    
    def __init__(self, alpha: float = 0.7, max_distance: float = 0.1):
        """
        Initialize tracker.
        
        Args:
            alpha: Smoothing factor (0-1), higher = more responsive
            max_distance: Maximum distance for matching (normalized)
        """
        self.alpha = alpha
        self.max_distance = max_distance
        self.previous_positions = []
    
    def update(self, detected_positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Update tracker with new detections.
        
        Args:
            detected_positions: List of (x, y) positions from current frame
            
        Returns:
            Smoothed positions
        """
        if not self.previous_positions:
            # First frame
            self.previous_positions = detected_positions
            return detected_positions
        
        smoothed = []
        used_prev = [False] * len(self.previous_positions)
        
        # Match each detection to previous position
        for detected in detected_positions:
            best_match_idx = None
            best_dist = float('inf')
            
            for i, prev in enumerate(self.previous_positions):
                if used_prev[i]:
                    continue
                
                dist = np.sqrt(
                    (detected[0] - prev[0])**2 + 
                    (detected[1] - prev[1])**2
                )
                
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Smooth
                prev = self.previous_positions[best_match_idx]
                smoothed_x = self.alpha * detected[0] + (1 - self.alpha) * prev[0]
                smoothed_y = self.alpha * detected[1] + (1 - self.alpha) * prev[1]
                smoothed.append((smoothed_x, smoothed_y))
                used_prev[best_match_idx] = True
            else:
                # New detection (no match)
                smoothed.append(detected)
        
        self.previous_positions = smoothed
        return smoothed
    
    def reset(self):
        """Reset tracker state."""
        self.previous_positions = []
