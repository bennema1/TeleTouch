"""
SAM (Segment Anything Model) Based Instrument Detector

Uses Meta's Segment Anything Model to detect surgical instruments.
More robust than color detection, works with any instrument appearance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from pathlib import Path
import os


class SAMInstrumentDetector:
    """
    Detects surgical instruments using SAM (Segment Anything Model).
    
    Uses prompt-based segmentation:
    - First frame: Manual clicks or automatic detection
    - Subsequent frames: Uses previous tip positions as prompts
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "vit_h"):
        """
        Initialize SAM detector.
        
        Args:
            model_path: Path to SAM model weights (.pth file)
            model_type: Model type - "vit_h" (largest, best), "vit_l", "vit_b" (smallest, fastest)
        """
        self.model_type = model_type
        self.model = None
        self.predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Download model if not provided
        if model_path is None:
            model_path = self._download_model()
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            print(f"[SAMDetector] Warning: Model not found at {model_path}")
            print("[SAMDetector] Will attempt to download automatically")
            model_path = self._download_model()
            if model_path:
                self._load_model(model_path)
    
    def _download_model(self) -> Optional[str]:
        """Download SAM model weights if not present."""
        model_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        model_dir = Path("models/sam")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"sam_{self.model_type}.pth"
        
        if model_path.exists():
            print(f"[SAMDetector] Using existing model: {model_path}")
            return str(model_path)
        
        print(f"[SAMDetector] Downloading model {self.model_type}...")
        print(f"[SAMDetector] This may take a few minutes (model is ~2.4GB for vit_h)")
        
        try:
            import urllib.request
            url = model_urls.get(self.model_type, model_urls["vit_h"])
            urllib.request.urlretrieve(url, model_path)
            print(f"[SAMDetector] Model downloaded: {model_path}")
            return str(model_path)
        except Exception as e:
            print(f"[SAMDetector] Failed to download model: {e}")
            print(f"[SAMDetector] Please download manually from: {model_urls.get(self.model_type)}")
            print(f"[SAMDetector] Save to: {model_path}")
            return None
    
    def _load_model(self, model_path: str):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            sam = sam_model_registry[self.model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            print(f"[SAMDetector] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"[SAMDetector] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.predictor = None
    
    def segment_with_prompts(self, frame: np.ndarray, 
                           point_prompts: Optional[List[Tuple[int, int]]] = None,
                           box_prompts: Optional[List[Tuple[int, int, int, int]]] = None) -> List[np.ndarray]:
        """
        Segment instruments using point or box prompts.
        
        Args:
            frame: Input image (BGR)
            point_prompts: List of (x, y) points on instruments
            box_prompts: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            List of binary masks (one per prompt)
        """
        if self.predictor is None:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image
        self.predictor.set_image(rgb_frame)
        
        masks = []
        
        # Process point prompts
        if point_prompts:
            for point in point_prompts:
                x, y = point
                input_point = np.array([[x, y]])
                input_label = np.array([1])  # 1 = foreground point
                
                mask, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
                masks.append(mask[0])  # Take best mask
        
        # Process box prompts
        if box_prompts:
            for box in box_prompts:
                x1, y1, x2, y2 = box
                input_box = np.array([x1, y1, x2, y2])
                
                mask, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                masks.append(mask[0])
        
        return masks
    
    def extract_tip_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Extract tip position from instrument mask.
        
        Uses multiple methods to find the most accurate tip position.
        
        Args:
            mask: Binary mask (white = instrument, black = background)
            
        Returns:
            (x, y) tip position, or None if not found
        """
        # Find all white pixels
        white_pixels = np.column_stack(np.where(mask > 0))
        
        if len(white_pixels) == 0:
            return None
        
        # Compute centroid
        centroid = np.mean(white_pixels, axis=0)
        cy, cx = centroid
        
        # Method 1: Use skeletonization to find endpoints, then select tip
        try:
            from skimage import morphology
            
            # Skeletonize the mask
            skeleton = morphology.skeletonize(mask > 0)
            skeleton_pixels = np.column_stack(np.where(skeleton))
            
            if len(skeleton_pixels) > 0:
                # Find endpoints (pixels with only 1 neighbor)
                endpoints = []
                for py, px in skeleton_pixels:
                    # Count neighbors in skeleton
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = py + dy, px + dx
                            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                                if skeleton[ny, nx]:
                                    neighbors += 1
                    
                    if neighbors == 1:  # Endpoint
                        endpoints.append((px, py))
                
                if len(endpoints) >= 2:
                    # Multiple endpoints - find the one that's the tip
                    # Tip is usually the one that's:
                    # 1. Furthest from centroid
                    # 2. On the "narrower" end (smaller local area around it)
                    
                    endpoint_scores = []
                    for ex, ey in endpoints:
                        # Distance from centroid
                        dist_from_centroid = np.sqrt((ex - cx)**2 + (ey - cy)**2)
                        
                        # Local density (check area around endpoint)
                        local_area = 0
                        for dy in range(-5, 6):
                            for dx in range(-5, 6):
                                ny, nx = ey + dy, ex + dx
                                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                                    if mask[ny, nx] > 0:
                                        local_area += 1
                        
                        # Score: prefer far from centroid AND low local density (tip is narrow)
                        score = dist_from_centroid * (1.0 / max(local_area, 1))
                        endpoint_scores.append((score, ex, ey))
                    
                    # Select endpoint with highest score
                    endpoint_scores.sort(reverse=True)
                    tip_x, tip_y = endpoint_scores[0][1], endpoint_scores[0][2]
                    return (int(tip_x), int(tip_y))
                
                elif len(endpoints) == 1:
                    # Only one endpoint - that's the tip
                    tip_x, tip_y = endpoints[0]
                    return (int(tip_x), int(tip_y))
        
        except ImportError:
            pass
        
        # Method 2: Fallback - find point on contour furthest from centroid
        # This works better than just finding furthest pixel
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Find point on contour furthest from centroid
                max_dist = 0
                tip_x, tip_y = cx, cy
                
                for point in largest_contour:
                    px, py = point[0]
                    dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                    if dist > max_dist:
                        max_dist = dist
                        tip_x, tip_y = px, py
                
                return (int(tip_x), int(tip_y))
        except:
            pass
        
        # Method 3: Final fallback - furthest pixel from centroid
        distances = np.sqrt(np.sum((white_pixels - centroid)**2, axis=1))
        max_idx = np.argmax(distances)
        tip_y, tip_x = white_pixels[max_idx]
        
        return (int(tip_x), int(tip_y))
    
    def detect(self, frame: np.ndarray, 
               previous_tips: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[float, float]]:
        """
        Detect instrument tips in frame.
        
        Args:
            frame: Input frame (BGR)
            previous_tips: Previous frame tip positions (used as prompts) - normalized 0-1
            
        Returns:
            List of normalized (x, y) tip positions (0-1 range)
        """
        if self.predictor is None:
            return []
        
        height, width = frame.shape[:2]
        tips = []
        
        # Convert BGR to RGB for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM (must be called for each frame)
        self.predictor.set_image(rgb_frame)
        
        if previous_tips:
            # Use previous tips as point prompts
            # Convert normalized coordinates to pixel coordinates
            point_prompts = [
                (int(tip[0] * width), int(tip[1] * height)) 
                for tip in previous_tips
            ]
            masks = self.segment_with_prompts(frame, point_prompts=point_prompts)
        else:
            # First frame: try automatic detection or return empty
            # For now, return empty - user should provide initial prompts
            return []
        
        # Extract tips from masks
        for mask in masks:
            tip = self.extract_tip_from_mask(mask)
            if tip:
                # Normalize to 0-1
                tip_x_norm = tip[0] / width
                tip_y_norm = tip[1] / height
                tips.append((tip_x_norm, tip_y_norm))
        
        return tips


class SAMTracker:
    """
    Tracks instruments across frames using SAM and distance matching.
    """
    
    def __init__(self, detector: SAMInstrumentDetector, max_distance: float = 0.1):
        """
        Initialize tracker.
        
        Args:
            detector: SAM detector instance
            max_distance: Maximum normalized distance for matching (default 0.1 = 10% of frame)
        """
        self.detector = detector
        self.max_distance = max_distance
        self.previous_tips = []
        self.frame_count = 0
    
    def update(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Update tracker with new frame.
        
        Args:
            frame: Current frame
            
        Returns:
            List of tracked tip positions
        """
        # Detect using previous tips as prompts
        current_tips = self.detector.detect(frame, previous_tips=self.previous_tips)
        
        if not current_tips:
            # No detections - keep previous positions
            return self.previous_tips
        
        # Match current tips to previous tips by distance
        matched_tips = []
        used_current = [False] * len(current_tips)
        
        for prev_tip in self.previous_tips:
            best_match_idx = None
            best_dist = float('inf')
            
            for i, curr_tip in enumerate(current_tips):
                if used_current[i]:
                    continue
                
                dist = np.sqrt(
                    (curr_tip[0] - prev_tip[0])**2 + 
                    (curr_tip[1] - prev_tip[1])**2
                )
                
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_match_idx = i
            
            if best_match_idx is not None:
                matched_tips.append(current_tips[best_match_idx])
                used_current[best_match_idx] = True
            else:
                # No match - instrument disappeared, keep previous position
                matched_tips.append(prev_tip)
        
        # Add new detections (instruments that appeared)
        for i, curr_tip in enumerate(current_tips):
            if not used_current[i]:
                matched_tips.append(curr_tip)
        
        self.previous_tips = matched_tips
        self.frame_count += 1
        
        return matched_tips
    
    def initialize(self, frame: np.ndarray, initial_tips: List[Tuple[int, int]]):
        """
        Initialize tracker with first frame.
        
        Args:
            frame: First frame
            initial_tips: List of (x, y) pixel coordinates for initial tips
        """
        height, width = frame.shape[:2]
        
        # Convert to normalized coordinates
        normalized_tips = [
            (tip[0] / width, tip[1] / height) 
            for tip in initial_tips
        ]
        
        self.previous_tips = normalized_tips
        self.frame_count = 0
    
    def reset(self):
        """Reset tracker."""
        self.previous_tips = []
        self.frame_count = 0
