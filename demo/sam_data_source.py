"""
SAM-based Data Source for Demo Integration

Provides instrument positions using SAM (Segment Anything Model) detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
try:
    from .sam_detector import SAMInstrumentDetector, SAMTracker
except ImportError:
    from sam_detector import SAMInstrumentDetector, SAMTracker


class SAMDataSource:
    """
    Data source that detects instruments using SAM.
    
    First frame requires manual initialization (click on instruments).
    Subsequent frames use automatic tracking.
    """
    
    def __init__(self, video_path: str, instrument_index: int = 0,
                 model_path: Optional[str] = None, model_type: str = "vit_h",
                 initial_tips: Optional[list] = None):
        """
        Initialize SAM data source.
        
        Args:
            video_path: Path to video file
            instrument_index: Which instrument to track (0 = first, 1 = second, etc.)
            model_path: Path to SAM model weights (auto-downloads if None)
            model_type: Model type - "vit_h" (best), "vit_l", "vit_b" (fastest)
            initial_tips: List of (x, y) pixel coordinates for first frame (auto-detects if None)
        """
        self.video_path = video_path
        self.instrument_index = instrument_index
        
        # Open video
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        # Initialize SAM detector
        print("[SAMDataSource] Initializing SAM detector...")
        self.detector = SAMInstrumentDetector(model_path=model_path, model_type=model_type)
        
        # Initialize tracker
        self.tracker = SAMTracker(self.detector, max_distance=0.1)
        
        # Current frame tracking
        self.current_frame = 0
        self.initialized = False
        self.last_known_position = (0.5, 0.5)
        
        # Initialize with first frame if tips provided
        if initial_tips:
            ret, frame = self.video_capture.read()
            if ret:
                self.tracker.initialize(frame, initial_tips)
                self.initialized = True
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        print(f"[SAMDataSource] Initialized")
        print(f"  Video: {Path(video_path).name}")
        print(f"  Resolution: {self.video_width}x{self.video_height}")
        print(f"  Frames: {self.video_frame_count}")
        print(f"  FPS: {self.video_fps:.1f}")
        print(f"  Instrument index: {instrument_index}")
        print(f"  Initialized: {self.initialized}")
    
    def initialize_from_frame(self, frame: np.ndarray, tips: list):
        """
        Initialize tracker from a frame with manual tips.
        
        Args:
            frame: Frame image (BGR)
            tips: List of (x, y) pixel coordinates
        """
        self.tracker.initialize(frame, tips)
        self.initialized = True
    
    def auto_initialize(self, frame: np.ndarray) -> bool:
        """
        Attempt automatic initialization using simple heuristics.
        
        This is a fallback - manual initialization is more reliable.
        
        Args:
            frame: First frame
            
        Returns:
            True if initialization succeeded
        """
        # Try to detect instruments using simple color-based detection first
        # Then use those as SAM prompts
        try:
            try:
                from .instrument_detector import InstrumentDetector
            except ImportError:
                from instrument_detector import InstrumentDetector
            
            # Use default color detector to get initial positions
            detector = InstrumentDetector()
            tips_normalized = detector.detect(frame)
            
            if tips_normalized:
                # Convert to pixel coordinates
                height, width = frame.shape[:2]
                tips_pixels = [
                    (int(tip[0] * width), int(tip[1] * height))
                    for tip in tips_normalized
                ]
                
                self.tracker.initialize(frame, tips_pixels)
                self.initialized = True
                print(f"[SAMDataSource] Auto-initialized with {len(tips_pixels)} instruments")
                return True
        except Exception as e:
            print(f"[SAMDataSource] Auto-initialization failed: {e}")
        
        return False
    
    def get_current_position(self) -> Tuple[float, float]:
        """
        Get position for current frame and advance.
        
        Returns:
            Normalized (x, y) position (0-1 range)
        """
        # Read current frame
        ret, frame = self.video_capture.read()
        
        if not ret:
            # End of video - loop
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_capture.read()
            self.tracker.reset()
            self.initialized = False
        
        if ret:
            # Initialize on first frame if not done
            if not self.initialized:
                if self.auto_initialize(frame):
                    # Try auto-initialization
                    pass
                else:
                    # Return default position until initialized
                    self.current_frame += 1
                    return self.last_known_position
            
            # Track instruments
            tips = self.tracker.update(frame)
            
            # Get the requested instrument
            if tips and self.instrument_index < len(tips):
                self.last_known_position = tips[self.instrument_index]
            # else: use last known position
        
        self.current_frame += 1
        return self.last_known_position
    
    def get_position(self, frame_number: int) -> Tuple[float, float]:
        """
        Get position for a specific frame.
        
        Args:
            frame_number: Frame number (0-indexed)
            
        Returns:
            Normalized (x, y) position
        """
        # Seek to frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_capture.read()
        
        if ret:
            if not self.initialized:
                self.auto_initialize(frame)
            
            tips = self.tracker.update(frame)
            if tips and self.instrument_index < len(tips):
                return tips[self.instrument_index]
        
        return self.last_known_position
    
    def reset(self) -> None:
        """Reset to beginning."""
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        self.tracker.reset()
        self.initialized = False
        self.last_known_position = (0.5, 0.5)
    
    def release(self) -> None:
        """Release video capture."""
        if self.video_capture:
            self.video_capture.release()
    
    def __len__(self) -> int:
        """Total number of frames."""
        return self.video_frame_count
    
    def get_name(self) -> str:
        """Get data source name."""
        return "SAM Detection"
