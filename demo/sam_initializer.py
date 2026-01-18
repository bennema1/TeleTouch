"""
Interactive SAM Initialization Tool

Allows user to click on instruments in the first frame to initialize SAM tracking.
"""

import cv2
import numpy as np
from pathlib import Path
from sam_detector import SAMInstrumentDetector, SAMTracker
from sam_skeleton_annotator import SkeletonAnnotator


class SAMInitializer:
    """Interactive tool to initialize SAM with manual clicks."""
    
    def __init__(self, video_path: str, frame_number: int = 0, model_path: str = None, use_skeleton: bool = True):
        """
        Initialize the tool.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to use for initialization
            model_path: Path to SAM model (auto-downloads if None)
        """
        self.video_path = video_path
        self.frame_number = frame_number
        self.clicked_points = []
        self.frame = None
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
        
        # Initialize SAM
        self.use_skeleton = use_skeleton
        
        if use_skeleton:
            # Use skeleton annotator (better method)
            print("[SAMInitializer] Using skeleton annotation method (multi-point)")
            self.annotator = SkeletonAnnotator(video_path, frame_number, model_path)
        else:
            # Use old single-click method
            print("[SAMInitializer] Loading SAM model (this may take a moment)...")
            self.detector = SAMInstrumentDetector(model_path=model_path)
            
            if self.detector.predictor is None:
                raise RuntimeError("Failed to load SAM model")
            
            print("[SAMInitializer] SAM model loaded successfully")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            print(f"  Clicked point {len(self.clicked_points)}: ({x}, {y})")
            
            # Draw point on frame
            cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, f"{len(self.clicked_points)}", 
                       (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('SAM Initializer', self.display_frame)
            cv2.waitKey(1)  # Force window update
    
    def run(self) -> list:
        """
        Run interactive initialization.
        
        Returns:
            List of (x, y) tip positions in pixels
        """
        if self.use_skeleton:
            # Use skeleton annotator
            return self.annotator.run()
        
        # Old single-click method (fallback)
        self.display_frame = self.frame.copy()
        
        cv2.namedWindow('SAM Initializer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SAM Initializer', 1280, 720)
        cv2.setMouseCallback('SAM Initializer', self.mouse_callback)
        
        print("\n" + "="*60)
        print("SAM INITIALIZATION")
        print("="*60)
        print("\nInstructions:")
        print("  1. Click on each instrument tip in the image")
        print("  2. Click on 1-3 instruments (usually 1-2 for surgery)")
        print("  3. Press 'S' to segment and see results")
        print("  4. Press 'R' to reset and start over")
        print("  5. Press 'Q' to finish and save")
        print("\n" + "="*60 + "\n")
        
        tips = []
        
        while True:
            cv2.imshow('SAM Initializer', self.display_frame)
            key = cv2.waitKey(30) & 0xFF  # Increased wait time for better responsiveness
            
            if key == ord('q') or key == 27:  # Q or ESC
                if tips:
                    break
                elif self.clicked_points:
                    # Allow quitting even without segmentation
                    print("Quitting without segmentation - tips will be saved from clicks")
                    tips = [(self.frame.shape[1] // 2, self.frame.shape[0] // 2)]  # Default center
                    break
                else:
                    print("Please click on instruments first, or press Q again to quit")
            
            elif key == ord('s'):  # Segment
                if not self.clicked_points:
                    print("Please click on instruments first")
                    continue
                
                print(f"\nSegmenting {len(self.clicked_points)} instruments...")
                
                # Segment using clicked points
                masks = self.detector.segment_with_prompts(
                    self.frame, 
                    point_prompts=self.clicked_points
                )
                
                # Extract tips from masks
                tips = []
                result_frame = self.frame.copy()
                
                for i, mask in enumerate(masks):
                    tip = self.detector.extract_tip_from_mask(mask)
                    if tip:
                        tips.append(tip)
                        tip_x, tip_y = tip
                        
                        # Draw mask overlay
                        mask_colored = np.zeros_like(self.frame)
                        mask_colored[mask > 0] = [0, 255, 0]  # Green
                        result_frame = cv2.addWeighted(result_frame, 0.7, mask_colored, 0.3, 0)
                        
                        # Draw tip
                        cv2.circle(result_frame, (tip_x, tip_y), 10, (0, 0, 255), -1)
                        cv2.putText(result_frame, f"Tip {i+1}", 
                                   (tip_x + 15, tip_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if tips:
                    print(f"  Found {len(tips)} instrument tips")
                    self.display_frame = result_frame
                else:
                    print("  Warning: No tips extracted from masks")
            
            elif key == ord('r'):  # Reset
                self.clicked_points = []
                self.display_frame = self.frame.copy()
                tips = []
                print("Reset - click on instruments again")
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        if tips:
            print(f"\n✓ Initialized with {len(tips)} instruments")
            for i, tip in enumerate(tips):
                print(f"  Instrument {i+1}: ({tip[0]}, {tip[1]})")
        else:
            print("\n⚠ No tips saved - using default center position")
            tips = [(self.frame.shape[1] // 2, self.frame.shape[0] // 2)]
        
        return tips


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize SAM with manual clicks')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number (default: 0)')
    parser.add_argument('--model', type=str, default=None, help='Path to SAM model weights')
    
    args = parser.parse_args()
    
    try:
        initializer = SAMInitializer(args.video, args.frame, args.model)
        tips = initializer.run()
        
        # Save tips to file
        tips_file = Path(args.video).with_suffix('.sam_tips.txt')
        with open(tips_file, 'w') as f:
            for tip in tips:
                f.write(f"{tip[0]},{tip[1]}\n")
        
        print(f"\nTips saved to: {tips_file}")
        print("\nTo use these tips, run:")
        print(f"  python demo/main.py --video \"{args.video}\" --sam-detect --sam-tips \"{tips_file}\"")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
