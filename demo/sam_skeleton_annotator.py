"""
SAM-based instrument tip detection using skeleton points.
Click along the instrument from base to tip for better accuracy.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
try:
    from .sam_detector import SAMInstrumentDetector
except ImportError:
    from sam_detector import SAMInstrumentDetector


class SkeletonAnnotator:
    """
    Annotate surgical instruments using skeleton points.
    Click along the instrument from base to tip.
    """
    
    def __init__(self, video_path: str, frame_number: int = 0, model_path: Optional[str] = None):
        """
        Initialize skeleton annotator.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to use
            model_path: Path to SAM model weights
        """
        self.video_path = video_path
        self.frame_number = frame_number
        
        # Load video frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
        
        # Convert BGR to RGB for SAM
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_frame = frame.copy()  # Keep BGR for display
        
        # Initialize SAM detector
        print("[SkeletonAnnotator] Loading SAM model...")
        self.detector = SAMInstrumentDetector(model_path=model_path)
        
        if self.detector.predictor is None:
            raise RuntimeError("Failed to load SAM model")
        
        # Set image for SAM
        self.detector.predictor.set_image(self.frame)
        
        # Annotation state
        self.skeleton_points = []
        self.current_mask = None
        self.tip_position = None
        
        print("[SkeletonAnnotator] Ready")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.skeleton_points.append((x, y))
            print(f"  Added point {len(self.skeleton_points)}: ({x}, {y})")
            
            # Clear previous segmentation
            self.current_mask = None
            self.tip_position = None
    
    def draw_display(self):
        """Draw current annotation state."""
        display = self.display_frame.copy()
        
        # Draw mask if available
        if self.current_mask is not None:
            # Convert mask to BGR overlay
            mask_overlay = np.zeros_like(display)
            mask_overlay[self.current_mask] = [0, 255, 0]  # Green
            
            # Blend with image
            display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)
        
        # Draw skeleton points with gradient colors
        for i, point in enumerate(self.skeleton_points):
            # Color: gradient from blue (base) to red (tip)
            progress = i / max(len(self.skeleton_points) - 1, 1) if len(self.skeleton_points) > 1 else 0
            color = (
                int(255 * (1 - progress)),  # Blue decreases
                0,
                int(255 * progress)         # Red increases
            )
            
            cv2.circle(display, point, 8, color, -1)
            cv2.circle(display, point, 10, (255, 255, 255), 2)
            
            # Draw number
            cv2.putText(display, str(i+1), 
                       (point[0] + 12, point[1] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw lines connecting points
        if len(self.skeleton_points) >= 2:
            for i in range(len(self.skeleton_points) - 1):
                cv2.line(display, 
                        self.skeleton_points[i], 
                        self.skeleton_points[i+1],
                        (255, 255, 0), 2)
        
        # Draw extracted tip
        if self.tip_position is not None:
            tip_x, tip_y = int(self.tip_position[0]), int(self.tip_position[1])
            cv2.circle(display, (tip_x, tip_y), 15, (0, 0, 255), 3)
            cv2.circle(display, (tip_x, tip_y), 5, (0, 0, 255), -1)
            
            cv2.putText(display, "TIP",
                       (tip_x + 20, tip_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw instructions
        cv2.putText(display, f"Points: {len(self.skeleton_points)}/5",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self.skeleton_points) >= 2:
            cv2.putText(display, "Press 'S' to segment",
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display
    
    def segment(self):
        """Segment instrument using skeleton points and extract tip."""
        if len(self.skeleton_points) < 2:
            print("Need at least 2 points")
            return
        
        try:
            print(f"\nSegmenting with {len(self.skeleton_points)} points...")
            
            # Prepare prompts for SAM
            point_coords = np.array(self.skeleton_points, dtype=np.float32)
            point_labels = np.ones(len(self.skeleton_points), dtype=np.int32)  # All points are positive
            
            # Run SAM
            masks, scores, logits = self.detector.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True  # Get multiple mask options
            )
            
            # Choose best mask (highest score)
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]
            
            print(f"  Segmentation complete (score: {scores[best_idx]:.3f})")
            
            # Extract tip from mask using base-to-tip method
            self.tip_position = self.extract_tip_from_mask(self.current_mask)
            
            if self.tip_position:
                print(f"  ✓ Tip extracted at ({self.tip_position[0]:.1f}, {self.tip_position[1]:.1f})")
            else:
                print("  ✗ Failed to extract tip")
        except Exception as e:
            print(f"  ✗ Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            # Don't crash - just show error and continue
    
    def extract_tip_from_mask(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Extract instrument tip from segmentation mask.
        
        Strategy: Find pixel furthest from the first skeleton point (base).
        This works because:
        - First point is at the base/shaft
        - Tip is furthest from base
        - More accurate than centroid-based methods
        
        Args:
            mask: Binary mask (True = instrument)
        
        Returns:
            (x, y) tuple of tip position
        """
        if not self.skeleton_points or len(mask.shape) != 2:
            return None
        
        # Get all mask pixels
        mask_points = np.argwhere(mask)  # Returns (y, x) pairs
        
        if len(mask_points) == 0:
            return None
        
        # Convert to (x, y)
        mask_points = mask_points[:, [1, 0]]  # Swap to (x, y)
        
        # Base point (first skeleton point)
        base_point = np.array(self.skeleton_points[0])
        
        # Find pixel furthest from base
        distances = np.linalg.norm(mask_points - base_point, axis=1)
        furthest_idx = np.argmax(distances)
        
        tip = mask_points[furthest_idx]
        
        return (float(tip[0]), float(tip[1]))
    
    def run(self) -> List[Tuple[int, int]]:
        """
        Run interactive annotation.
        
        Returns:
            List of (x, y) tip positions (one per instrument)
        """
        cv2.namedWindow('SAM Skeleton Annotator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SAM Skeleton Annotator', 1280, 720)
        cv2.setMouseCallback('SAM Skeleton Annotator', self.mouse_callback)
        
        print("\n" + "="*60)
        print("SAM SKELETON ANNOTATION")
        print("="*60)
        print("\nInstructions:")
        print("  Click 3-5 points along the instrument from BASE to TIP:")
        print("    1. Base/shaft entry point (blue)")
        print("    2. First joint/bend")
        print("    3. Wrist joint")
        print("    4. Near tip")
        print("    5. Exact tip (red)")
        print("\nControls:")
        print("  - Click: Add point along instrument spine")
        print("  - 'S': Segment and extract tip")
        print("  - 'R': Reset and start over")
        print("  - 'D': Delete last point")
        print("  - 'Q': Finish and save")
        print("\n" + "="*60 + "\n")
        
        tips = []
        
        while True:
            display = self.draw_display()
            cv2.imshow('SAM Skeleton Annotator', display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('s'):
                if len(self.skeleton_points) >= 2:
                    # Segment
                    try:
                        self.segment()
                        if self.tip_position:
                            tips = [self.tip_position]
                    except Exception as e:
                        print(f"Error during segmentation: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Window will stay open - try again or press 'R' to reset")
                else:
                    print("Need at least 2 points to segment. Click more points first.")
            
            elif key == ord('r'):
                # Reset
                self.skeleton_points = []
                self.current_mask = None
                self.tip_position = None
                tips = []
                print("Reset - click points again")
            
            elif key == ord('d') and self.skeleton_points:
                # Delete last point
                self.skeleton_points.pop()
                self.current_mask = None
                self.tip_position = None
                tips = []
                print(f"Deleted point. {len(self.skeleton_points)} points remaining")
            
            elif key == ord('q'):
                if tips:
                    # Tips from segmentation - use those
                    break
                elif self.skeleton_points and len(self.skeleton_points) >= 2:
                    # Use last point as tip if segmentation wasn't done
                    print("Using last clicked point as tip (segmentation not done)")
                    last_point = self.skeleton_points[-1]
                    tips = [last_point]
                    break
                elif self.skeleton_points:
                    print("Need at least 2 points. Add more points or press 'S' to segment.")
                else:
                    print("No points added. Quitting...")
                    break
        
        cv2.destroyAllWindows()
        
        if tips:
            # Tips are already in pixel coordinates
            print(f"\n✓ Annotated {len(tips)} instrument(s)")
            for i, tip in enumerate(tips):
                print(f"  Instrument {i+1}: ({tip[0]}, {tip[1]})")
            
            return tips
        elif self.skeleton_points and len(self.skeleton_points) >= 2:
            # Fallback: use last point as tip
            last_point = self.skeleton_points[-1]
            print(f"\n✓ Using last point as tip: ({last_point[0]}, {last_point[1]})")
            return [last_point]
        else:
            print("\n⚠ No tips saved - need at least 2 points")
            return []


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAM Skeleton Annotation')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number (default: 0)')
    parser.add_argument('--model', type=str, default=None, help='Path to SAM model weights')
    
    args = parser.parse_args()
    
    try:
        annotator = SkeletonAnnotator(args.video, args.frame, args.model)
        tips = annotator.run()
        
        if tips:
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
